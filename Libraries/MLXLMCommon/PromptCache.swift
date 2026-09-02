// Copyright © 2026 Apple Inc.

import Foundation

/// Why a cache could not serve a prompt.
///
/// Diagnostic only: nothing branches on it, and every case means the same thing
/// to the caller — build a fresh cache. It exists because a refusal otherwise
/// reaches a log or a trace as a bare zero, and the zeroes mean very different
/// things. ``notTrimmable`` in particular is a *structural* refusal that no
/// amount of prompt tuning removes.
public enum PromptCacheRefusal: String, Equatable, Sendable {
    /// The cache is empty, or the new prompt is too short to leave a token to
    /// feed the model once anything is reused.
    case nothingToReuse

    /// The prompts share fewer leading tokens than `minimumReuse` asked for.
    case belowMinimum

    /// A partial match, and these caches cannot be rewound to it. Every
    /// recurrent hybrid lands here on any prompt that is not a pure extension,
    /// which is every prompt that is not the next hop of the same turn.
    case notTrimmable

    /// The generation asked for a different `maxKVSize`, and cache geometry
    /// depends on it.
    case windowMismatch

    /// `offset == tokens.count` no longer holds, so the cache does not describe
    /// itself truthfully.
    case brokenInvariant

    /// A cache at its ceiling has silently dropped its early tokens, so the
    /// token array no longer describes it.
    case rotated

    /// The rewind trimmed fewer tokens than it was asked for.
    case partialRewind

    /// The cache array mixes window geometries, so no single entry describes
    /// it and the model's mask would be built for the wrong one. See
    /// ``PromptCache/hasMixedWindowGeometry(_:)``.
    case mixedWindowGeometry
}

/// What a ``PromptCache`` can do for a new prompt.
public enum PromptCacheReuse: Equatable, Sendable {
    /// The cache is unusable for this prompt; build a fresh one.
    ///
    /// `commonPrefix` is how far the two prompts did agree. It is what
    /// separates "these prompts have nothing in common" from "they agreed for
    /// 690 tokens and the cache could not be rewound to it".
    case fresh(reason: PromptCacheRefusal, commonPrefix: Int)

    /// The first `prefix` tokens of the new prompt are already in the cache.
    /// Only `newTokens[prefix...]` needs to be prefilled.
    case reuse(prefix: Int)
}

/// Decide how much of `new` a cache holding `cached` can serve.
///
/// Pure, and deliberately so: this is the whole reuse policy, and it is
/// decided on **token ids after templating** rather than on any notion of
/// message identity. A caller that re-renders its conversation differently —
/// a chat template that moves tool schemas, a dropped history turn, a date
/// that rolled over — simply gets a shorter common prefix. The failure mode is
/// "no faster than a cold cache", never a wrong answer.
///
/// - Parameters:
///   - cached: the tokens the cache currently holds, in order
///   - new: the tokens of the prompt about to be run
///   - canTrim: whether every cache in the array is trimmable, i.e. whether
///     the cache can be rewound to a shorter prefix. Attention caches can;
///     recurrent (Mamba-style) state cannot, so those may only be *extended*.
///   - minimumReuse: reuse fewer tokens than this and it is not worth keeping
///     the old buffers alive
public func promptCacheReusePlan(
    cached: [Int], new: [Int], canTrim: Bool, minimumReuse: Int = 1
) -> PromptCacheReuse {
    // `new.count > 1` because of the back-off below: a one-token prompt has
    // nothing left to feed once anything is reused.
    guard !cached.isEmpty, new.count > 1 else {
        return .fresh(reason: .nothingToReuse, commonPrefix: 0)
    }

    var n = 0
    let limit = min(cached.count, new.count)
    while n < limit, cached[n] == new[n] { n += 1 }
    let commonPrefix = n

    // Always leave at least one token for the model to process. A fully cached
    // prompt would otherwise hand `TokenIterator` an empty input and it has
    // nothing to produce logits from.
    n = min(n, new.count - 1)

    guard n >= max(1, minimumReuse) else {
        return .fresh(reason: .belowMinimum, commonPrefix: commonPrefix)
    }

    // Serving less than the cache holds means rewinding it first.
    if n < cached.count, !canTrim {
        return .fresh(reason: .notTrimmable, commonPrefix: commonPrefix)
    }

    return .reuse(prefix: n)
}

/// The part of `text` still needing a prefill once `cachedPrefix` of its tokens
/// are already in the cache.
///
/// The token axis is the **last** one, not the first. VLM and multimodal
/// processors hand tokens over as `[1, N]`, and slicing axis 0 of one of those
/// drops the entire prompt: the suffix comes back empty and the model dies on
/// `reshape` with "cannot infer the shape of an empty array". A 1-D prompt
/// slices identically either way, which is why it took a multimodal text tower
/// to find. Same batch-axis confusion `TokenRing.loadPrompt` was fixed for.
public func promptSuffix(_ text: LMInput.Text, cachedPrefix: Int) -> LMInput.Text {
    guard cachedPrefix > 0 else { return text }
    if text.tokens.ndim == 1 {
        return .init(
            tokens: text.tokens[cachedPrefix...],
            mask: text.mask.map { $0[cachedPrefix...] })
    }
    return .init(
        tokens: text.tokens[.ellipsis, cachedPrefix...],
        mask: text.mask.map { $0[.ellipsis, cachedPrefix...] })
}

/// A KV cache plus the exact token sequence that produced it.
///
/// The invariant that makes reuse safe is `caches[i].offset == tokens.count`
/// for every `i`. Every entry point either establishes it or refuses.
///
/// Two ways to reach that invariant after a generation, because the generated
/// token ids are not surfaced by the generate loop:
///
///  * ``retaining(promptTokens:caches:maxKVSize:)`` rewinds the cache past the
///    generated tail. Free, and available whenever the caches are trimmable —
///    which is every pure-attention model.
///  * ``snapshotting(promptTokens:caches:maxKVSize:)`` copies the caches while
///    they still hold exactly the prompt, i.e. immediately after prefill and
///    before the first token is decoded. Works for every architecture,
///    including recurrent hybrids whose state cannot be rewound, at the cost
///    of one duplicate of the cache.
///
/// Not `Sendable`: `KVCache` holds `MLXArray`, and two generations must never
/// share one of these. Callers hand it to exactly one generation at a time.
public final class PromptCache {

    /// The tokens this cache holds, in order.
    public private(set) var tokens: [Int]

    /// One cache per layer.
    public private(set) var caches: [KVCache]

    /// The `maxKVSize` the caches were built with. Cache geometry depends on
    /// it, so a generation asking for a different one cannot reuse these.
    public let maxKVSize: Int?

    public init(tokens: [Int], caches: [KVCache], maxKVSize: Int?) {
        self.tokens = tokens
        self.caches = caches
        self.maxKVSize = maxKVSize
    }

    public var tokenCount: Int { tokens.count }

    /// How many tokens a cache array holds.
    ///
    /// The **maximum** offset, not `caches.first.offset`. A hybrid model's
    /// array mixes attention caches, which count tokens, with recurrent ones,
    /// which carry a fixed-size state and never track an offset at all —
    /// `NemotronH.newCache` returns `[MambaCache, ...]` and the model never
    /// touches `offset`, so the first entry reports 0 for the whole run.
    /// Reading it made every hybrid refuse reuse silently: measured as a flat
    /// 0 % cache hit on Nemotron 3 Nano 4B across two devices, on the models
    /// where prefill is the *largest* share of a turn.
    public static func tokenCount(of caches: [KVCache]) -> Int {
        caches.reduce(0) { max($0, $1.offset) }
    }

    /// Whether these caches must be copied to be retained, rather than rewound.
    public static func requiresSnapshot(_ caches: [KVCache]) -> Bool {
        !canTrimPromptCache(caches)
    }

    /// Whether any cache has reached its ceiling and started rotating.
    ///
    /// A rotated cache has silently dropped everything but its first few
    /// tokens, so the token array no longer describes it. Reuse is refused
    /// rather than trusted.
    public static func hasRotated(_ caches: [KVCache]) -> Bool {
        caches.contains { cache in
            guard let maxSize = cache.maxSize else { return false }
            return cache.offset >= maxSize
        }
    }

    /// Whether one entry of this array can speak for all of them.
    ///
    /// The library derives a prompt's attention mask from a *single* cache:
    /// `createAttentionMask(h:cache:)` takes one, and a model with interleaved
    /// window sizes hand-picks an index per geometry — Gemma 3n reads
    /// `cacheArray[firstFullIdx]` and `cacheArray[firstSlidingIdx]`, Mistral 3
    /// reads `cache?[faIdx]` and `cache?[swaIdx]`. That is sound while every
    /// layer enters a forward pass at the same offset, which is true of a cold
    /// prefill and false the moment a prefix is reused: the mask is built for
    /// one geometry while the keys come from another, and MLX fails the
    /// broadcast rather than the answer.
    ///
    /// Uniform arrays are unaffected, which is most of them: every layer
    /// rotating at the same size — what `newCache(parameters:)` returns for any
    /// model given a `maxKVSize` — or none of them rotating at all, since a
    /// recurrent hybrid's state reports no ceiling. Only a model that
    /// interleaves *local* and *global* attention in one array is refused.
    ///
    /// Measured: reusing 116 tokens into Gemma 3n E2B produced a (641, 641)
    /// mask against (1, 8, 641, 757) scores and killed the process.
    public static func hasMixedWindowGeometry(_ caches: [KVCache]) -> Bool {
        var geometries = Set<Int>()
        for cache in caches {
            // -1 stands in for "no ceiling", which is itself a geometry.
            geometries.insert(cache.maxSize ?? -1)
            if geometries.count > 1 { return true }
        }
        return false
    }

    /// Rewind `caches` past the generated tail so they hold exactly
    /// `promptTokens`, and retain them.
    ///
    /// Returns nil when the caches cannot be rewound, when they hold fewer
    /// tokens than the prompt, or when they have rotated.
    public static func retaining(
        promptTokens: [Int], caches: [KVCache], maxKVSize: Int?
    ) -> PromptCache? {
        guard !caches.isEmpty, !promptTokens.isEmpty else { return nil }
        guard !hasRotated(caches), canTrimPromptCache(caches) else { return nil }
        // Never adoptable, so retaining it would only cost memory.
        guard !hasMixedWindowGeometry(caches) else { return nil }

        let excess = tokenCount(of: caches) - promptTokens.count
        guard excess >= 0 else { return nil }
        if excess > 0 {
            trimPromptCache(caches, numTokens: excess)
        }
        guard tokenCount(of: caches) == promptTokens.count else { return nil }

        return PromptCache(tokens: promptTokens, caches: caches, maxKVSize: maxKVSize)
    }

    /// Copy `caches` as they stand and retain the copy.
    ///
    /// Must be called while the caches hold exactly `promptTokens` — after
    /// prefill, before any token is decoded. Returns nil if they do not.
    public static func snapshotting(
        promptTokens: [Int], caches: [KVCache], maxKVSize: Int?
    ) -> PromptCache? {
        guard !caches.isEmpty, !promptTokens.isEmpty else { return nil }
        guard !hasRotated(caches) else { return nil }
        guard !hasMixedWindowGeometry(caches) else { return nil }
        guard tokenCount(of: caches) == promptTokens.count else { return nil }

        return PromptCache(
            tokens: promptTokens,
            caches: caches.map { $0.copy() },
            maxKVSize: maxKVSize
        )
    }

    /// Prepare this cache to serve `newTokens`, rewinding it if needed.
    ///
    /// On success `self` describes the cache truthfully again: `tokens` holds
    /// the reused prefix and every cache's offset matches it. On a refusal the
    /// caller should drop this cache and build a fresh one, and
    /// ``PromptCacheAdoption/refusal`` says why — a zero prefix on its own
    /// cannot distinguish "these prompts share nothing" from "they shared 690
    /// tokens and this cache cannot be rewound".
    public func adopt(
        _ newTokens: [Int], maxKVSize: Int?, minimumReuse: Int = 1
    ) -> PromptCacheAdoption {
        let held = tokens.count
        func refuse(
            _ reason: PromptCacheRefusal, commonPrefix: Int = 0
        ) -> PromptCacheAdoption {
            PromptCacheAdoption(
                prefix: 0, cachedTokens: held,
                commonPrefix: commonPrefix, refusal: reason)
        }

        guard self.maxKVSize == maxKVSize else { return refuse(.windowMismatch) }
        guard !caches.isEmpty, Self.tokenCount(of: caches) == tokens.count else {
            return refuse(.brokenInvariant)
        }
        guard !Self.hasRotated(caches) else { return refuse(.rotated) }
        guard !Self.hasMixedWindowGeometry(caches) else {
            return refuse(.mixedWindowGeometry)
        }

        let prefix: Int
        switch promptCacheReusePlan(
            cached: tokens,
            new: newTokens,
            canTrim: canTrimPromptCache(caches),
            minimumReuse: minimumReuse
        ) {
        case .reuse(let p):
            prefix = p
        case .fresh(let reason, let commonPrefix):
            return refuse(reason, commonPrefix: commonPrefix)
        }

        let rewind = tokens.count - prefix
        if rewind > 0 {
            let trimmed = trimPromptCache(caches, numTokens: rewind)
            guard trimmed == rewind else {
                // Partially rewound: keep `tokens` describing the caches
                // truthfully so nothing downstream trusts a stale count, and
                // refuse the reuse.
                tokens = Array(tokens[..<(tokens.count - trimmed)])
                return refuse(.partialRewind, commonPrefix: prefix)
            }
        }
        tokens = Array(newTokens[..<prefix])
        return PromptCacheAdoption(
            prefix: prefix, cachedTokens: held,
            commonPrefix: prefix, refusal: nil)
    }
}

/// What ``PromptCache/adopt(_:maxKVSize:minimumReuse:)`` did, and why.
///
/// Every field is here so a caller can report a refusal without inferring it.
/// `prefix == 0` alone is three different situations — no cache was offered,
/// the prompts diverged immediately, or a partial match could not be rewound to
/// — and telling them apart is the difference between a prompt worth changing
/// and a model that will never reuse across turns.
public struct PromptCacheAdoption: Equatable, Sendable {
    /// Leading tokens of the new prompt the caller may skip prefilling. Zero
    /// when the cache was refused.
    public let prefix: Int

    /// How many tokens the cache held when it was offered this prompt.
    public let cachedTokens: Int

    /// How far the two token sequences agreed. On success this equals
    /// ``prefix``, except in the rare case where the whole new prompt was
    /// already cached and one token had to be left over to feed the model.
    public let commonPrefix: Int

    /// Why the cache was refused; nil when it was not.
    public let refusal: PromptCacheRefusal?

    public init(
        prefix: Int, cachedTokens: Int, commonPrefix: Int,
        refusal: PromptCacheRefusal?
    ) {
        self.prefix = prefix
        self.cachedTokens = cachedTokens
        self.commonPrefix = commonPrefix
        self.refusal = refusal
    }

    public var didReuse: Bool { prefix > 0 }
}

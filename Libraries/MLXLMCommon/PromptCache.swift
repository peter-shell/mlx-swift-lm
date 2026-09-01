// Copyright © 2026 Apple Inc.

import Foundation

/// What a ``PromptCache`` can do for a new prompt.
public enum PromptCacheReuse: Equatable, Sendable {
    /// The cache is unusable for this prompt; build a fresh one.
    case fresh

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
    guard !cached.isEmpty, new.count > 1 else { return .fresh }

    var n = 0
    let limit = min(cached.count, new.count)
    while n < limit, cached[n] == new[n] { n += 1 }

    // Always leave at least one token for the model to process. A fully cached
    // prompt would otherwise hand `TokenIterator` an empty input and it has
    // nothing to produce logits from.
    n = min(n, new.count - 1)

    guard n >= max(1, minimumReuse) else { return .fresh }

    // Serving less than the cache holds means rewinding it first.
    if n < cached.count, !canTrim { return .fresh }

    return .reuse(prefix: n)
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

    /// Rewind `caches` past the generated tail so they hold exactly
    /// `promptTokens`, and retain them.
    ///
    /// Returns nil when the caches cannot be rewound, when they hold fewer
    /// tokens than the prompt, or when they have rotated.
    public static func retaining(
        promptTokens: [Int], caches: [KVCache], maxKVSize: Int?
    ) -> PromptCache? {
        guard let first = caches.first, !promptTokens.isEmpty else { return nil }
        guard !hasRotated(caches), canTrimPromptCache(caches) else { return nil }

        let excess = first.offset - promptTokens.count
        guard excess >= 0 else { return nil }
        if excess > 0 {
            trimPromptCache(caches, numTokens: excess)
        }
        guard first.offset == promptTokens.count else { return nil }

        return PromptCache(tokens: promptTokens, caches: caches, maxKVSize: maxKVSize)
    }

    /// Copy `caches` as they stand and retain the copy.
    ///
    /// Must be called while the caches hold exactly `promptTokens` — after
    /// prefill, before any token is decoded. Returns nil if they do not.
    public static func snapshotting(
        promptTokens: [Int], caches: [KVCache], maxKVSize: Int?
    ) -> PromptCache? {
        guard let first = caches.first, !promptTokens.isEmpty else { return nil }
        guard !hasRotated(caches) else { return nil }
        guard first.offset == promptTokens.count else { return nil }

        return PromptCache(
            tokens: promptTokens,
            caches: caches.map { $0.copy() },
            maxKVSize: maxKVSize
        )
    }

    /// Prepare this cache to serve `newTokens`, rewinding it if needed.
    ///
    /// Returns the number of leading tokens of `newTokens` the caller may skip
    /// prefilling, or nil when the cache cannot serve this prompt at all — in
    /// which case the caller should drop it and build a fresh one.
    ///
    /// On success `self` describes the cache truthfully again: `tokens` holds
    /// the reused prefix and every cache's offset matches it.
    public func adopt(_ newTokens: [Int], maxKVSize: Int?, minimumReuse: Int = 1) -> Int? {
        guard self.maxKVSize == maxKVSize else { return nil }
        guard let first = caches.first, first.offset == tokens.count else { return nil }
        guard !Self.hasRotated(caches) else { return nil }

        let plan = promptCacheReusePlan(
            cached: tokens,
            new: newTokens,
            canTrim: canTrimPromptCache(caches),
            minimumReuse: minimumReuse
        )
        guard case .reuse(let prefix) = plan else { return nil }

        let rewind = tokens.count - prefix
        if rewind > 0 {
            let trimmed = trimPromptCache(caches, numTokens: rewind)
            guard trimmed == rewind else {
                // Partially rewound: keep `tokens` describing the caches
                // truthfully so nothing downstream trusts a stale count, and
                // refuse the reuse.
                tokens = Array(tokens[..<(tokens.count - trimmed)])
                return nil
            }
        }
        tokens = Array(newTokens[..<prefix])
        return prefix
    }
}

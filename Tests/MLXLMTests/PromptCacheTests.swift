import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

// MARK: - helpers

private func feed(_ caches: [KVCache], tokens: Int) {
    let keys = MLXArray.ones([1, 8, tokens, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, tokens, 64], dtype: .bfloat16)
    for cache in caches {
        _ = cache.update(keys: keys, values: values)
    }
}

private func simpleCaches(_ count: Int, tokens: Int) -> [KVCache] {
    let caches: [KVCache] = (0 ..< count).map { _ in KVCacheSimple() }
    feed(caches, tokens: tokens)
    return caches
}

// MARK: - the reuse policy, no MLX involved

@Test
func testPlanPureExtension() {
    let plan = promptCacheReusePlan(
        cached: [1, 2, 3, 4], new: [1, 2, 3, 4, 5, 6], canTrim: false)
    #expect(plan == .reuse(prefix: 4))
}

@Test
func testPlanIdenticalPromptLeavesOneTokenToFeed() {
    // Nothing left to prefill would hand the iterator an empty input.
    let plan = promptCacheReusePlan(
        cached: [1, 2, 3, 4], new: [1, 2, 3, 4], canTrim: true)
    #expect(plan == .reuse(prefix: 3))
}

@Test
func testPlanDivergenceNeedsTrim() {
    let cached = [1, 2, 3, 4, 5]
    let new = [1, 2, 3, 9, 9, 9]
    #expect(promptCacheReusePlan(cached: cached, new: new, canTrim: true) == .reuse(prefix: 3))
    // Recurrent state cannot be rewound, so a divergence is a cold start —
    // and the plan still reports how far the prompts did agree, which is the
    // number that says this was a refusal and not a divergence at token 0.
    #expect(
        promptCacheReusePlan(cached: cached, new: new, canTrim: false)
            == .fresh(reason: .notTrimmable, commonPrefix: 3))
}

@Test
func testPlanRefusesWhenNothingShared() {
    #expect(
        promptCacheReusePlan(cached: [1, 2, 3], new: [9, 9, 9], canTrim: true)
            == .fresh(reason: .belowMinimum, commonPrefix: 0))
}

@Test
func testPlanRefusesDegenerateInputs() {
    let nothing = PromptCacheReuse.fresh(reason: .nothingToReuse, commonPrefix: 0)
    #expect(promptCacheReusePlan(cached: [], new: [1, 2, 3], canTrim: true) == nothing)
    #expect(promptCacheReusePlan(cached: [1, 2, 3], new: [1], canTrim: true) == nothing)
    #expect(promptCacheReusePlan(cached: [1, 2, 3], new: [], canTrim: true) == nothing)
}

@Test
func testPlanHonoursMinimumReuse() {
    let cached = [1, 2, 3, 4, 5]
    let new = [1, 2, 9, 9, 9, 9]
    #expect(
        promptCacheReusePlan(cached: cached, new: new, canTrim: true, minimumReuse: 2)
            == .reuse(prefix: 2))
    #expect(
        promptCacheReusePlan(cached: cached, new: new, canTrim: true, minimumReuse: 3)
            == .fresh(reason: .belowMinimum, commonPrefix: 2))
}

// MARK: - trimPromptCache trims every layer

@Test
func testTrimPromptCacheTrimsEveryLayer() {
    let caches = simpleCaches(4, tokens: 10)
    #expect(caches.allSatisfy { $0.offset == 10 })

    let trimmed = trimPromptCache(caches, numTokens: 4)

    #expect(trimmed == 4)
    // Trimming only `cache.first` would leave layers 1..3 at 10 and corrupt
    // the model silently.
    #expect(caches.allSatisfy { $0.offset == 6 })
}

// MARK: - retaining: rewind past the generated tail

@Test
func testRetainingRewindsGeneratedTokens() throws {
    // 6 prompt tokens, then 4 decoded.
    let caches = simpleCaches(3, tokens: 10)
    let prompt = [11, 12, 13, 14, 15, 16]

    let retained = try #require(
        PromptCache.retaining(promptTokens: prompt, caches: caches, maxKVSize: nil))

    #expect(retained.tokenCount == 6)
    #expect(retained.caches.allSatisfy { $0.offset == 6 })
}

@Test
func testRetainingRefusesNonTrimmableCaches() {
    let mamba = MambaCache()
    mamba.offset = 10
    #expect(
        PromptCache.retaining(
            promptTokens: [1, 2, 3, 4, 5, 6], caches: [mamba], maxKVSize: nil) == nil)
}

@Test
func testRetainingRefusesWhenCacheHoldsLessThanThePrompt() {
    let caches = simpleCaches(2, tokens: 4)
    #expect(
        PromptCache.retaining(
            promptTokens: [1, 2, 3, 4, 5, 6], caches: caches, maxKVSize: nil) == nil)
}

// MARK: - snapshotting: the path for caches that cannot be rewound

@Test
func testSnapshotIsIndependentOfContinuedGeneration() throws {
    let caches = simpleCaches(2, tokens: 6)
    let prompt = [1, 2, 3, 4, 5, 6]

    let snapshot = try #require(
        PromptCache.snapshotting(promptTokens: prompt, caches: caches, maxKVSize: nil))

    // keep generating into the live caches
    feed(caches, tokens: 3)

    #expect(caches.allSatisfy { $0.offset == 9 })
    #expect(snapshot.caches.allSatisfy { $0.offset == 6 })
    #expect(snapshot.tokenCount == 6)
}

@Test
func testSnapshotRefusesWhenOffsetDoesNotMatchThePrompt() {
    let caches = simpleCaches(2, tokens: 9)
    #expect(
        PromptCache.snapshotting(
            promptTokens: [1, 2, 3, 4, 5, 6], caches: caches, maxKVSize: nil) == nil)
}

@Test
func testRequiresSnapshotFollowsTrimmability() {
    #expect(PromptCache.requiresSnapshot([KVCacheSimple()]) == false)
    #expect(PromptCache.requiresSnapshot([MambaCache()]) == true)
    // A hybrid is only as rewindable as its least rewindable half.
    #expect(PromptCache.requiresSnapshot([CacheList(KVCacheSimple(), MambaCache())]) == true)
}

// MARK: - adopt

@Test
func testAdoptExtensionKeepsTheWholeCache() throws {
    let caches = simpleCaches(2, tokens: 4)
    let cache = PromptCache(tokens: [1, 2, 3, 4], caches: caches, maxKVSize: nil)

    let adoption = cache.adopt([1, 2, 3, 4, 5, 6], maxKVSize: nil)

    #expect(adoption.prefix == 4)
    #expect(adoption.refusal == nil)
    #expect(adoption.cachedTokens == 4)
    #expect(cache.tokens == [1, 2, 3, 4])
    #expect(caches.allSatisfy { $0.offset == 4 })
}

@Test
func testAdoptDivergenceRewindsToTheCommonPrefix() throws {
    let caches = simpleCaches(2, tokens: 5)
    let cache = PromptCache(tokens: [1, 2, 3, 4, 5], caches: caches, maxKVSize: nil)

    let adoption = cache.adopt([1, 2, 3, 9, 9, 9], maxKVSize: nil)

    #expect(adoption.prefix == 3)
    #expect(adoption.refusal == nil)
    #expect(cache.tokens == [1, 2, 3])
    #expect(caches.allSatisfy { $0.offset == 3 })
}

@Test
func testAdoptRefusesOnDifferentMaxKVSize() {
    let caches = simpleCaches(1, tokens: 4)
    let cache = PromptCache(tokens: [1, 2, 3, 4], caches: caches, maxKVSize: 4096)
    // Cache geometry is fixed by maxKVSize, so a hop asking for a different
    // window cannot share these.
    #expect(cache.adopt([1, 2, 3, 4, 5], maxKVSize: 8192).refusal == .windowMismatch)
}

@Test
func testAdoptRefusesWhenTheCacheHasRotated() {
    let rotating = RotatingKVCache(maxSize: 8)
    feed([rotating], tokens: 10)

    #expect(PromptCache.hasRotated([rotating]))

    let cache = PromptCache(
        tokens: Array(1 ... 10), caches: [rotating], maxKVSize: 8)
    // A rotated cache has silently dropped its early tokens, so the token
    // array no longer describes it.
    #expect(cache.adopt(Array(1 ... 12), maxKVSize: 8).refusal == .rotated)
}

@Test
func testAdoptRefusesWhenOffsetDisagreesWithTokens() {
    let caches = simpleCaches(1, tokens: 7)
    let cache = PromptCache(tokens: [1, 2, 3, 4], caches: caches, maxKVSize: nil)
    #expect(cache.adopt([1, 2, 3, 4, 5], maxKVSize: nil).refusal == .brokenInvariant)
}

// MARK: - equivalence: a reused prefix must generate what a cold cache does

private func equivalenceModel() -> LlamaModel {
    let config = LlamaConfiguration(
        hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128, attentionHeads: 8,
        rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4)
    return LlamaModel(config)
}

private func generate(
    _ tokens: [Int], model: LlamaModel, cache: [KVCache]?, cachedPrefix: Int,
    parameters: GenerateParameters, count: Int
) throws -> [Int] {
    let input = LMInput(tokens: MLXArray(tokens))
    var iterator: TokenIterator
    if let cache {
        iterator = try TokenIterator(
            input: input, model: model, cache: cache, cachedPrefix: cachedPrefix,
            parameters: parameters)
    } else {
        iterator = try TokenIterator(
            input: input, model: model, cache: nil, parameters: parameters)
    }
    return Array(iterator.prefix(count))
}

/// The property the whole feature rests on: prefilling `[prefix + suffix]` into
/// a cold cache and prefilling `suffix` into a cache already holding `prefix`
/// must put the model in the same state.
@Test
func testReusedPrefixGeneratesTheSameTokensAsAColdCache() throws {
    let model = equivalenceModel()
    // maxKVSize non-nil, which is what production always passes -- so this
    // exercises RotatingKVCache, not KVCacheSimple.
    let parameters = GenerateParameters(maxKVSize: 512, temperature: 0)

    let prompt = (0 ..< 24).map { ($0 * 7 + 3) % 97 }
    let prefixLength = 16

    let cold = try generate(
        prompt, model: model, cache: nil, cachedPrefix: 0,
        parameters: parameters, count: 6)

    // Build a cache holding exactly the first 16 tokens.
    let warmup = try TokenIterator(
        input: LMInput(tokens: MLXArray(Array(prompt[..<prefixLength]))),
        model: model, cache: nil, parameters: parameters)
    let retained = try #require(
        PromptCache.retaining(
            promptTokens: Array(prompt[..<prefixLength]),
            caches: warmup.currentCache,
            maxKVSize: parameters.maxKVSize))

    let reused = retained.adopt(prompt, maxKVSize: parameters.maxKVSize).prefix
    #expect(reused == prefixLength)

    let warm = try generate(
        prompt, model: model, cache: retained.caches, cachedPrefix: reused,
        parameters: parameters, count: 6)

    #expect(warm == cold)
}

/// A `[1, N]` prompt -- how VLM and multimodal processors hand tokens over --
/// must lose its prefix from the token axis, not from the batch axis.
///
/// Slicing axis 0 of one of those leaves nothing at all, and the model dies on
/// `reshape` rather than generating. Mistral 3 is such a text tower and it
/// crashed on the second turn of every conversation. A 1-D prompt slices the
/// same either way, so nothing else here catches it.
@Test
func testPromptSuffixSlicesTheTokenAxisNotTheBatchAxis() {
    let batched = LMInput.Text(tokens: MLXArray(0 ..< 24).reshaped([1, 24]))
    let suffix = promptSuffix(batched, cachedPrefix: 16)

    #expect(suffix.tokens.shape == [1, 8])
    #expect(suffix.tokens.asArray(Int.self) == Array(16 ..< 24))

    // 1-D prompts are unchanged, which is every text-only model.
    let flat = LMInput.Text(tokens: MLXArray(0 ..< 24))
    #expect(promptSuffix(flat, cachedPrefix: 16).tokens.shape == [8])

    // A mask, when there is one, follows the tokens.
    let masked = LMInput.Text(
        tokens: MLXArray(0 ..< 24).reshaped([1, 24]),
        mask: MLXArray.ones([1, 24]))
    #expect(promptSuffix(masked, cachedPrefix: 16).mask?.shape == [1, 8])

    // And nothing is sliced when nothing was cached.
    #expect(promptSuffix(batched, cachedPrefix: 0).tokens.shape == [1, 24])
}

/// The same property across a rewind: a cache holding one prompt, trimmed back
/// to the common prefix, must serve a diverging prompt correctly.
@Test
func testRewoundCacheGeneratesTheSameTokensAsAColdCache() throws {
    let model = equivalenceModel()
    let parameters = GenerateParameters(maxKVSize: 512, temperature: 0)

    let first = (0 ..< 24).map { ($0 * 7 + 3) % 97 }
    // shares the first 16 tokens, then diverges -- the shape a tool loop
    // produces when `shedToFit` drops an exchange.
    var second = Array(first[..<16])
    second.append(contentsOf: (0 ..< 10).map { ($0 * 11 + 5) % 97 })

    let cold = try generate(
        second, model: model, cache: nil, cachedPrefix: 0,
        parameters: parameters, count: 6)

    let warmup = try TokenIterator(
        input: LMInput(tokens: MLXArray(first)), model: model, cache: nil,
        parameters: parameters)
    let retained = try #require(
        PromptCache.retaining(
            promptTokens: first, caches: warmup.currentCache,
            maxKVSize: parameters.maxKVSize))

    let reused = retained.adopt(second, maxKVSize: parameters.maxKVSize).prefix
    #expect(reused == 16)

    let warm = try generate(
        second, model: model, cache: retained.caches, cachedPrefix: reused,
        parameters: parameters, count: 6)

    #expect(warm == cold)
}

/// A snapshot taken after prefill is as good as a rewind -- this is the path
/// recurrent hybrids take, exercised here on a cache that could also rewind so
/// the two are directly comparable.
@Test
func testSnapshottedPrefixGeneratesTheSameTokensAsAColdCache() throws {
    let model = equivalenceModel()
    let parameters = GenerateParameters(maxKVSize: 512, temperature: 0)

    let prompt = (0 ..< 24).map { ($0 * 7 + 3) % 97 }
    let prefixLength = 16

    let cold = try generate(
        prompt, model: model, cache: nil, cachedPrefix: 0,
        parameters: parameters, count: 6)

    var warmup = try TokenIterator(
        input: LMInput(tokens: MLXArray(Array(prompt[..<prefixLength]))),
        model: model, cache: nil, parameters: parameters)
    let snapshot = try #require(
        PromptCache.snapshotting(
            promptTokens: Array(prompt[..<prefixLength]),
            caches: warmup.currentCache,
            maxKVSize: parameters.maxKVSize))
    // keep decoding into the original caches; the snapshot must not follow
    _ = warmup.next()
    _ = warmup.next()

    let reused = snapshot.adopt(prompt, maxKVSize: parameters.maxKVSize).prefix
    let warm = try generate(
        prompt, model: model, cache: snapshot.caches, cachedPrefix: reused,
        parameters: parameters, count: 6)

    #expect(warm == cold)
}

// MARK: - hybrid cache arrays

/// `NemotronH.newCache` returns a flat mix of `MambaCache` and `KVCacheSimple`
/// with a **Mamba layer first**, and the model never touches `MambaCache.offset`
/// — it reports 0 for the whole run. Anything reading `caches.first.offset` to
/// learn how many tokens a cache holds therefore sees 0 and silently refuses
/// every reuse, which is what a 0 % cache-hit rate on Nemotron 3 Nano 4B across
/// two devices turned out to be.
private func nemotronShapedCaches(tokens: Int) -> [KVCache] {
    let caches: [KVCache] = [
        MambaCache(), KVCacheSimple(), MambaCache(), KVCacheSimple(),
    ]
    let keys = MLXArray.ones([1, 8, tokens, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, tokens, 64], dtype: .bfloat16)
    for cache in caches {
        switch cache {
        case let arrays as ArraysCache:
            // recurrent state: fixed size, no offset tracking
            arrays[0] = keys
            arrays[1] = values
        default:
            _ = cache.update(keys: keys, values: values)
        }
    }
    return caches
}

@Test
func testTokenCountIgnoresCachesThatDoNotTrackAnOffset() {
    let caches = nemotronShapedCaches(tokens: 12)

    #expect(caches.first!.offset == 0, "the Mamba cache tracks no offset")
    #expect(PromptCache.tokenCount(of: caches) == 12)
}

@Test
func testHybridCacheIsRetainedBySnapshot() throws {
    let caches = nemotronShapedCaches(tokens: 6)
    let prompt = [1, 2, 3, 4, 5, 6]

    #expect(PromptCache.requiresSnapshot(caches), "recurrent state cannot rewind")

    let snapshot = try #require(
        PromptCache.snapshotting(
            promptTokens: prompt, caches: caches, maxKVSize: nil))
    let extension_ = snapshot.adopt([1, 2, 3, 4, 5, 6, 7, 8], maxKVSize: nil)

    #expect(extension_.prefix == 6)
    #expect(extension_.refusal == nil)
}

/// Gemma 3n / Mistral 3 shape: local sliding layers interleaved with global
/// ones, so the array holds two different window geometries at once.
private func interleavedSlidingCaches(window: Int, tokens: Int) -> [KVCache] {
    let caches: [KVCache] = [
        RotatingKVCache(maxSize: window, keep: 0),
        RotatingKVCache(maxSize: window, keep: 0),
        KVCacheSimple(),
    ]
    feed(caches, tokens: tokens)
    return caches
}

@Test
func testMixedWindowGeometryIsRefused() {
    // A model that interleaves local and global attention builds one mask per
    // geometry from a single hand-picked cache index. That holds while every
    // layer starts a forward pass at the same offset -- which a cold prefill
    // guarantees and a reused prefix does not. Reusing 116 tokens into Gemma 3n
    // E2B built a (641, 641) mask against (1, 8, 641, 757) scores and killed
    // the process, so the array is refused before it can.
    let mixed = interleavedSlidingCaches(window: 512, tokens: 120)
    #expect(PromptCache.hasMixedWindowGeometry(mixed))

    let cache = PromptCache(
        tokens: Array(0 ..< 120), caches: mixed, maxKVSize: nil)
    let adoption = cache.adopt(Array(0 ..< 200), maxKVSize: nil)

    #expect(adoption.prefix == 0)
    #expect(adoption.refusal == .mixedWindowGeometry)

    // And it is never retained in the first place, so it costs no memory
    // either.
    #expect(
        PromptCache.retaining(
            promptTokens: Array(0 ..< 120), caches: mixed, maxKVSize: nil) == nil)
    #expect(
        PromptCache.snapshotting(
            promptTokens: Array(0 ..< 120), caches: mixed, maxKVSize: nil) == nil)
}

@Test
func testUniformGeometryIsStillReused() {
    // The guard must not catch the ordinary cases. Every layer rotating at one
    // size is what `newCache` returns for any model given a maxKVSize; no layer
    // rotating at all is a plain attention model, and a recurrent hybrid whose
    // state reports no ceiling looks the same.
    let allRotating = (0 ..< 3).map { _ in RotatingKVCache(maxSize: 512, keep: 0) }
    #expect(!PromptCache.hasMixedWindowGeometry(allRotating))
    #expect(!PromptCache.hasMixedWindowGeometry(simpleCaches(3, tokens: 4)))
    #expect(!PromptCache.hasMixedWindowGeometry(nemotronShapedCaches(tokens: 4)))

    // ... and reuse still happens on one of them.
    let caches = simpleCaches(2, tokens: 4)
    let cache = PromptCache(tokens: [1, 2, 3, 4], caches: caches, maxKVSize: nil)
    #expect(cache.adopt([1, 2, 3, 4, 5, 6], maxKVSize: nil).prefix == 4)
}

@Test
func testHybridRefusesAnythingButAPureExtension() throws {
    // The structural ceiling on every recurrent hybrid, and the reason a
    // Nemotron trace shows within-turn reuse and a flat zero across turns: a
    // tool loop appends, so it extends; a new turn rewrites the middle of the
    // prompt, and there is no rewinding a Mamba state to meet it.
    let caches = nemotronShapedCaches(tokens: 6)
    let snapshot = try #require(
        PromptCache.snapshotting(
            promptTokens: [1, 2, 3, 4, 5, 6], caches: caches, maxKVSize: nil))

    let adoption = snapshot.adopt([1, 2, 3, 9, 9, 9, 9], maxKVSize: nil)

    #expect(adoption.prefix == 0)
    #expect(adoption.refusal == .notTrimmable)
    // The prompts agreed for three tokens; the cache simply could not be
    // rewound to them. Without this a caller sees only a zero and cannot tell
    // the two apart.
    #expect(adoption.commonPrefix == 3)
    #expect(adoption.cachedTokens == 6)
}

// MARK: - LFM: conv state, sliding windows, and the geometry guard

/// LFM2 shape: conv state beside full attention rotating at `maxKVSize`.
private func lfm2ShapedCaches(maxKVSize: Int, tokens: Int) -> [KVCache] {
    let caches: [KVCache] = [
        MambaCache(), RotatingKVCache(maxSize: maxKVSize, keep: 4),
        MambaCache(), RotatingKVCache(maxSize: maxKVSize, keep: 4),
    ]
    let keys = MLXArray.ones([1, 8, tokens, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, tokens, 64], dtype: .bfloat16)
    for cache in caches {
        switch cache {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        default:
            _ = cache.update(keys: keys, values: values)
        }
    }
    return caches
}

@Test
func testRecurrentStateIsNotAGeometry() throws {
    // A MambaCache takes no mask and reports no ceiling. Counting it as a
    // geometry made every LFM array read as mixed once its attention began
    // rotating at maxKVSize, and LFM2.5 lost the snapshot reuse it had.
    let caches = lfm2ShapedCaches(maxKVSize: 512, tokens: 6)
    #expect(!PromptCache.hasMixedWindowGeometry(caches))

    let snapshot = try #require(
        PromptCache.snapshotting(
            promptTokens: [1, 2, 3, 4, 5, 6], caches: caches, maxKVSize: 512))
    #expect(snapshot.adopt([1, 2, 3, 4, 5, 6, 7], maxKVSize: 512).prefix == 6)
}

@Test
func testInterleavedAttentionNeedsTheOptIn() {
    // LFM3 shape: conv, a sliding window, and full attention at the ceiling.
    // Two attention geometries remain once conv state is set aside, so the
    // array is refused unless the model says it masks each one from its own
    // cache.
    let caches: [KVCache] = [
        MambaCache(), RotatingKVCache(maxSize: 16, keep: 0),
        RotatingKVCache(maxSize: 512, keep: 4),
    ]
    feed(Array(caches[1...]), tokens: 6)
    let prompt = [1, 2, 3, 4, 5, 6]

    #expect(PromptCache.hasMixedWindowGeometry(caches))
    #expect(
        PromptCache.snapshotting(promptTokens: prompt, caches: caches, maxKVSize: 512)
            == nil)
    #expect(
        PromptCache(tokens: prompt, caches: caches, maxKVSize: 512)
            .adopt(prompt + [7], maxKVSize: 512).refusal == .mixedWindowGeometry)

    let snapshot = PromptCache.snapshotting(
        promptTokens: prompt, caches: caches, maxKVSize: 512,
        allowsMixedGeometry: true)
    #expect(snapshot != nil)
    #expect(
        snapshot?.adopt(prompt + [7], maxKVSize: 512, allowsMixedGeometry: true).prefix
            == 6)
}

@Test
func testSlidingWindowRotationBelowTheCeilingIsNotLoss() {
    // A window the model chose rotates by design; attention there never reads
    // past it. Only rotation at the requested ceiling drops tokens a layer
    // would have read.
    let window = RotatingKVCache(maxSize: 16, keep: 0)
    feed([window], tokens: 40)
    #expect(PromptCache.hasRotated([window]))
    #expect(!PromptCache.hasRotated([window], maxKVSize: 512))
    #expect(!window.isTrimmable, "a rotated window can only be extended")

    let ceiling = RotatingKVCache(maxSize: 32, keep: 4)
    feed([ceiling], tokens: 40)
    #expect(PromptCache.hasRotated([ceiling], maxKVSize: 32))

    // A window clamped to the ceiling is indistinguishable from the ceiling,
    // so it is refused.
    #expect(PromptCache.hasRotated([window], maxKVSize: 16))
}

private func lfm2EquivalenceModel() throws -> LFM2Model {
    let json = """
        {
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "norm_eps": 1e-5,
            "block_ff_dim": 64,
            "block_auto_adjust_ff_dim": false,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"]
        }
        """
    return LFM2Model(
        try JSONDecoder().decode(LFM2Configuration.self, from: Data(json.utf8)))
}

private func lfm3EquivalenceModel(slidingWindow: Int) throws -> LFM3Model {
    let json = """
        {
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "norm_eps": 1e-5,
            "layer_types": ["conv", "full_attention", "conv", "sliding_attention",
                            "conv", "sliding_attention"],
            "sliding_window": \(slidingWindow),
            "rope_parameters": {
                "full_attention": {"rope_theta": 10000.0},
                "sliding_attention": {"rope_theta": 10000.0}
            }
        }
        """
    return LFM3Model(
        try JSONDecoder().decode(LFM3Configuration.self, from: Data(json.utf8)))
}

/// Snapshot a hybrid after `prefixLength` tokens, extend it to `prompt`, and
/// compare every suffix position's logits with one cold forward pass — what a
/// tool hop does, end to end.
///
/// Logits, not generated tokens. On a tiny random model the argmax of the last
/// position survives a wrong mask: a sliding mask built from a nil cache, which
/// lets each query read past its window, still generated the cold tokens.
private func snapshotExtensionDivergence(
    model: any LLMModel, prompt: [Int], prefixLength: Int,
    allowsMixedGeometry: Bool
) throws -> (divergence: Float, reused: Int) {
    let parameters = GenerateParameters(maxKVSize: 512, temperature: 0)
    let cold = model(
        MLXArray(prompt)[.newAxis], cache: model.newCache(parameters: parameters))

    let caches = model.newCache(parameters: parameters)
    let prefix = Array(prompt[..<prefixLength])
    eval(model(MLXArray(prefix)[.newAxis], cache: caches))
    #expect(PromptCache.requiresSnapshot(caches))
    let snapshot = try #require(
        PromptCache.snapshotting(
            promptTokens: prefix, caches: caches, maxKVSize: parameters.maxKVSize,
            allowsMixedGeometry: allowsMixedGeometry))
    // keep running the original caches; the snapshot must not follow
    eval(model(MLXArray([1, 2])[.newAxis], cache: caches))

    let reused = snapshot.adopt(
        prompt, maxKVSize: parameters.maxKVSize,
        allowsMixedGeometry: allowsMixedGeometry
    ).prefix
    let warm = model(
        MLXArray(Array(prompt[reused...]))[.newAxis], cache: snapshot.caches)

    let divergence = abs(warm - cold[0..., reused..., 0...]).max().item(Float.self)
    return (divergence, reused)
}

@Test
func testLFM2SnapshotExtensionMatchesAColdPrefill() throws {
    let prompt = (0 ..< 24).map { ($0 * 7 + 3) % 97 }
    let result = try snapshotExtensionDivergence(
        model: try lfm2EquivalenceModel(), prompt: prompt, prefixLength: 16,
        allowsMixedGeometry: false)
    #expect(result.reused == 16)
    #expect(result.divergence < 1e-4)
}

/// The test the LFM3 opt-in rests on. The prefix is longer than the sliding
/// window, so those layers have rotated before the snapshot, and the extension
/// starts a forward pass at a non-zero offset with two mask geometries — the
/// conditions under which Gemma 3n built a no-cache mask and died. Breaking
/// LFM3's sliding mask that way fails this test.
@Test
func testLFM3SnapshotExtensionMatchesAColdPrefill() throws {
    let model = try lfm3EquivalenceModel(slidingWindow: 16)
    #expect((model as Any) is MasksEachCacheGeometry)

    let prompt = (0 ..< 48).map { ($0 * 7 + 3) % 97 }
    let result = try snapshotExtensionDivergence(
        model: model, prompt: prompt, prefixLength: 32, allowsMixedGeometry: true)
    #expect(result.reused == 32)
    #expect(result.divergence < 1e-4)
}

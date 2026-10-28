//
//  LFM3.swift
//  mlx-swift-lm
//

// Port of LiquidAI's `modeling_lfm3.py` (transformers remote code shipped with
// LiquidAI/LFM3-1.7B); mlx-lm has no lfm3 model yet.
//
// What differs from LFM2:
// - a third layer type, `sliding_attention`, on a rotating cache with its own
//   RoPE base;
// - a per-head sigmoid output gate (`gate_proj`) applied before `o_proj`;
// - partial RoPE that keeps the full head dimension's frequency progression
//   and zeroes the tail, rather than rotating a shorter prefix;
// - optional sandwich norms on each branch's output;
// - `head_dim` and `intermediate_size` read directly, with no auto-adjust.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct LFM3RopeParameters: Codable, Sendable {
    let ropeType: String
    let ropeTheta: Float
    let partialRotaryFactor: Float

    enum CodingKeys: String, CodingKey {
        case ropeType = "rope_type"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.partialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        guard ropeType == "default" else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeType, in: container,
                debugDescription: "LFM3: unsupported rope_type \(ropeType)")
        }
    }
}

public struct LFM3Configuration: Codable, Sendable {
    let modelType: String
    let vocabularySize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let normEps: Float
    let convBias: Bool
    let convLCache: Int
    let layerTypes: [String]
    let slidingWindow: Int
    let useSandwichNorm: Bool
    let ropeParameters: [String: LFM3RopeParameters]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3"
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        self.normEps = try container.decode(Float.self, forKey: .normEps)
        self.convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        self.convLCache = try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
        self.layerTypes = try container.decode([String].self, forKey: .layerTypes)
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
        self.useSandwichNorm =
            try container.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? false
        self.ropeParameters = try container.decode(
            [String: LFM3RopeParameters].self, forKey: .ropeParameters)
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "LFM3: \(layerTypes.count) layer_types for \(hiddenLayers) layers")
        }
    }

    func isAttention(_ layerIdx: Int) -> Bool { layerTypes[layerIdx] != "conv" }
    func isSliding(_ layerIdx: Int) -> Bool { layerTypes[layerIdx] == "sliding_attention" }

    /// Full attention honors `maxKVSize`. Sliding attention keeps its window,
    /// capped by that ceiling when the ceiling is tighter. Conv layers stay a
    /// fixed-size `MambaCache`.
    public func makeCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< hiddenLayers).map { layerIdx -> KVCache in
            switch layerTypes[layerIdx] {
            case "conv":
                return MambaCache()
            case "sliding_attention":
                let window = slidingWindow
                let maxSize = parameters?.maxKVSize.map { min(window, $0) } ?? window
                return RotatingKVCache(maxSize: maxSize, keep: 0)
            default:
                if let maxKVSize = parameters?.maxKVSize {
                    return RotatingKVCache(maxSize: maxKVSize, keep: 4)
                }
                return KVCacheSimple()
            }
        }
    }
}

/// RoPE with LFM3's zero-tail partial rotation.
///
/// HF builds `head_dim / 2` inverse frequencies, the first
/// `partial_rotary_factor * head_dim / 2` from the full head dimension's
/// progression and the rest zero, and rotates with the half-split convention.
/// So the rotated dimensions are `[0, n)` paired with `[D/2, D/2 + n)` — not
/// the first `2n` dimensions, which is what `RoPE(dimensions: 2n)` would rotate.
/// MLX takes periods (the reciprocal of an inverse frequency), so a zero
/// frequency is an infinite period: angle zero, the dimension left unrotated.
class LFM3RoPE: Module {
    let dims: Int
    let _freqs: MLXArray

    init(dims: Int, base: Float, partialRotaryFactor: Float) {
        self.dims = dims
        let rotated = Int(partialRotaryFactor * Float(dims)) / 2
        var periods = [Float](repeating: .infinity, count: dims / 2)
        for i in 0 ..< rotated {
            periods[i] = pow(base, Float(2 * i) / Float(dims))
        }
        self._freqs = MLXArray(periods)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dims, traditional: false, base: nil, scale: 1.0, offset: offset,
            freqs: _freqs)
    }
}

class LFM3Attention: Module {
    let args: LFM3Configuration
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: LFM3RoPE

    init(_ args: LFM3Configuration, layerIdx: Int) {
        self.args = args
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
        _gateProj.wrappedValue = Linear(dim, heads, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)

        let layerType = args.layerTypes[layerIdx]
        guard let rope = args.ropeParameters[layerType] else {
            fatalError("LFM3: no rope_parameters for \(layerType)")
        }
        self.rope = LFM3RoPE(
            dims: headDim, base: rope.ropeTheta, partialRotaryFactor: rope.partialRotaryFactor)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, args.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)

        // One scalar gate per head, from the same normalized input.
        let gate = sigmoid(gateProj(x)).expandedDimensions(axis: -1)
        return oProj((output * gate).reshaped(B, L, -1))
    }
}

class LFM3ShortConv: Module {
    let hiddenSize: Int
    let lCache: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM3Configuration) {
        self.hiddenSize = args.hiddenSize
        self.lCache = args.convLCache

        _conv.wrappedValue = Conv1d(
            inputChannels: args.hiddenSize,
            outputChannels: args.hiddenSize,
            kernelSize: lCache,
            groups: args.hiddenSize,
            bias: args.convBias
        )
        _inProj.wrappedValue = Linear(args.hiddenSize, 3 * args.hiddenSize, bias: args.convBias)
        _outProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let BCx = inProj(x).split(parts: 3, axis: -1)
        let (B, C, xs) = (BCx[0], BCx[1], BCx[2])
        var Bx = B * xs

        let state =
            cache?[0]
            ?? MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)

        Bx = concatenated([state, Bx], axis: -2)
        if let cache {
            cache[0] = Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...]
        }

        return outProj(C * conv(Bx))
    }
}

class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        _w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        _w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

class LFM3DecoderLayer: Module {
    let isAttentionLayer: Bool

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    init(_ args: LFM3Configuration, layerIdx: Int) {
        self.isAttentionLayer = args.isAttention(layerIdx)

        if isAttentionLayer {
            _attention.wrappedValue = LFM3Attention(args, layerIdx: layerIdx)
        } else {
            _conv.wrappedValue = LFM3ShortConv(args)
        }

        _feedForward.wrappedValue = LFM3MLP(dim: args.hiddenSize, hiddenDim: args.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        if args.useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r: MLXArray
        if isAttentionLayer {
            r = attention!(operatorNorm(x), mask: mask, cache: cache)
        } else {
            r = conv!(operatorNorm(x), cache: cache as? MambaCache)
        }
        if let postOperatorNorm { r = postOperatorNorm(r) }
        let h = x + r

        var f = feedForward(ffnNorm(h))
        if let postFFNNorm { f = postFFNNorm(f) }
        return h + f
    }
}

public class LFM3ModelInner: Module {
    let args: LFM3Configuration
    let firstFullIdx: Int?
    let firstSlidingIdx: Int?

    fileprivate let layers: [LFM3DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ args: LFM3Configuration) {
        self.args = args
        self.firstFullIdx = args.layerTypes.firstIndex(of: "full_attention")
        self.firstSlidingIdx = args.layerTypes.firstIndex(of: "sliding_attention")

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { LFM3DecoderLayer(args, layerIdx: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // One mask per attention geometry, each built from a cache of its own
        // kind (the Gemma 3 pattern); conv layers take no mask.
        let fullMask = createAttentionMask(h: h, cache: firstFullIdx.flatMap { cache?[$0] })
        let slidingMask = createAttentionMask(
            h: h, cache: firstSlidingIdx.flatMap { cache?[$0] }, windowSize: args.slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = args.isSliding(i) ? slidingMask : fullMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return embeddingNorm(h)
    }
}

public class LFM3Model: Module, LLMModel, KVCacheDimensionProvider, MasksEachCacheGeometry {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LFM3ModelInner
    let configuration: LFM3Configuration

    public init(_ args: LFM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { args.isAttention($0) ? args.kvHeads : 0 }
        self.model = LFM3ModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.embedTokens.asLinear(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // LiquidAI's MLX builds are converted with mlx-vlm and keep the text
        // model under `language_model`.
        var weights = weights
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            weights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        var sanitized: [String: MLXArray] = [:]
        for (name, param) in weights {
            // Embeddings are tied; a stray head would fail the strict load.
            if name.hasPrefix("lm_head.") { continue }
            var param = param
            // PyTorch depthwise conv is (out, 1, K); MLX wants (out, K, 1).
            if name.contains("conv.weight"), param.shape[param.shape.count - 1] > param.shape[1] {
                param = param.transposed(0, 2, 1)
            }
            sanitized[name] = param
        }
        return sanitized
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.makeCache(parameters: parameters)
    }
}

extension LFM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

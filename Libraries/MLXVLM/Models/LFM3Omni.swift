//
//  LFM3Omni.swift
//  mlx-swift-lm
//

// Port of LiquidAI/LFM3-Omni-2B, input side only: text, images and audio in,
// text out. The reference is mlx-vlm's `lfm3_omni` (0.7.12), which composes
// `lfm3` (the text backbone), `lfm3_vl` (the vision tower and projector) and
// `lfm3_audio` (the audio encoder and projector); those in turn port the
// checkpoint's PyTorch remote code.
//
// Not ported: speech out. The depth decoder, the codebook embedding and the
// codec under `audio_tokenizer/` are dropped by `sanitize`, and generation
// never samples a media placeholder.
//
// The text backbone is a private copy of MLXLLM's LFM3, because MLXVLM does not
// depend on MLXLLM (LFM2-VL keeps its own LFM2 for the same reason). It keeps
// LFM3's fixes: `newCache` honours `maxKVSize`, and the 16-row attention gate
// is stored unquantized (the A14 GPU fault, fork d4f4f47).

import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Text backbone (private copy of LFM3)

enum OmniText {

    final class RoPE: Module {
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

        func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
            MLXFast.RoPE(
                x, dimensions: dims, traditional: false, base: nil, scale: 1.0, offset: offset,
                freqs: _freqs)
        }
    }

    final class Attention: Module {
        let config: LFM3OmniConfiguration.TextConfiguration
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rope: RoPE

        init(_ config: LFM3OmniConfiguration.TextConfiguration, layerIdx: Int) {
            self.config = config
            let headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)
            let dim = config.hiddenSize
            _qProj.wrappedValue = Linear(dim, config.attentionHeads * headDim, bias: false)
            _kProj.wrappedValue = Linear(dim, config.kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(dim, config.kvHeads * headDim, bias: false)
            _oProj.wrappedValue = Linear(config.attentionHeads * headDim, dim, bias: false)
            _gateProj.wrappedValue = Linear(dim, config.attentionHeads, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)

            let layerType = config.layerTypes[layerIdx]
            guard let rope = config.ropeParameters[layerType] else {
                fatalError("LFM3-Omni: no rope_parameters for \(layerType)")
            }
            self.rope = RoPE(
                dims: headDim, base: rope.ropeTheta,
                partialRotaryFactor: rope.partialRotaryFactor)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            var queries = qNorm(qProj(x).reshaped(B, L, config.attentionHeads, -1))
                .transposed(0, 2, 1, 3)
            var keys = kNorm(kProj(x).reshaped(B, L, config.kvHeads, -1))
                .transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)

            let gate = sigmoid(gateProj(x)).expandedDimensions(axis: -1)
            return oProj((output * gate).reshaped(B, L, -1))
        }
    }

    final class ShortConv: Module {
        let hiddenSize: Int
        let lCache: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM3OmniConfiguration.TextConfiguration) {
            self.hiddenSize = config.hiddenSize
            self.lCache = config.convLCache
            _conv.wrappedValue = Conv1d(
                inputChannels: config.hiddenSize, outputChannels: config.hiddenSize,
                kernelSize: lCache, groups: config.hiddenSize, bias: config.convBias)
            _inProj.wrappedValue = Linear(
                config.hiddenSize, 3 * config.hiddenSize, bias: config.convBias)
            _outProj.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: config.convBias)
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

    final class MLP: Module, UnaryLayer {
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

    final class DecoderLayer: Module {
        let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
        @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
        @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

        init(_ config: LFM3OmniConfiguration.TextConfiguration, layerIdx: Int) {
            self.isAttentionLayer = config.isAttention(layerIdx)
            if isAttentionLayer {
                _attention.wrappedValue = Attention(config, layerIdx: layerIdx)
            } else {
                _conv.wrappedValue = ShortConv(config)
            }
            _feedForward.wrappedValue = MLP(
                dim: config.hiddenSize, hiddenDim: config.intermediateSize)
            _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            if config.useSandwichNorm {
                _postOperatorNorm.wrappedValue = RMSNorm(
                    dimensions: config.hiddenSize, eps: config.normEps)
                _postFFNNorm.wrappedValue = RMSNorm(
                    dimensions: config.hiddenSize, eps: config.normEps)
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

    final class ModelInner: Module {
        let config: LFM3OmniConfiguration.TextConfiguration
        let firstFullIdx: Int?
        let firstSlidingIdx: Int?

        let layers: [DecoderLayer]
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM3OmniConfiguration.TextConfiguration) {
            self.config = config
            self.firstFullIdx = config.layerTypes.firstIndex(of: "full_attention")
            self.firstSlidingIdx = config.layerTypes.firstIndex(of: "sliding_attention")
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            self.layers = (0 ..< config.hiddenLayers).map { DecoderLayer(config, layerIdx: $0) }
            _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(_ h: MLXArray, cache: [KVCache]?) -> MLXArray {
            var h = h
            let fullMask = createAttentionMask(h: h, cache: firstFullIdx.flatMap { cache?[$0] })
            let slidingMask = createAttentionMask(
                h: h, cache: firstSlidingIdx.flatMap { cache?[$0] },
                windowSize: config.slidingWindow)
            for (i, layer) in layers.enumerated() {
                let mask = config.isSliding(i) ? slidingMask : fullMask
                h = layer(h, mask: mask, cache: cache?[i])
            }
            return embeddingNorm(h)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo(key: "model") var model: ModelInner

        init(_ config: LFM3OmniConfiguration.TextConfiguration) {
            _model.wrappedValue = ModelInner(config)
        }
    }
}

// MARK: - Vision tower (lfm3_vl_vision)

private enum OmniVision {

    /// Patch `(row, column)` coordinates in the order the processor lays
    /// patches out: merge blocks row by row, each block's patches row by row.
    static func gridPositions(height: Int, width: Int, merge: Int) -> MLXArray {
        var coords = [Float]()
        coords.reserveCapacity(height * width * 2)
        for bh in 0 ..< (height / merge) {
            for bw in 0 ..< (width / merge) {
                for mh in 0 ..< merge {
                    for mw in 0 ..< merge {
                        coords.append(Float(bh * merge + mh))
                        coords.append(Float(bw * merge + mw))
                    }
                }
            }
        }
        return MLXArray(coords, [height * width, 2])
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm

        init(_ config: LFM3OmniConfiguration.VisionConfiguration) {
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = config.hiddenSize / config.attentionHeads
            self.scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _outProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let n = x.dim(0)
            var q = qNorm(qProj(x).reshaped(n, heads, headDim))
            var k = kNorm(kProj(x).reshaped(n, kvHeads, headDim))
            let v = vProj(x).reshaped(n, kvHeads, headDim)

            // Axial RoPE, half-split, evaluated in float32 as the reference does.
            func rotate(_ t: MLXArray) -> MLXArray {
                let dtype = t.dtype
                let t32 = t.asType(.float32)
                let halves = t32.split(parts: 2, axis: -1)
                let rotated = concatenated([-halves[1], halves[0]], axis: -1)
                return (t32 * cos.expandedDimensions(axis: 1)
                    + rotated * sin.expandedDimensions(axis: 1)).asType(dtype)
            }
            q = rotate(q)
            k = rotate(k)

            let out = MLXFast.scaledDotProductAttention(
                queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
                keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
                values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
                scale: scale, mask: .none)
            return outProj(out[0].transposed(1, 0, 2).reshaped(n, -1))
        }
    }

    final class Conv: Module {
        let merge: Int

        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "spatial_conv") var spatialConv: Conv2d
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM3OmniConfiguration.VisionConfiguration) {
            self.merge = config.spatialMergeSize
            let c = config.hiddenSize
            _inProj.wrappedValue = Linear(c, 3 * c, bias: false)
            _spatialConv.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c,
                kernelSize: .init(config.convKernelSize),
                padding: .init(config.convKernelSize / 2), groups: c, bias: false)
            _outProj.wrappedValue = Linear(c, c, bias: false)
        }

        func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
            let parts = inProj(x).split(parts: 3, axis: -1)
            let (b, c, v) = (parts[0], parts[1], parts[2])
            let m = merge
            // Merge-block order to a raster grid, depthwise conv, and back.
            var grid = (b * v)
                .reshaped(height / m, width / m, m, m, -1)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(1, height, width, -1)
            grid = spatialConv(grid)
            let out = grid
                .reshaped(height / m, m, width / m, m, -1)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(height * width, -1)
            return outProj(c * out)
        }
    }

    final class Layer: Module {
        let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: Conv?
        @ModuleInfo(key: "feed_forward") var feedForward: OmniText.MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM3OmniConfiguration.VisionConfiguration, index: Int) {
            self.isAttentionLayer = config.layerTypes[index] == "full_attention"
            if isAttentionLayer {
                _attention.wrappedValue = Attention(config)
            } else {
                _conv.wrappedValue = Conv(config)
            }
            _feedForward.wrappedValue = OmniText.MLP(
                dim: config.hiddenSize, hiddenDim: config.intermediateSize)
            _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, height: Int, width: Int, cos: MLXArray, sin: MLXArray
        ) -> MLXArray {
            let h = operatorNorm(x)
            let r =
                isAttentionLayer
                ? attention!(h, cos: cos, sin: sin)
                : conv!(h, height: height, width: width)
            let x = x + r
            return x + feedForward(ffnNorm(x))
        }
    }

    final class Model: Module {
        let config: LFM3OmniConfiguration.VisionConfiguration

        @ModuleInfo(key: "patchifier") var patchifier: Linear
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
        let layers: [Layer]

        init(_ config: LFM3OmniConfiguration.VisionConfiguration) {
            self.config = config
            _patchifier.wrappedValue = Linear(
                config.inChannels * config.temporalPatchSize * config.patchSize
                    * config.patchSize,
                config.hiddenSize, bias: false)
            _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            self.layers = (0 ..< config.hiddenLayers).map { Layer(config, index: $0) }
        }

        /// `pixels` holds every frame's patches back to back; `frames` gives each
        /// one's `(t, h, w)` in patches. Returns one row per patch.
        func callAsFunction(_ pixels: MLXArray, frames: [THW]) -> MLXArray {
            let spatialDim = config.hiddenSize / config.attentionHeads / 2
            let exponents = MLXArray(stride(from: 0, to: spatialDim, by: 2)).asType(.float32)
                / Float(spatialDim)
            let invFreq = pow(MLXArray(config.ropeTheta), -exponents)

            var outputs = [MLXArray]()
            var offset = 0
            for frame in frames {
                let (temporal, height, width) = frame.values
                let coords = gridPositions(
                    height: height, width: width, merge: config.spatialMergeSize)
                var angles = (coords.expandedDimensions(axis: -1) * invFreq)
                    .reshaped(height * width, -1)
                angles = concatenated([angles, angles], axis: -1)
                let cos = MLX.cos(angles)
                let sin = MLX.sin(angles)
                for _ in 0 ..< temporal {
                    let patches = pixels[offset ..< (offset + height * width)]
                        .asType(embeddingNorm.weight.dtype)
                    var x = embeddingNorm(patchifier(patches))
                    for layer in layers {
                        x = layer(x, height: height, width: width, cos: cos, sin: sin)
                    }
                    outputs.append(x)
                    offset += height * width
                }
            }
            return concatenated(outputs, axis: 0)
        }
    }

    final class Projector: Module, UnaryLayer {
        let inputDim: Int

        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear

        init(_ config: LFM3OmniConfiguration) {
            let vision = config.visionConfiguration.hiddenSize
            self.inputDim = vision * config.downsampleFactor * config.downsampleFactor
            _layerNorm.wrappedValue = LayerNorm(
                dimensions: vision, eps: config.projectorLayerNormEps)
            _linear1.wrappedValue = Linear(
                inputDim, config.projectorHiddenSize, bias: config.projectorBias)
            _linear2.wrappedValue = Linear(
                config.projectorHiddenSize, config.textConfiguration.hiddenSize,
                bias: config.projectorBias)
        }

        /// Each merge block's patches are adjacent rows, so the 2x2
        /// pixel-unshuffle is a reshape.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(gelu(linear1(layerNorm(x).reshaped(-1, inputDim))))
        }
    }
}

// MARK: - Audio encoder (lfm3_omni_encoder)

private enum OmniAudio {

    /// An activation evaluated in float32 and rounded once, as torch does for
    /// bf16 inputs (a bf16 erf alone doubles the projector's error).
    static func fp32(_ act: (MLXArray) -> MLXArray, _ x: MLXArray) -> MLXArray {
        act(x.asType(.float32)).asType(x.dtype)
    }

    final class Downsampler: Module {
        @ModuleInfo(key: "conv_in") var convIn: Conv2d
        @ModuleInfo(key: "conv_mid") var convMid: Conv2d
        @ModuleInfo(key: "lin_mid") var linMid: Conv2d
        @ModuleInfo(key: "conv_end") var convEnd: Conv2d
        @ModuleInfo(key: "lin_end") var linEnd: Conv2d
        @ModuleInfo(key: "lin_out") var linOut: Linear

        init(_ config: LFM3OmniConfiguration.AudioEncoderConfiguration) {
            let c = config.convChannels
            _convIn.wrappedValue = Conv2d(
                inputChannels: 1, outputChannels: c, kernelSize: 3, stride: 2, padding: 1)
            _convMid.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1,
                groups: c)
            _linMid.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
            _convEnd.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1,
                groups: c)
            _linEnd.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
            _linOut.wrappedValue = Linear(config.numMelBins * c / 8, config.hiddenSize)
        }

        /// `x` is `(clips, mel, frames)`; NHWC throughout, with the mel bins as
        /// height and time as width.
        func callAsFunction(_ x: MLXArray, lengths: [Int]) -> MLXArray {
            func mask(_ x: MLXArray, _ n: [Int]) -> MLXArray {
                let positions = MLXArray(Int32(0) ..< Int32(x.dim(2)))
                let valid = positions.expandedDimensions(axis: 0)
                    .< MLXArray(n.map(Int32.init)).expandedDimensions(axis: 1)
                return x * valid[0..., .newAxis, 0..., .newAxis].asType(x.dtype)
            }
            var n = lengths
            var x = mask(x.expandedDimensions(axis: -1).asType(convIn.weight.dtype), n)
            x = OmniAudio.fp32(silu, convIn(x))
            n = n.map { ($0 + 1) / 2 }
            x = mask(x, n)
            x = OmniAudio.fp32(silu, linMid(convMid(x)))
            n = n.map { ($0 + 1) / 2 }
            x = mask(x, n)
            x = OmniAudio.fp32(silu, linEnd(convEnd(x)))
            let (B, F, T, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            // The reference flattens (channel, mel) channel-major.
            return linOut(x.transposed(0, 2, 3, 1).reshaped(B, T, C * F))
        }
    }

    /// LFM2's attention, with no cache: the whole clip at once.
    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm
        let rope: MLXNN.RoPE

        init(_ config: LFM3OmniConfiguration.AudioTrunkConfiguration) {
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = config.hiddenSize / config.attentionHeads
            self.scale = pow(Float(headDim), -0.5)
            let dim = config.hiddenSize
            _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let q = rope(qNorm(qProj(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3))
            let k = rope(kNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3))
            let v = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
            return outProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    /// LFM2's short convolution with no cache: causal, left-padded.
    final class ShortConv: Module {
        let lCache: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM3OmniConfiguration.AudioTrunkConfiguration) {
            self.lCache = config.convLCache
            let dim = config.hiddenSize
            _conv.wrappedValue = Conv1d(
                inputChannels: dim, outputChannels: dim, kernelSize: lCache, groups: dim,
                bias: config.convBias)
            _inProj.wrappedValue = Linear(dim, 3 * dim, bias: config.convBias)
            _outProj.wrappedValue = Linear(dim, dim, bias: config.convBias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = inProj(x).split(parts: 3, axis: -1)
            let (B, C, xs) = (parts[0], parts[1], parts[2])
            let Bx = padded(B * xs, widths: [0, .init((lCache - 1, 0)), 0])
            return outProj(C * conv(Bx))
        }
    }

    final class Layer: Module {
        let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: OmniText.MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM3OmniConfiguration.AudioTrunkConfiguration, index: Int) {
            self.isAttentionLayer = config.layerTypes[index] == "full_attention"
            if isAttentionLayer {
                _attention.wrappedValue = Attention(config)
            } else {
                _conv.wrappedValue = ShortConv(config)
            }
            _feedForward.wrappedValue = OmniText.MLP(
                dim: config.hiddenSize, hiddenDim: config.feedForwardDim)
            _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
            let h = operatorNorm(x)
            let r = isAttentionLayer ? attention!(h, mask: mask) : conv!(h)
            let x = x + r
            return x + feedForward(ffnNorm(x))
        }
    }

    final class Trunk: Module {
        let layers: [Layer]
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM3OmniConfiguration.AudioTrunkConfiguration) {
            self.layers = (0 ..< config.hiddenLayers).map { Layer(config, index: $0) }
            _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(_ h: MLXArray, window: Int) -> MLXArray {
            let mask = createCausalMask(n: h.dim(1), offset: 0, windowSize: window)
            var h = h
            for layer in layers {
                h = layer(h, mask: mask)
            }
            return embeddingNorm(h)
        }
    }

    final class Encoder: Module {
        let window: Int

        @ModuleInfo(key: "downsample") var downsample: Downsampler
        @ModuleInfo(key: "lfm") var lfm: Trunk

        init(_ config: LFM3OmniConfiguration.AudioEncoderConfiguration) {
            self.window = config.slidingWindow
            _downsample.wrappedValue = Downsampler(config)
            _lfm.wrappedValue = Trunk(config.textConfiguration)
        }

        /// The valid rows of every clip, in order: one per 8 feature frames.
        /// Padded rows trail each clip and the trunk is causal, so they change
        /// nothing before them.
        func callAsFunction(_ features: MLXArray, lengths: [Int]) -> MLXArray {
            let h = lfm(downsample(features, lengths: lengths), window: window)
            let rows = lengths.map(LFM3OmniAudioFeatures.encoderTokens(frames:))
            return concatenated(
                rows.enumerated().map { i, n in h[i, 0 ..< n] }, axis: 0)
        }
    }

    final class Projector: Module, UnaryLayer {
        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear

        init(inputs: Int, outputs: Int) {
            _layerNorm.wrappedValue = LayerNorm(dimensions: inputs, eps: 1e-5)
            _linear1.wrappedValue = Linear(inputs, outputs)
            _linear2.wrappedValue = Linear(outputs, outputs)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(OmniAudio.fp32(gelu, linear1(layerNorm(x))))
        }
    }
}

// MARK: - Model

/// LFM3-Omni, reading text, images and audio and writing text.
///
/// Typically created by ``VLMModelFactory``. A prompt that carries an image or
/// a clip is never served from a prompt cache (``LMInput/carriesMedia``).
public class LFM3Omni: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "language_model") private var languageModel: OmniText.LanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: OmniVision.Model
    @ModuleInfo(key: "multi_modal_projector") private var visionProjector: OmniVision.Projector
    @ModuleInfo(key: "audio_tower") private var audioTower: OmniAudio.Encoder
    @ModuleInfo(key: "audio_projector") private var audioProjector: OmniAudio.Projector

    public let config: LFM3OmniConfiguration

    /// The speech path (`LFM3OmniSpeech`) reads the backbone directly.
    var speechLanguageModel: OmniText.LanguageModel { languageModel }

    /// Added to every logit row so text generation never picks a placeholder
    /// (image, video, audio, prior speech) the prompt alone may carry.
    private let _placeholderLogitMask: MLXArray

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public let kvHeads: [Int]

    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: LFM3OmniConfiguration) {
        self.config = config
        let text = config.textConfiguration
        self.kvHeads = (0 ..< text.hiddenLayers).map { text.isAttention($0) ? text.kvHeads : 0 }

        var mask = [Float](repeating: 0, count: text.vocabularySize)
        for id in config.placeholderTokenIds where id < mask.count {
            mask[id] = -.infinity
        }
        self._placeholderLogitMask = MLXArray(mask)

        _languageModel.wrappedValue = OmniText.LanguageModel(text)
        _visionTower.wrappedValue = OmniVision.Model(config.visionConfiguration)
        _visionProjector.wrappedValue = OmniVision.Projector(config)
        _audioTower.wrappedValue = OmniAudio.Encoder(config.audioEncoderConfiguration)
        _audioProjector.wrappedValue = OmniAudio.Projector(
            inputs: config.audioEncoderConfiguration.hiddenSize, outputs: text.hiddenSize)
    }

    // MARK: Encoders (internal so the parity tests can read them)

    /// One embedding per image placeholder, for every image in order.
    func encodeImages(_ image: LMInput.ProcessedImage) throws -> MLXArray {
        guard let frames = image.frames, !frames.isEmpty else {
            throw VLMError.imageProcessingFailure("LFM3-Omni needs each image's grid")
        }
        let pixels = image.pixels.asType(visionTower.patchifier.weight.dtype)
        return visionProjector(visionTower(pixels, frames: frames))
    }

    /// One embedding per audio placeholder, for every clip in order.
    func encodeAudio(_ audio: LMInput.ProcessedAudio) -> MLXArray {
        audioProjector(audioTower(audio.features, lengths: audio.lengths))
    }

    private func logits(_ hidden: MLXArray) -> MLXArray {
        let logits = languageModel.model.embedTokens.asLinear(hidden)
        return logits + _placeholderLogitMask.asType(logits.dtype)
    }

    /// Text embeddings with every placeholder replaced by its feature row.
    func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        var embeddings = languageModel.model.embedTokens(tokens)

        let ids = tokens.flattened().asArray(Int.self)
        func scatter(_ features: MLXArray, tokenId: Int, label: String) throws {
            let positions = ids.indices.filter { ids[$0] == tokenId }
            guard positions.count == features.dim(0) else {
                throw VLMError.processing(
                    "LFM3-Omni: \(positions.count) \(label) placeholders for "
                        + "\(features.dim(0)) feature rows")
            }
            guard !positions.isEmpty else { return }
            embeddings[0..., MLXArray(positions.map(Int32.init)), 0...] =
                features.asType(embeddings.dtype).expandedDimensions(axis: 0)
        }

        if let image = input.image {
            try scatter(encodeImages(image), tokenId: config.imageTokenId, label: "image")
        }
        if let audio = input.audio {
            try scatter(encodeAudio(audio), tokenId: config.audioTokenId, label: "audio")
        }
        return embeddings
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let embeddings = try inputEmbeddings(input)
        let hidden = languageModel.model(embeddings, cache: cache)
        return .logits(LMOutput(logits: logits(hidden)))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        var tokens = inputs
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        return logits(languageModel.model(languageModel.model.embedTokens(tokens), cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Input side only: the speech-out modules and anything a codec folder
        // merged in are left out, so the strict load sees only what is built.
        let kept = [
            "language_model.", "vision_tower.", "multi_modal_projector.", "audio_tower.",
            "audio_projector.",
        ]
        var sanitized = [String: MLXArray]()
        for (name, value) in weights {
            var name = name
            if name.hasPrefix("audio_encoder.") {
                name = "audio_tower." + name.dropFirst("audio_encoder.".count)
            }
            guard kept.contains(where: { name.hasPrefix($0) }) else { continue }
            if name.hasPrefix("language_model.lm_head.") { continue }

            var value = value
            // PyTorch layouts, should a checkpoint carry them; MLX builds do not.
            if name.hasSuffix("conv.conv.weight"), value.ndim == 3,
                value.dim(-1) > value.dim(1)
            {
                value = value.transposed(0, 2, 1)
            } else if name.hasSuffix("spatial_conv.weight"), value.ndim == 4, value.dim(1) == 1,
                value.dim(-1) != 1
            {
                value = value.transposed(0, 2, 3, 1)
            } else if name.hasPrefix("audio_tower.downsample."), value.ndim == 4,
                value.dim(2) == value.dim(3), value.dim(1) != value.dim(2)
            {
                value = value.transposed(0, 2, 3, 1)
            }
            sanitized[name] = value
        }
        return dequantizingGates(sanitized)
    }

    /// Stores each text attention gate unquantized, as MLXLLM's LFM3 does: its
    /// 16-row 4-bit matmul GPU-faulted an A14 iPad (fork d4f4f47).
    func dequantizingGates(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        let suffix = "self_attn.gate_proj.weight"
        for (name, packed) in weights where name.hasSuffix(suffix) {
            let prefix = String(name.dropLast(".weight".count))
            guard let scales = weights[prefix + ".scales"] else { continue }
            let inputs = config.textConfiguration.hiddenSize
            weights[name] = dequantized(
                packed, scales: scales, biases: weights[prefix + ".biases"],
                groupSize: inputs / scales.dim(-1), bits: packed.dim(-1) * 32 / inputs)
            weights[prefix + ".scales"] = nil
            weights[prefix + ".biases"] = nil
        }
        return weights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let text = config.textConfiguration
        return (0 ..< text.hiddenLayers).map { layerIdx -> KVCache in
            switch text.layerTypes[layerIdx] {
            case "conv":
                return MambaCache()
            case "sliding_attention":
                let window = text.slidingWindow
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

// MARK: - Configuration

public struct LFM3OmniConfiguration: Codable, Sendable {

    public struct RopeParameters: Codable, Sendable {
        public let ropeType: String
        public let ropeTheta: Float
        public let partialRotaryFactor: Float

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            ropeType = try c.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
            ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
            partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        }
    }

    public struct TextConfiguration: Codable, Sendable {
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let normEps: Float
        public let convBias: Bool
        public let convLCache: Int
        public let layerTypes: [String]
        public let slidingWindow: Int
        public let useSandwichNorm: Bool
        public let ropeParameters: [String: RopeParameters]

        enum CodingKeys: String, CodingKey {
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

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
            attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
            normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
            convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
            convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
            layerTypes = try c.decode([String].self, forKey: .layerTypes)
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
            useSandwichNorm = try c.decodeIfPresent(Bool.self, forKey: .useSandwichNorm) ?? false
            ropeParameters = try c.decode([String: RopeParameters].self, forKey: .ropeParameters)
            guard layerTypes.count == hiddenLayers else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes, in: c,
                    debugDescription:
                        "LFM3-Omni: \(layerTypes.count) layer_types for \(hiddenLayers) layers")
            }
        }

        func isAttention(_ i: Int) -> Bool { layerTypes[i] != "conv" }
        func isSliding(_ i: Int) -> Bool { layerTypes[i] == "sliding_attention" }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let inChannels: Int
        public let patchSize: Int
        public let temporalPatchSize: Int
        public let spatialMergeSize: Int
        public let convKernelSize: Int
        public let normEps: Float
        public let layerTypes: [String]
        public let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case inChannels = "in_channels"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case convKernelSize = "conv_kernel_size"
            case normEps = "norm_eps"
            case layerTypes = "layer_types"
            case ropeParameters = "rope_parameters"
        }

        private struct Rope: Codable {
            let ropeType: String?
            let ropeTheta: Float?
            enum CodingKeys: String, CodingKey {
                case ropeType = "rope_type"
                case ropeTheta = "rope_theta"
            }
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2560
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 14
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 3
            normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
            layerTypes = try c.decode([String].self, forKey: .layerTypes)
            let rope = try c.decodeIfPresent(Rope.self, forKey: .ropeParameters)
            guard (rope?.ropeType ?? "axial") == "axial" else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeParameters, in: c,
                    debugDescription: "LFM3-Omni vision needs axial rotary embeddings")
            }
            ropeTheta = rope?.ropeTheta ?? 1e6
            guard layerTypes.count == hiddenLayers,
                Set(layerTypes).isSubset(of: ["conv", "full_attention"])
            else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes, in: c,
                    debugDescription: "LFM3-Omni vision: unsupported layer_types")
            }
        }

        public func encode(to encoder: Swift.Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(hiddenSize, forKey: .hiddenSize)
            try c.encode(layerTypes, forKey: .layerTypes)
        }
    }

    /// The audio encoder's LFM2 trunk.
    public struct AudioTrunkConfiguration: Codable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let normEps: Float
        public let convLCache: Int
        public let convBias: Bool
        public let layerTypes: [String]
        public let ropeTheta: Float
        let blockAutoAdjustFFDim: Bool
        let blockFFNDimMultiplier: Float?
        let blockMultipleOf: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case normEps = "norm_eps"
            case convLCache = "conv_L_cache"
            case convBias = "conv_bias"
            case layerTypes = "layer_types"
            case ropeParameters = "rope_parameters"
            case ropeTheta = "rope_theta"
            case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
            case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case blockMultipleOf = "block_multiple_of"
        }

        private struct Rope: Codable {
            let ropeTheta: Float?
            enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
            attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
            convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
            convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
            layerTypes = try c.decode([String].self, forKey: .layerTypes)
            let rope = try c.decodeIfPresent(Rope.self, forKey: .ropeParameters)
            ropeTheta =
                try rope?.ropeTheta ?? c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1e6
            blockAutoAdjustFFDim =
                try c.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDim) ?? true
            blockFFNDimMultiplier = try c.decodeIfPresent(
                Float.self, forKey: .blockFFNDimMultiplier)
            blockMultipleOf = try c.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
        }

        public func encode(to encoder: Swift.Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(hiddenSize, forKey: .hiddenSize)
        }

        /// LFM2's auto-adjusted feed-forward width (2560 → 1792 here).
        var feedForwardDim: Int {
            guard blockAutoAdjustFFDim else { return intermediateSize }
            var dim = Int(2 * Float(intermediateSize) / 3)
            if let blockFFNDimMultiplier { dim = Int(blockFFNDimMultiplier * Float(dim)) }
            return blockMultipleOf * ((dim + blockMultipleOf - 1) / blockMultipleOf)
        }
    }

    public struct AudioEncoderConfiguration: Codable, Sendable {
        public let convChannels: Int
        public let hiddenSize: Int
        public let numMelBins: Int
        public let slidingWindow: Int
        public let textConfiguration: AudioTrunkConfiguration

        enum CodingKeys: String, CodingKey {
            case convChannels = "conv_channels"
            case hiddenSize = "hidden_size"
            case numMelBins = "num_mel_bins"
            case slidingWindow = "sliding_window"
            case textConfiguration = "text_config"
        }

        public init(from decoder: Swift.Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            convChannels = try c.decodeIfPresent(Int.self, forKey: .convChannels) ?? 256
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
            numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 32
            textConfiguration = try c.decode(AudioTrunkConfiguration.self, forKey: .textConfiguration)
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let audioEncoderConfiguration: AudioEncoderConfiguration
    public let downsampleFactor: Int
    public let projectorHiddenSize: Int
    public let projectorBias: Bool
    public let projectorLayerNormEps: Float
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let audioTokenId: Int
    public let audioOutTokenId: Int

    /// Prompt-only ids the text head never samples.
    var placeholderTokenIds: [Int] { [audioTokenId, audioOutTokenId, imageTokenId, videoTokenId] }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioEncoderConfiguration = "audio_encoder_config"
        case downsampleFactor = "downsample_factor"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorBias = "projector_bias"
        case projectorLayerNormEps = "projector_layer_norm_eps"
        case projectorHiddenAct = "projector_hidden_act"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case audioTokenId = "audio_token_id"
        case audioOutTokenId = "audio_out_token_id"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        audioEncoderConfiguration = try c.decode(
            AudioEncoderConfiguration.self, forKey: .audioEncoderConfiguration)
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        projectorHiddenSize = try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorLayerNormEps =
            try c.decodeIfPresent(Float.self, forKey: .projectorLayerNormEps) ?? 1e-6
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 125030
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 125029
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 125031
        audioOutTokenId = try c.decodeIfPresent(Int.self, forKey: .audioOutTokenId) ?? 125032
        let act = try c.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        guard act == "gelu", downsampleFactor == visionConfiguration.spatialMergeSize else {
            throw DecodingError.dataCorruptedError(
                forKey: .downsampleFactor, in: c,
                debugDescription: "LFM3-Omni: a GELU projector whose downsampling is the merge size")
        }
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(imageTokenId, forKey: .imageTokenId)
        try c.encode(audioTokenId, forKey: .audioTokenId)
    }
}

//
//  LFM3OmniSpeech.swift
//  mlx-swift-lm
//

// LFM3-Omni's speech out (Mortise docs/spoken-replies.md, SR2 and SR3).
//
// Port of mlx-vlm 0.7.12's `lfm3_audio` speech path: the depth decoder fills
// each 12-codebook frame in MaskGIT passes from the backbone's hidden state,
// the frame's summed codebook embeddings feed the next backbone step, and the
// Liquid codec decodes the codes to 24 kHz audio. Plain TTS only
// (`<|audio_start|>`); the interleaved mode is not ported. One pass loops past
// about 60 words, so a caller speaks long text a sentence chunk at a time.
//
// The depth decoder and the codebook embedding live in `model.safetensors`
// beside the input side, which `LFM3Omni.sanitize` drops; they are loaded here
// on their own. The codec is `audio_tokenizer/`.

import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import MLXNN

// MARK: - Depth decoder

/// RMSNorm that scales by the weight in fp32 and rounds once, as the trained
/// heads do.
final class OmniHeadRMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
            .asType(x.dtype)
    }
}

/// One codebook's input embedding and tied output head.
final class OmniCodebookEmbedding: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: OmniHeadRMSNorm

    init(codebookSize: Int, dimensions: Int, eps: Float) {
        _embedding.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dimensions)
        _embeddingNorm.wrappedValue = OmniHeadRMSNorm(dimensions: dimensions, eps: eps)
    }

    func logits(_ x: MLXArray) -> MLXArray { embedding.asLinear(embeddingNorm(x)) }
}

/// The LFM3 trunk without `embed_tokens`, attending bidirectionally.
final class OmniDepthModel: Module {
    let layers: [OmniText.DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ config: LFM3OmniConfiguration.TextConfiguration) {
        self.layers = (0 ..< config.hiddenLayers).map {
            OmniText.DecoderLayer(config, layerIdx: $0)
        }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for layer in layers { x = layer(x, mask: .none, cache: nil) }
        return embeddingNorm(x)
    }
}

final class OmniDepthDecoder: Module {
    let numCodebooks: Int
    let hiddenSize: Int

    @ModuleInfo(key: "depth_linear") var depthLinear: Linear
    @ParameterInfo(key: "mask_embedding") var maskEmbedding: MLXArray
    @ModuleInfo(key: "model") var model: OmniDepthModel
    @ModuleInfo(key: "depth_embeddings") var depthEmbeddings: [OmniCodebookEmbedding]

    init(
        _ config: LFM3OmniConfiguration.TextConfiguration, inputSize: Int, numCodebooks: Int,
        codebookSize: Int
    ) {
        self.numCodebooks = numCodebooks
        self.hiddenSize = config.hiddenSize
        _depthLinear.wrappedValue = Linear(inputSize, numCodebooks * config.hiddenSize)
        _maskEmbedding.wrappedValue = MLXArray.zeros([config.hiddenSize])
        _model.wrappedValue = OmniDepthModel(config)
        _depthEmbeddings.wrappedValue = (0 ..< numCodebooks).map { _ in
            OmniCodebookEmbedding(
                codebookSize: codebookSize, dimensions: config.hiddenSize, eps: config.normEps)
        }
    }

    /// hidden [B, D], codes [B, K], masked [B, K] (true = hidden) → logits [B, K, V].
    func callAsFunction(_ hidden: MLXArray, codes: MLXArray, masked: MLXArray) -> MLXArray {
        let B = hidden.dim(0)
        let x = depthLinear(hidden).reshaped(B, numCodebooks, hiddenSize)
        let embeds = stacked(
            depthEmbeddings.enumerated().map { i, e in e.embedding(codes[0..., i]) }, axis: -2)
        let mask = maskEmbedding.asType(embeds.dtype)
        let h = model(x + MLX.where(masked.expandedDimensions(axis: -1), mask, embeds))
        return stacked(
            depthEmbeddings.enumerated().map { i, e in e.logits(h[0..., i, 0...]) }, axis: -2)
    }
}

// MARK: - Liquid codec (decode only)

final class OmniCodecTrunk: Module {
    let config: LFM3OmniConfiguration.TextConfiguration
    let layers: [OmniText.DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    init(_ config: LFM3OmniConfiguration.TextConfiguration) {
        self.config = config
        self.layers = (0 ..< config.hiddenLayers).map {
            OmniText.DecoderLayer(config, layerIdx: $0)
        }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        var h = h
        let mask = createAttentionMask(h: h, cache: nil, windowSize: config.slidingWindow)
        for layer in layers { h = layer(h, mask: mask, cache: nil) }
        return embeddingNorm(h)
    }
}

final class OmniCodecQuantizer: Module {
    let levels: [Int]
    let basis: [Int]
    @ModuleInfo(key: "project_out") var projectOut: Linear

    init(levels: [Int], numQuantizers: Int, quantizerDim: Int) {
        self.levels = levels
        self.basis = levels.indices.map { levels[..<$0].reduce(1, *) }
        _projectOut.wrappedValue = Linear(numQuantizers * levels.count, quantizerDim)
    }

    /// FSQ dequantization: codes [B, n_q, T] → [B, T, quantizerDim].
    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let levelsA = MLXArray(levels.map(Int32.init))
        // Integer division, as the reference's `//`: MLX's `/` on ints is true division.
        let half = floorDivide(levelsA, MLXArray(Int32(2))).asType(.float32)
        let digits = remainder(
            floorDivide(
                codes.asType(.int32).transposed(0, 2, 1).expandedDimensions(axis: -1),
                MLXArray(basis.map(Int32.init))),
            levelsA)
        let latents = ((digits.asType(.float32) - half) / half)
            .reshaped(codes.dim(0), codes.dim(2), -1)
        return projectOut(latents.asType(projectOut.weight.dtype))
    }
}

final class OmniISTFT: Module {
    let nFFT: Int
    let hop: Int
    let window: MLXArray

    init(nFFT: Int, hop: Int) {
        self.nFFT = nFFT
        self.hop = hop
        self.window = MLXArray.zeros([nFFT], dtype: .float32)
    }

    private func overlapAdd(_ frames: MLXArray) -> MLXArray {
        let (B, n) = (frames.dim(0), frames.dim(1))
        let k = nFFT / hop
        let blocks = frames.reshaped(B, n, k, hop)
        var out: MLXArray? = nil
        for j in 0 ..< k {
            let padded = padded(
                blocks[0..., 0..., j, 0...],
                widths: [.init((0, 0)), .init((j, k - 1 - j)), .init((0, 0))])
            out = out.map { $0 + padded } ?? padded
        }
        return out!.reshaped(B, -1)
    }

    func callAsFunction(_ spec: MLXArray) -> MLXArray {
        let w = window.asType(.float32)
        let frames = irfft(spec, n: nFFT, axis: -1) * w
        let audio = overlapAdd(frames)
        let envelope = overlapAdd(broadcast(w * w, to: [1, spec.dim(1), nFFT]))
        let trim = (nFFT - hop) / 2
        let end = audio.dim(1) - trim
        return audio[0..., trim ..< end] / envelope[0..., trim ..< end]
    }
}

final class OmniCodecDecoder: Module {
    let stack: Int
    @ModuleInfo(key: "lfm") var lfm: OmniCodecTrunk
    @ModuleInfo(key: "lin") var lin: Linear
    @ModuleInfo(key: "istft") var istft: OmniISTFT

    init(config: LFM3OmniCodecConfiguration, trunk: LFM3OmniConfiguration.TextConfiguration) {
        self.stack = config.stftFrameStack
        _lfm.wrappedValue = OmniCodecTrunk(trunk)
        _lin.wrappedValue = Linear(config.encoderDim, config.stftNFFT + 2)
        _istft.wrappedValue = OmniISTFT(nFFT: config.stftNFFT, hop: config.stftHopLength)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = lfm(repeated(x, count: stack, axis: 1))
        let parts = lin(h).asType(.float32).split(parts: 2, axis: -1)
        let mag = exp(parts[0])
        let angle = parts[1]
        let spec = (mag * cos(angle)).asType(.complex64) + (mag * sin(angle)).asImaginary()
        return istft(spec)
    }
}

final class OmniLiquidCodec: Module {
    let config: LFM3OmniCodecConfiguration
    @ModuleInfo(key: "quantizer") var quantizer: OmniCodecQuantizer
    @ModuleInfo(key: "from_quantizer") var fromQuantizer: Linear
    @ModuleInfo(key: "decoder") var decoder: OmniCodecDecoder

    init(_ config: LFM3OmniCodecConfiguration) throws {
        self.config = config
        _quantizer.wrappedValue = OmniCodecQuantizer(
            levels: config.fsqLevels, numQuantizers: config.nQ, quantizerDim: config.quantizerDim)
        _fromQuantizer.wrappedValue = Linear(config.quantizerDim, config.encoderDim)
        _decoder.wrappedValue = OmniCodecDecoder(config: config, trunk: try config.trunk())
    }

    /// codes [B, n_q, T] in [0, codebook size) → float32 audio [B, T × 1920].
    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        decoder(fromQuantizer(quantizer(codes.asType(.int32))))
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (name, value) in weights {
            let name = name.replacingOccurrences(of: "quantizer.fsq.", with: "quantizer.")
            guard
                name.hasPrefix("decoder.") || name.hasPrefix("from_quantizer.")
                    || name.hasPrefix("quantizer.project_out.")
            else { continue }
            var value = value
            if name.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1 {
                value = value.transposed(0, 2, 1)
            }
            result[name] = value
        }
        return result
    }
}

public struct LFM3OmniCodecConfiguration: Codable, Sendable {
    public let encoderDim: Int
    public let quantizerDim: Int
    public let fsqLevels: [Int]
    public let nQ: Int
    public let stftNFFT: Int
    public let stftHopLength: Int
    public let stftFrameStack: Int
    public let decoderWindow: Int
    public let samplingRate: Int

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case quantizerDim = "quantizer_dim"
        case fsqLevels = "fsq_levels"
        case nQ = "n_q"
        case stftNFFT = "stft_n_fft"
        case stftHopLength = "stft_hop_length"
        case stftFrameStack = "stft_frame_stack"
        case decoderWindow = "decoder_window"
        case samplingRate = "sampling_rate"
    }

    /// The trunk's shape is fixed by the reference config class, not config.json.
    func trunk() throws -> LFM3OmniConfiguration.TextConfiguration {
        let json: [String: Any] = [
            "vocab_size": 1, "hidden_size": encoderDim, "intermediate_size": 2304,
            "num_hidden_layers": 8, "num_attention_heads": 8, "num_key_value_heads": 2,
            "head_dim": 64, "norm_eps": 1e-5, "conv_bias": false, "conv_L_cache": 7,
            "layer_types": (0 ..< 8).map { [2, 4, 6].contains($0) ? "sliding_attention" : "conv" },
            "sliding_window": decoderWindow, "use_sandwich_norm": true,
            "rope_parameters": ["sliding_attention": ["rope_theta": 1e6]],
        ]
        return try JSONDecoder().decode(
            LFM3OmniConfiguration.TextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: json))
    }
}

// MARK: - Speech configuration

public struct LFM3OmniSpeechConfiguration: Codable, Sendable {
    public let depthConfig: LFM3OmniConfiguration.TextConfiguration
    public let numCodebooks: Int
    public let codebookSize: Int
    public let maskgitSchedule: [Int]
    public let audioEosCode: Int
    public let audioStartTokenId: Int
    public let hiddenSize: Int

    enum CodingKeys: String, CodingKey {
        case depthConfig = "depth_config"
        case numCodebooks = "num_codebooks"
        case codebookSize = "codebook_size"
        case maskgitSchedule = "maskgit_schedule"
        case audioEosCode = "audio_eos_code"
        case audioStartTokenId = "audio_start_token_id"
        case textConfig = "text_config"
    }

    enum TextKeys: String, CodingKey { case hiddenSize = "hidden_size" }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        depthConfig = try c.decode(
            LFM3OmniConfiguration.TextConfiguration.self, forKey: .depthConfig)
        numCodebooks = try c.decode(Int.self, forKey: .numCodebooks)
        codebookSize = try c.decode(Int.self, forKey: .codebookSize)
        maskgitSchedule = try c.decode([Int].self, forKey: .maskgitSchedule)
        audioEosCode = try c.decode(Int.self, forKey: .audioEosCode)
        audioStartTokenId = try c.decode(Int.self, forKey: .audioStartTokenId)
        hiddenSize = try c.nestedContainer(keyedBy: TextKeys.self, forKey: .textConfig)
            .decode(Int.self, forKey: .hiddenSize)
    }

    public func encode(to encoder: Swift.Encoder) throws {}
}

// MARK: - Speech

public struct LFM3OmniSpeechParameters: Sendable {
    /// The model card's speech sampling for LFM3-Omni.
    public var temperature: Float = 0.4
    public var topK: Int? = nil
    public var repetitionPenalty: Float = 1.2
    public var repetitionWindow: Int = 64
    public var maxFrames: Int = 3000

    public init() {}
}

public struct LFM3OmniSpeechStats: Sendable {
    public var promptTokens = 0
    public var frames = 0
    public var prefillMs = 0.0
    public var firstFrameMs = 0.0
    public var framesMs = 0.0
    public var codecMs = 0.0
    public var endedOnEos = false
    /// `shouldContinue` answered false before the end code.
    public var cancelled = false

    public init() {}
}

public final class LFM3OmniSpeech: Module {
    public let config: LFM3OmniSpeechConfiguration
    public let codecConfig: LFM3OmniCodecConfiguration

    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding
    @ModuleInfo(key: "depth_decoder") var depthDecoder: OmniDepthDecoder
    @ModuleInfo(key: "codec") var codec: OmniLiquidCodec

    init(config: LFM3OmniSpeechConfiguration, codecConfig: LFM3OmniCodecConfiguration) throws {
        self.config = config
        self.codecConfig = codecConfig
        _codebookEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numCodebooks * config.codebookSize,
            dimensions: config.hiddenSize)
        _depthDecoder.wrappedValue = OmniDepthDecoder(
            config.depthConfig, inputSize: config.hiddenSize, numCodebooks: config.numCodebooks,
            codebookSize: config.codebookSize)
        _codec.wrappedValue = try OmniLiquidCodec(codecConfig)
    }

    /// Loads the speech head from an Omni folder: the depth decoder and the
    /// codebook embedding from its safetensors, the codec from `audio_tokenizer/`.
    public static func load(
        directory: URL, codecDirectory: URL? = nil, dtype: DType = .bfloat16
    ) throws -> LFM3OmniSpeech {
        let config = try JSONDecoder().decode(
            LFM3OmniSpeechConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "config.json")))
        let codecDir = codecDirectory ?? directory.appending(path: "audio_tokenizer")
        let codecConfig = try JSONDecoder().decode(
            LFM3OmniCodecConfiguration.self,
            from: Data(contentsOf: codecDir.appending(path: "config.json")))
        let speech = try LFM3OmniSpeech(config: config, codecConfig: codecConfig)

        var weights = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "safetensors" {
            for (name, value) in try loadArrays(url: file)
            where name.hasPrefix("depth_decoder.") || name.hasPrefix("codebook_embedding.") {
                guard !name.contains(".to_logits.") else { continue }
                weights[name] = value.asType(dtype)
            }
        }
        let codecFiles = try FileManager.default.contentsOfDirectory(
            at: codecDir, includingPropertiesForKeys: nil)
        for file in codecFiles where file.pathExtension == "safetensors" {
            for (name, value) in OmniLiquidCodec.sanitize(try loadArrays(url: file)) {
                // The ISTFT window stays fp32, as the reference keeps the whole ISTFT.
                weights["codec." + name] =
                    name == "decoder.istft.window" ? value : value.asType(dtype)
            }
        }
        try speech.update(
            parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(speech)
        return speech
    }

    /// Sum of each frame's codebook embeddings: codes [..., K] → [..., D].
    func embedFrames(_ codes: MLXArray) -> MLXArray {
        let k = codes.dim(-1)
        let offsets = MLXArray(Int32(0) ..< Int32(k)) * Int32(config.codebookSize)
        return codebookEmbedding(codes.asType(.int32) + offsets).sum(axis: -2)
    }

    /// Samples ids from logits [..., V], as the reference `sample` does.
    static func sample(
        _ logits: MLXArray, temperature: Float, topK: Int?, repetitionPenalty: Float,
        history: MLXArray?
    ) -> MLXArray {
        var scores = logits.asType(.float32)
        if repetitionPenalty != 1, let history {
            let ids = broadcast(
                history, to: Array(scores.shape.dropLast()) + [history.dim(-1)])
            let picked = takeAlong(scores, ids, axis: -1)
            let penalized = MLX.where(
                picked .< 0, picked * repetitionPenalty, picked / repetitionPenalty)
            scores = putAlong(scores, ids, values: penalized, axis: -1)
        }
        if temperature == 0 { return argMax(scores, axis: -1) }
        scores = scores / temperature
        scores = scores - logSumExp(scores, axis: -1, keepDims: true)
        if let topK, topK < scores.dim(-1) {
            let kth = sorted(scores, axis: -1)[.ellipsis, (scores.dim(-1) - topK) ..< (scores.dim(-1) - topK + 1)]
            scores = MLX.where(scores .< kth, MLXArray(-Float.infinity), scores)
        }
        return categorical(scores)
    }

    /// MaskGIT one frame: hidden [1, D] → codes [1, K], before EOS handling.
    /// `order` [1, K] fixes the reveal order, otherwise random.
    func sampleFrame(
        _ hidden: MLXArray, parameters: LFM3OmniSpeechParameters, history: MLXArray?,
        order: MLXArray? = nil
    ) -> MLXArray {
        let k = config.numCodebooks
        let order = order ?? MLXRandom.permutation(k).reshaped(1, k)
        let history = history?.swappedAxes(-1, -2)
        var codes = MLXArray.zeros([1, k], dtype: .int32)
        var revealed = MLXArray.zeros([1, k], dtype: .bool)
        var start = 0
        for size in config.maskgitSchedule {
            let pick = putAlong(
                MLXArray.zeros([1, k], dtype: .bool), order[0..., start ..< (start + size)],
                values: MLXArray(true), axis: -1)
            let logits = depthDecoder(hidden, codes: codes, masked: .!revealed)
            let sampled = Self.sample(
                logits, temperature: parameters.temperature, topK: parameters.topK,
                repetitionPenalty: parameters.repetitionPenalty, history: history)
            codes = MLX.where(pick, sampled.asType(.int32), codes)
            revealed = revealed .|| pick
            start += size
        }
        return codes
    }

    /// Plain TTS: speaks `promptTokens` (a chat prompt ending in the assistant
    /// turn's opening) and returns the frames' codes [K, T], EOS frame excluded.
    public func generateCodes(
        model: LFM3Omni, promptTokens: [Int], parameters: LFM3OmniSpeechParameters = .init(),
        frameOrder: [[Int]]? = nil, stats: inout LFM3OmniSpeechStats,
        shouldContinue: () -> Bool = { true }
    ) -> MLXArray {
        var ids = promptTokens
        if ids.last != config.audioStartTokenId { ids.append(config.audioStartTokenId) }
        stats.promptTokens = ids.count
        let cache = model.newCache(parameters: nil)
        let start = Date()

        var hidden = model.speechHidden(
            model.speechEmbedTokens(MLXArray(ids.map(Int32.init)).reshaped(1, -1)), cache: cache)
        eval(hidden)
        stats.prefillMs = Date().timeIntervalSince(start) * 1000
        var history = [MLXArray]()
        var frames = [MLXArray]()
        let framesStart = Date()
        for f in 0 ..< parameters.maxFrames {
            // Checked per frame (80 ms of audio), so a stop never waits on a chunk.
            if !shouldContinue() {
                stats.cancelled = true
                break
            }
            let order = frameOrder.flatMap { f < $0.count ? MLXArray($0[f].map(Int32.init)).reshaped(1, -1) : nil }
            let window = history.suffix(parameters.repetitionWindow)
            let h = window.isEmpty ? nil : stacked(Array(window), axis: 0).expandedDimensions(axis: 0)
            var codes = sampleFrame(hidden, parameters: parameters, history: h, order: order)[0]
            let isEos = (codes .== Int32(config.audioEosCode)).any()
            codes = MLX.where(isEos, MLXArray(Int32(config.audioEosCode)), codes)
            eval(codes)
            if f == 0 { stats.firstFrameMs = Date().timeIntervalSince(start) * 1000 }
            if isEos.item(Bool.self) {
                stats.endedOnEos = true
                break
            }
            history.append(codes)
            frames.append(codes)
            hidden = model.speechHidden(embedFrames(codes).reshaped(1, 1, -1), cache: cache)
        }
        stats.frames = frames.count
        stats.framesMs = Date().timeIntervalSince(framesStart) * 1000
        guard !frames.isEmpty else {
            return MLXArray.zeros([config.numCodebooks, 0], dtype: .int32)
        }
        return stacked(frames, axis: -1)
    }

    /// Decodes codes [K, T] to float32 24 kHz samples.
    public func decode(_ codes: MLXArray) -> MLXArray {
        guard codes.dim(-1) > 0 else { return MLXArray.zeros([0], dtype: .float32) }
        return codec(codes.expandedDimensions(axis: 0))[0]
    }

    public var sampleRate: Int { codecConfig.samplingRate }

    /// Omni's plain-TTS prompt (its card): the system message names the
    /// language and voice, the user turn holds the text. `generateCodes` adds
    /// `<|audio_start|>`.
    public static func ttsPromptTokens(
        tokenizer: any Tokenizer, text: String, voice: String = "female",
        language: String = "American English"
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(messages: [
            ["role": "system", "content": "Perform TTS in \(language) using a \(voice) voice."],
            ["role": "user", "content": text],
        ])
    }
}

// MARK: - Backbone access for speech

extension LFM3Omni {
    /// Token embeddings from the backbone's table.
    func speechEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        speechLanguageModel.model.embedTokens(tokens)
    }

    /// The backbone's last normalized hidden state [B, D] for `embeddings`.
    func speechHidden(_ embeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        let h = speechLanguageModel.model(embeddings, cache: cache)
        return h[0..., h.dim(1) - 1, 0...]
    }
}

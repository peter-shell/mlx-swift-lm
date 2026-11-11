//
//  LFM3OmniProcessor.swift
//  mlx-swift-lm
//

// The input side of LFM3-Omni: images at native resolution up to a cap, audio
// clips as log-mel features, and the chat template's media placeholders
// expanded to one token per feature row. Ports mlx-vlm's `lfm3_omni`,
// `lfm3_vl` and `lfm3_audio` processors (0.7.12).

@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Tokenizers

// MARK: - Audio features

/// LFM3-Omni's audio front end: 16 kHz mono, 128 Slaney mel bins every 10 ms
/// (a 400-sample periodic Hann window, hop 160), `log10` floored at −12, then
/// `(x + 4) / 4`. The encoder reads one token per 8 feature frames, 12.5 a
/// second.
public enum LFM3OmniAudioFeatures {
    public static let sampleRate = 16_000
    public static let nFFT = 400
    public static let hop = 160
    public static let melBins = 128
    static let logFloor: Float = -12

    /// Feature frames of a clip of `samples`: it is right-padded to
    /// `400 + k * 160` samples.
    public static func frames(samples: Int) -> Int {
        let over = samples - nFFT
        let steps = over > 0 ? (over + hop - 1) / hop : 0
        return 1 + steps
    }

    /// Encoder tokens (and audio placeholders) for `frames` feature frames.
    public static func encoderTokens(frames: Int) -> Int { (frames + 7) / 8 }

    /// The Slaney-normalized Slaney-scale filterbank, `(melBins, nFFT/2 + 1)`,
    /// built in double precision as the reference's `precise=True` does.
    static let melFilterValues: [Float] = {
        func hzToMel(_ f: Double) -> Double {
            let fSp = 200.0 / 3
            let minLogHz = 1000.0
            let minLogMel = minLogHz / fSp
            let logStep = log(6.4) / 27.0
            return f >= minLogHz ? minLogMel + log(f / minLogHz) / logStep : f / fSp
        }
        func melToHz(_ m: Double) -> Double {
            let fSp = 200.0 / 3
            let minLogHz = 1000.0
            let minLogMel = minLogHz / fSp
            let logStep = log(6.4) / 27.0
            return m >= minLogMel ? minLogHz * exp(logStep * (m - minLogMel)) : fSp * m
        }
        let nFreqs = nFFT / 2 + 1
        let allFreqs = (0 ..< nFreqs).map {
            Double(sampleRate / 2) * Double($0) / Double(nFreqs - 1)
        }
        let mMax = hzToMel(Double(sampleRate) / 2)
        let fPts = (0 ..< (melBins + 2)).map {
            melToHz(mMax * Double($0) / Double(melBins + 1))
        }
        var bank = [Float](repeating: 0, count: melBins * nFreqs)
        for m in 0 ..< melBins {
            let enorm = 2.0 / (fPts[m + 2] - fPts[m])
            for k in 0 ..< nFreqs {
                let down = (allFreqs[k] - fPts[m]) / (fPts[m + 1] - fPts[m])
                let up = (fPts[m + 2] - allFreqs[k]) / (fPts[m + 2] - fPts[m + 1])
                bank[m * nFreqs + k] = Float(max(0, min(down, up)) * enorm)
            }
        }
        return bank
    }()

    static let windowValues: [Float] = {
        let n = Double(nFFT)
        return (0 ..< nFFT).map { Float(0.5 - 0.5 * cos(2 * .pi * Double($0) / n)) }
    }()

    /// One mono 16 kHz clip → features `(128, frames(samples.count))`.
    public static func logMel(_ samples: [Float]) -> MLXArray {
        let t = frames(samples: samples.count)
        let total = nFFT + (t - 1) * hop
        var padded = samples
        if padded.count < total {
            padded.append(contentsOf: [Float](repeating: 0, count: total - padded.count))
        }
        var framed = [Float](repeating: 0, count: t * nFFT)
        padded.withUnsafeBufferPointer { src in
            framed.withUnsafeMutableBufferPointer { dst in
                for i in 0 ..< t {
                    let from = src.baseAddress! + i * hop
                    (dst.baseAddress! + i * nFFT).update(from: from, count: nFFT)
                }
            }
        }
        // On the CPU: the FFT and the log are where float32 drifts, and the
        // CPU's results do not depend on the GPU family.
        return Device.withDefaultDevice(.cpu) {
            let x = MLXArray(framed, [t, nFFT]) * MLXArray(windowValues)
            let spectrum = rfft(x, axis: -1)
            let power = abs(spectrum).square()
            let filters = MLXArray(melFilterValues, [melBins, nFFT / 2 + 1])
            let mel = matmul(filters, power.transposed())
            let logged = maximum(log10(mel), MLXArray(logFloor))
            return (logged + 4) / 4
        }
    }

    /// Decodes a clip to 16 kHz mono float samples.
    public static func samples(_ audio: UserInput.Audio) throws -> [Float] {
        switch audio {
        case .url(let url):
            let file = try AVAudioFile(forReading: url)
            let format = file.processingFormat
            let count = AVAudioFrameCount(file.length)
            guard count > 0,
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count)
            else {
                throw VLMError.processing("LFM3-Omni: \(url.lastPathComponent) holds no audio")
            }
            try file.read(into: buffer)
            return try resample(buffer)
        case .samples(let samples, let rate):
            guard Int(rate) != sampleRate else { return samples }
            guard
                let format = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1,
                    interleaved: false),
                let buffer = AVAudioPCMBuffer(
                    pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
            else {
                throw VLMError.processing("LFM3-Omni: cannot hold \(rate) Hz samples")
            }
            buffer.frameLength = AVAudioFrameCount(samples.count)
            samples.withUnsafeBufferPointer {
                buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count)
            }
            return try resample(buffer)
        }
    }

    /// Any PCM buffer → 16 kHz mono float32, through `AVAudioConverter`.
    static func resample(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        guard
            let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
                interleaved: false),
            let converter = AVAudioConverter(from: buffer.format, to: target)
        else {
            throw VLMError.processing("LFM3-Omni: cannot convert \(buffer.format)")
        }
        let ratio = Double(sampleRate) / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            throw VLMError.processing("LFM3-Omni: cannot allocate the converted audio")
        }
        var fed = false
        var error: NSError?
        let status = converter.convert(to: out, error: &error) { _, inputStatus in
            if fed {
                inputStatus.pointee = .endOfStream
                return nil
            }
            fed = true
            inputStatus.pointee = .haveData
            return buffer
        }
        if status == .error || error != nil {
            throw VLMError.processing(
                "LFM3-Omni: audio conversion failed: \(error?.localizedDescription ?? "")")
        }
        let n = Int(out.frameLength)
        return Array(UnsafeBufferPointer(start: out.floatChannelData![0], count: n))
    }

    /// Clips → features `(clips, 128, longest)` zero-padded, and each clip's
    /// valid frame count.
    public static func batch(_ clips: [[Float]]) -> LMInput.ProcessedAudio {
        let features = clips.map(logMel)
        let lengths = features.map { $0.dim(-1) }
        let longest = lengths.max() ?? 0
        let padded = features.map { f -> MLXArray in
            let pad = longest - f.dim(-1)
            return pad > 0 ? MLX.padded(f, widths: [0, .init((0, pad))]) : f
        }
        return .init(features: stacked(padded, axis: 0), lengths: lengths)
    }
}

// MARK: - Processor

/// `UserInputProcessor` for ``LFM3Omni``. Typically created by
/// ``VLMModelFactory``.
///
/// Each message's images, then its audio clips, then its text, as the
/// reference's `apply_chat_template` lays them out. An image is resized to a
/// multiple of 32 px between the configured pixel bounds; pass
/// `additionalContext["max_image_tokens"]` to cap it lower (one token per
/// 32×32 px).
public struct LFM3OmniProcessor: UserInputProcessor {
    private let config: LFM3OmniProcessorConfiguration
    private let tokenizer: any Tokenizer

    public static let maxImageTokensKey = "max_image_tokens"

    public init(_ config: LFM3OmniProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func messages(from input: UserInput) -> [Message] {
        func content(_ message: Chat.Message) -> [[String: any Sendable]] {
            var parts = [[String: any Sendable]]()
            parts += message.images.map { _ in ["type": "image"] }
            parts += message.audios.map { _ in ["type": "audio"] }
            parts.append(["type": "text", "text": message.content])
            return parts
        }
        switch input.prompt {
        case .text(let text):
            return [
                [
                    "role": "user",
                    "content": input.images.map { _ in ["type": "image"] as [String: any Sendable] }
                        + input.audios.map { _ in ["type": "audio"] as [String: any Sendable] }
                        + [["type": "text", "text": text]],
                ]
            ]
        case .messages(let messages):
            return messages
        case .chat(let messages):
            return messages.map { ["role": $0.role.rawValue, "content": content($0)] }
        }
    }

    /// One image → its patches and `(1, h, w)` grid in patches.
    func preprocess(image: CIImage, processing: UserInput.Processing?, maxTokens: Int?) throws
        -> (MLXArray, THW)
    {
        let image = MediaProcessing.apply(image, processing: processing)
        let factor = config.patchSize * config.mergeSize
        var maxPixels = config.maxPixels
        if let maxTokens { maxPixels = min(maxPixels, maxTokens * factor * factor) }
        let minPixels = min(config.minPixels, maxPixels)
        let size = image.extent.size
        let (height, width) = try QwenVL.targetSize(
            height: Int(size.height), width: Int(size.width), factor: factor,
            minPixels: minPixels, maxPixels: maxPixels)
        let resized = MediaProcessing.resampleBicubic(
            image.toSRGB(), to: CGSize(width: width, height: height))
        let normalized = MediaProcessing.normalize(
            resized, mean: config.imageMeanTuple, std: config.imageStdTuple)
        return try QwenVL.patchify(
            images: [MediaProcessing.asMLXArray(normalized)], mergeSize: config.mergeSize,
            patchSize: config.patchSize, temporalPatchSize: config.temporalPatchSize)
    }

    /// Replaces the `index`-th occurrence of `token` with `counts[index]` copies.
    static func expand(_ tokens: [Int], token: Int, counts: [Int], label: String) throws -> [Int] {
        let present = tokens.filter { $0 == token }.count
        guard present == counts.count else {
            throw VLMError.processing(
                "LFM3-Omni: \(present) \(label) placeholders for \(counts.count) \(label)s")
        }
        var result = [Int]()
        result.reserveCapacity(tokens.count + counts.reduce(0, +))
        var next = 0
        for t in tokens {
            if t == token {
                result.append(contentsOf: repeatElement(token, count: counts[next]))
                next += 1
            } else {
                result.append(t)
            }
        }
        return result
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages(from: input), tools: input.tools,
            additionalContext: input.additionalContext)

        var image: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let maxTokens = input.additionalContext?[Self.maxImageTokensKey] as? Int
            let processed = try input.images.map {
                try preprocess(
                    image: $0.asCIImage(), processing: input.processing, maxTokens: maxTokens)
            }
            let frames = processed.map(\.1)
            image = .init(pixels: concatenated(processed.map(\.0)), frames: frames)
            let merge = config.mergeSize * config.mergeSize
            tokens = try Self.expand(
                tokens, token: config.imageTokenId,
                counts: frames.map { $0.product / merge }, label: "image")
        }

        var audio: LMInput.ProcessedAudio?
        if !input.audios.isEmpty {
            let clips = try input.audios.map(LFM3OmniAudioFeatures.samples)
            let batch = LFM3OmniAudioFeatures.batch(clips)
            audio = batch
            tokens = try Self.expand(
                tokens, token: config.audioTokenId,
                counts: batch.lengths.map(LFM3OmniAudioFeatures.encoderTokens(frames:)),
                label: "audio")
        }

        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: image, audio: audio)
    }
}

/// `processor_config.json` for ``LFM3OmniProcessor``: its `image_processor`
/// and `feature_extractor` sections. A feature setting other than the ones the
/// front end implements is refused rather than silently ignored.
public struct LFM3OmniProcessorConfiguration: Codable, Sendable {

    struct ImageProcessor: Codable, Sendable {
        struct Size: Codable, Sendable {
            let shortestEdge: Int
            let longestEdge: Int
            enum CodingKeys: String, CodingKey {
                case shortestEdge = "shortest_edge"
                case longestEdge = "longest_edge"
            }
        }
        let imageMean: [CGFloat]
        let imageStd: [CGFloat]
        let mergeSize: Int
        let patchSize: Int
        let temporalPatchSize: Int
        let size: Size
        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case mergeSize = "merge_size"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case size
        }
    }

    struct FeatureExtractor: Codable, Sendable {
        let samplingRate: Int?
        let nFFT: Int?
        let hopLength: Int?
        let featureSize: Int?
        let logFloor: Float?
        let encoderStride: Int?
        enum CodingKeys: String, CodingKey {
            case samplingRate = "sampling_rate"
            case nFFT = "n_fft"
            case hopLength = "hop_length"
            case featureSize = "feature_size"
            case logFloor = "log_floor"
            case encoderStride = "encoder_stride"
        }
    }

    let imageProcessor: ImageProcessor
    let featureExtractor: FeatureExtractor?
    public let imageTokenId = 125030
    public let audioTokenId = 125031

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case featureExtractor = "feature_extractor"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageProcessor = try c.decode(ImageProcessor.self, forKey: .imageProcessor)
        featureExtractor = try c.decodeIfPresent(FeatureExtractor.self, forKey: .featureExtractor)
        if let f = featureExtractor {
            typealias A = LFM3OmniAudioFeatures
            let matches =
                (f.samplingRate ?? A.sampleRate) == A.sampleRate
                && (f.nFFT ?? A.nFFT) == A.nFFT
                && (f.hopLength ?? A.hop) == A.hop
                && (f.featureSize ?? A.melBins) == A.melBins
                && (f.logFloor ?? A.logFloor) == A.logFloor
                && (f.encoderStride ?? 8) == 8
            guard matches else {
                throw DecodingError.dataCorruptedError(
                    forKey: .featureExtractor, in: c,
                    debugDescription: "LFM3-Omni: unsupported feature_extractor settings")
            }
        }
    }

    var patchSize: Int { imageProcessor.patchSize }
    var mergeSize: Int { imageProcessor.mergeSize }
    var temporalPatchSize: Int { imageProcessor.temporalPatchSize }
    var minPixels: Int { imageProcessor.size.shortestEdge }
    var maxPixels: Int { imageProcessor.size.longestEdge }
    var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageProcessor.imageMean[0], imageProcessor.imageMean[1], imageProcessor.imageMean[2])
    }
    var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageProcessor.imageStd[0], imageProcessor.imageStd[1], imageProcessor.imageStd[2])
    }
}

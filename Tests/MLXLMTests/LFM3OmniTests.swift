// LFM3-Omni without the model: the audio front end against a committed
// mlx-vlm reference, and a tiny random-weight configuration through every
// path the real one takes (text, image, audio). Parity on the real weights is
// `MLXLMIntegrationTests/LFM3OmniParityTests`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXVLM
import XCTest

final class LFM3OmniTests: XCTestCase {

    // Placeholders sit inside the tiny vocabulary, so the mask and the scatter
    // are exercised as they are at 128,000.
    static let imageId = 60
    static let videoId = 61
    static let audioId = 62
    static let audioOutId = 63

    static func tinyConfig() throws -> LFM3OmniConfiguration {
        let json = """
            {
              "model_type": "lfm3_omni",
              "image_token_id": \(imageId), "video_token_id": \(videoId),
              "audio_token_id": \(audioId), "audio_out_token_id": \(audioOutId),
              "downsample_factor": 2, "projector_hidden_size": 64,
              "projector_hidden_act": "gelu", "projector_bias": true,
              "text_config": {
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 4, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "conv_L_cache": 3,
                "layer_types": ["conv", "full_attention", "conv", "sliding_attention"],
                "sliding_window": 4, "use_sandwich_norm": true,
                "rope_parameters": {
                  "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
                  "sliding_attention": {"rope_theta": 10000}
                }
              },
              "vision_config": {
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2, "patch_size": 2,
                "temporal_patch_size": 2, "spatial_merge_size": 2,
                "layer_types": ["conv", "full_attention"],
                "rope_parameters": {"rope_type": "axial", "rope_theta": 1000000.0}
              },
              "audio_encoder_config": {
                "conv_channels": 8, "hidden_size": 16, "num_mel_bins": 16,
                "sliding_window": 4,
                "text_config": {
                  "hidden_size": 16, "intermediate_size": 48, "num_hidden_layers": 2,
                  "num_attention_heads": 2, "num_key_value_heads": 1,
                  "layer_types": ["conv", "full_attention"], "block_multiple_of": 16,
                  "rope_parameters": {"rope_theta": 1000000.0}
                }
              }
            }
            """
        return try JSONDecoder().decode(LFM3OmniConfiguration.self, from: Data(json.utf8))
    }

    static func tinyModel() throws -> LFM3Omni {
        MLXRandom.seed(7)
        let model = LFM3Omni(try tinyConfig())
        eval(model)
        return model
    }

    /// One 8×8 image: a `(1, 4, 4)` grid of 2×2 patches, 4 tokens after the merge.
    static func tinyImage() -> LMInput.ProcessedImage {
        MLXRandom.seed(11)
        return .init(
            pixels: MLXRandom.normal([16, 3 * 2 * 2 * 2]), frames: [THW(1, 4, 4)])
    }

    /// Clips of `frames` feature frames each, 16 mel bins.
    static func tinyAudio(_ frames: [Int]) -> LMInput.ProcessedAudio {
        MLXRandom.seed(13)
        let longest = frames.max()!
        var features = MLXRandom.normal([frames.count, 16, longest])
        // Zero past each clip, as `LFM3OmniAudioFeatures.batch` pads.
        let valid = MLXArray(Int32(0) ..< Int32(longest)).expandedDimensions(axis: 0)
            .< MLXArray(frames.map(Int32.init)).expandedDimensions(axis: 1)
        features = features * valid.expandedDimensions(axis: 1).asType(.float32)
        return .init(features: features, lengths: frames)
    }

    static func tokens(_ ids: [Int]) -> LMInput.Text {
        .init(tokens: MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0))
    }

    // MARK: - Audio front end

    func testLogMelMatchesTheReference() {
        let mel = LFM3OmniAudioFeatures.logMel(melFixtureSignal)
        XCTAssertEqual(mel.shape, [128, 6])
        let expected = MLXArray(melFixtureExpected).reshaped(128, 6)
        let worst = abs(mel.asType(.float32) - expected).max().item(Float.self)
        XCTAssertLessThan(worst, 1e-3)
    }

    func testFrameAndTokenCounts() {
        typealias F = LFM3OmniAudioFeatures
        XCTAssertEqual(F.frames(samples: 1), 1)
        XCTAssertEqual(F.frames(samples: 400), 1)
        XCTAssertEqual(F.frames(samples: 401), 2)
        XCTAssertEqual(F.frames(samples: 1200), 6)
        // One second: 99 frames, 13 tokens (12.5 a second).
        XCTAssertEqual(F.frames(samples: 16_000), 99)
        XCTAssertEqual(F.encoderTokens(frames: 1), 1)
        XCTAssertEqual(F.encoderTokens(frames: 8), 1)
        XCTAssertEqual(F.encoderTokens(frames: 9), 2)
        XCTAssertEqual(F.encoderTokens(frames: 99), 13)
    }

    func testBatchPadsToTheLongestClip() {
        let batch = LFM3OmniAudioFeatures.batch([
            Array(melFixtureSignal.prefix(400)), melFixtureSignal,
        ])
        XCTAssertEqual(batch.features.shape, [2, 128, 6])
        XCTAssertEqual(batch.lengths, [1, 6])
        XCTAssertEqual(abs(batch.features[0, 0..., 1...]).max().item(Float.self), 0)
    }

    // MARK: - Configuration

    func testTinyConfigurationDecodes() throws {
        let config = try Self.tinyConfig()
        XCTAssertEqual(config.audioEncoderConfiguration.textConfiguration.feedForwardDim, 32)
        XCTAssertEqual(config.placeholderTokenIds.sorted(), [60, 61, 62, 63])
    }

    func testRejectsAVisionTowerWithoutAxialRope() {
        let json = """
            {"hidden_size": 32, "num_hidden_layers": 1, "layer_types": ["conv"],
             "rope_parameters": {"rope_type": "default"}}
            """
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LFM3OmniConfiguration.VisionConfiguration.self, from: Data(json.utf8)))
    }

    // MARK: - Forward pass

    func testTextOnlyLogitsNeverPickAPlaceholder() throws {
        let model = try Self.tinyModel()
        let input = LMInput(text: Self.tokens([1, 5, 9, 2, 7, 3]))
        guard case .logits(let out) = try model.prepare(
            input, cache: model.newCache(parameters: nil), windowSize: nil)
        else { return XCTFail("no logits") }
        XCTAssertEqual(out.logits.shape, [1, 6, 64])

        let rows = out.logits[0].asType(.float32)
        let real = rows[0..., 0 ..< Self.imageId]
        XCTAssertTrue(isFinite(real).all().item(Bool.self))
        XCTAssertTrue((rows[0..., Self.imageId...] .== -Float.infinity).all().item(Bool.self))
        let picks = argMax(rows, axis: -1).asArray(Int32.self)
        XCTAssertTrue(picks.allSatisfy { $0 < Self.imageId })
    }

    func testImageAndAudioFillTheirPlaceholdersOnly() throws {
        let model = try Self.tinyModel()
        let image = Self.tinyImage()
        let audio = Self.tinyAudio([20])  // 3 tokens
        let ids = [1, 2] + Array(repeating: Self.imageId, count: 4) + [3]
            + Array(repeating: Self.audioId, count: 3) + [4, 5]
        let input = LMInput(text: Self.tokens(ids), image: image, audio: audio)
        XCTAssertTrue(input.carriesMedia)

        let embedded = try model.inputEmbeddings(input).asType(.float32)
        let plain = try model.inputEmbeddings(LMInput(text: Self.tokens(ids)))
            .asType(.float32)
        XCTAssertEqual(embedded.shape, [1, ids.count, 32])

        let vision = try model.encodeImages(image).asType(.float32)
        let sound = model.encodeAudio(audio).asType(.float32)
        XCTAssertEqual(vision.shape, [4, 32])
        XCTAssertEqual(sound.shape, [3, 32])
        XCTAssertEqual(abs(embedded[0, 2 ..< 6] - vision).max().item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(abs(embedded[0, 7 ..< 10] - sound).max().item(Float.self), 0, accuracy: 1e-6)
        for i in [0, 1, 6, 10, 11] {
            XCTAssertEqual(
                abs(embedded[0, i] - plain[0, i]).max().item(Float.self), 0, "position \(i)")
        }

        guard case .logits(let out) = try model.prepare(
            input, cache: model.newCache(parameters: nil), windowSize: nil)
        else { return XCTFail("no logits") }
        let real = out.logits[0, 0..., 0 ..< Self.imageId].asType(.float32)
        XCTAssertTrue(isFinite(real).all().item(Bool.self))
    }

    func testAPlaceholderCountMismatchThrows() throws {
        let model = try Self.tinyModel()
        let ids = [1] + Array(repeating: Self.imageId, count: 3) + [2]
        let input = LMInput(text: Self.tokens(ids), image: Self.tinyImage())
        XCTAssertThrowsError(try model.inputEmbeddings(input))

        let audioIds = [1, Self.audioId, 2]
        XCTAssertThrowsError(
            try model.inputEmbeddings(
                LMInput(text: Self.tokens(audioIds), audio: Self.tinyAudio([20]))))
    }

    func testPaddingAShorterClipLeavesItsRowsUnchanged() throws {
        let model = try Self.tinyModel()
        let both = Self.tinyAudio([20, 40])
        let short = LMInput.ProcessedAudio(
            features: both.features[0 ..< 1, 0..., 0 ..< 20], lengths: [20])
        let batched = model.encodeAudio(both).asType(.float32)
        let alone = model.encodeAudio(short).asType(.float32)
        XCTAssertEqual(batched.dim(0), 3 + 5)
        XCTAssertEqual(alone.dim(0), 3)
        XCTAssertLessThan(abs(batched[0 ..< 3] - alone).max().item(Float.self), 1e-4)
    }

    func testNewCacheHonoursMaxKVSize() throws {
        let model = try Self.tinyModel()
        let bounded = model.newCache(parameters: GenerateParameters(maxKVSize: 8))
        XCTAssertTrue(bounded[0] is MambaCache)
        XCTAssertEqual(bounded[1].maxSize, 8)
        XCTAssertTrue(bounded[2] is MambaCache)
        XCTAssertEqual(bounded[3].maxSize, 4)  // the sliding window is smaller

        let small = model.newCache(parameters: GenerateParameters(maxKVSize: 2))
        XCTAssertEqual(small[3].maxSize, 2)

        let unbounded = model.newCache(parameters: nil)
        XCTAssertTrue(unbounded[1] is KVCacheSimple)
        XCTAssertEqual(unbounded[3].maxSize, 4)
    }

    // MARK: - Weights

    func testSanitizeKeepsOnlyTheInputSide() throws {
        let model = try Self.tinyModel()
        let w = MLXArray.zeros([2, 2])
        let sanitized = model.sanitize(weights: [
            "language_model.model.embed_tokens.weight": w,
            "language_model.lm_head.weight": w,
            "vision_tower.patchifier.weight": w,
            "multi_modal_projector.linear_1.weight": w,
            "audio_encoder.lfm.embedding_norm.weight": w,
            "audio_projector.linear_1.weight": w,
            "depth_decoder.layers.0.weight": w,
            "codebook_embedding.weight": w,
            "audio_tokenizer.decoder.weight": w,
            "something_else.weight": w,
        ])
        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "language_model.model.embed_tokens.weight",
                "vision_tower.patchifier.weight",
                "multi_modal_projector.linear_1.weight",
                "audio_tower.lfm.embedding_norm.weight",
                "audio_projector.linear_1.weight",
            ])
    }

    func testTheModelLoadsItsOwnSanitizedParameters() throws {
        let model = try Self.tinyModel()
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let sanitized = model.sanitize(weights: weights)
        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
    }

    // MARK: - Processor

    func testExpandRepeatsEachPlaceholderByItsCount() throws {
        let expanded = try LFM3OmniProcessor.expand(
            [1, 9, 2, 9, 3], token: 9, counts: [2, 3], label: "image")
        XCTAssertEqual(expanded, [1, 9, 9, 2, 9, 9, 9, 3])
        XCTAssertThrowsError(
            try LFM3OmniProcessor.expand([1, 9, 2], token: 9, counts: [1, 1], label: "image"))
    }

    func testUserInputCollectsAudioFromChat() {
        let clip = UserInput.Audio.samples([0, 0.5], sampleRate: 16_000)
        let input = UserInput(chat: [
            .system("s"),
            .user("first", audios: [clip]),
            .assistant("ok"),
            .user("second", audios: [clip, clip]),
        ])
        XCTAssertEqual(input.audios.count, 3)
        XCTAssertTrue(input.images.isEmpty)
    }
}

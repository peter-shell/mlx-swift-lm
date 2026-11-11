// Parity of the LFM3-Omni port against mlx-vlm 0.7.12 on the same 4-bit weights.
//
// Skipped unless all three are set:
//   LFM3_OMNI_DIR       a local LiquidAI/LFM3-Omni-2B-MLX-4bit checkout
//   LFM3_OMNI_FIXTURES  holding clip.wav and sign.png
//   LFM3_OMNI_REF       mlx-vlm's dump of the same cases (asr, image), each a
//                       <case>.safetensors of prompt ids, model inputs, encoder
//                       outputs, last-position logits and the greedy tokens
// LFM3_OMNI_OUT, if set, receives the measurements as JSON.
//
// Each stage is compared on the reference's own inputs, so a difference is
// that stage's: the processor (ids exact, mel features by max abs error), the
// vision and audio encoders (cosine), the full prompt (last-position logits,
// cosine and top-10), and greedy decoding (identical tokens), both from the
// reference inputs and from this processor's.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class LFM3OmniParityTests: XCTestCase {

    struct Env {
        let model: URL
        let fixtures: URL
        let reference: URL
        let out: URL?
    }

    static func env() -> Env? {
        let e = ProcessInfo.processInfo.environment
        guard let m = e["LFM3_OMNI_DIR"], let f = e["LFM3_OMNI_FIXTURES"],
            let r = e["LFM3_OMNI_REF"]
        else { return nil }
        return Env(
            model: URL(fileURLWithPath: m), fixtures: URL(fileURLWithPath: f),
            reference: URL(fileURLWithPath: r),
            out: e["LFM3_OMNI_OUT"].map { URL(fileURLWithPath: $0) })
    }

    /// Over the entries finite in both: the placeholder logits are −∞ in each.
    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        var a = a.asType(.float32).flattened()
        var b = b.asType(.float32).flattened()
        let finite = (abs(a) .< 1e30) .&& (abs(b) .< 1e30)
        a = MLX.where(finite, a, 0)
        b = MLX.where(finite, b, 0)
        return ((a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))).item(Float.self)
    }

    static func topK(_ logits: MLXArray, _ k: Int) -> [Int] {
        let v = logits.asType(.float32).asArray(Float.self)
        return Array(v.indices.sorted { v[$0] > v[$1] }.prefix(k))
    }

    /// Greedy decoding from `input` with a cold cache, stopping at an EOS.
    static func greedy(
        _ model: LFM3Omni, _ input: LMInput, eos: Set<Int>, maxTokens: Int = 64
    ) throws -> [Int] {
        let cache = model.newCache(parameters: nil)
        guard case .logits(let first) = try model.prepare(input, cache: cache, windowSize: nil)
        else { return [] }
        var next = argMax(first.logits[0, -1]).item(Int.self)
        var tokens = [Int]()
        while tokens.count < maxTokens, !eos.contains(next) {
            tokens.append(next)
            let step = model(MLXArray([Int32(next)]).expandedDimensions(axis: 0), cache: cache)
            next = argMax(step[0, -1]).item(Int.self)
        }
        return tokens
    }

    func testParityWithMlxVlm() async throws {
        guard let env = Self.env() else {
            throw XCTSkip("set LFM3_OMNI_DIR, LFM3_OMNI_FIXTURES and LFM3_OMNI_REF")
        }
        let context = try await VLMModelFactory.shared.load(
            configuration: ModelConfiguration(directory: env.model))
        let model = try XCTUnwrap(context.model as? LFM3Omni)
        // mlx-vlm also stops where the model would start to speak.
        var eos = context.configuration.eosTokenIds
        for token in ["<|audio_start|>", "<|mixed_start|>", "<|im_end|>"] {
            if let id = context.tokenizer.convertTokenToId(token) { eos.insert(id) }
        }

        var report = [String: [String: any Sendable]]()
        for name in ["asr", "image"] {
            let ref = try loadArrays(url: env.reference.appending(component: "\(name).safetensors"))
            let meta = try JSONSerialization.jsonObject(
                with: Data(contentsOf: env.reference.appending(component: "\(name).json")))
                as! [String: Any]
            var r = [String: any Sendable]()

            // The processor, from the same files mlx-vlm read.
            let system = meta["system"] as! String
            let prompt = meta["prompt"] as! String
            let user: Chat.Message =
                name == "asr"
                ? .user(prompt, audios: [.url(env.fixtures.appending(component: "clip.wav"))])
                : .user(prompt, images: [.url(env.fixtures.appending(component: "sign.png"))])
            let ours = try await context.processor.prepare(
                input: UserInput(chat: [.system(system), user]))
            let refIds = ref["input_ids"]!.flattened().asArray(Int32.self).map(Int.init)
            let ourIds = ours.text.tokens.flattened().asArray(Int32.self).map(Int.init)
            r["promptTokens"] = refIds.count
            r["idsIdentical"] = refIds == ourIds
            if refIds != ourIds {
                let first = zip(refIds, ourIds).enumerated().first { $0.element.0 != $0.element.1 }
                r["idsFirstDifference"] = first?.offset ?? min(refIds.count, ourIds.count)
                r["ourPromptTokens"] = ourIds.count
            }

            // The reference's own model inputs.
            var refInput = LMInput(text: .init(tokens: ref["input_ids"]!))
            if let pixels = ref["pixel_values"], let grid = ref["image_grid_thw"] {
                let g = grid.flattened().asArray(Int32.self).map(Int.init)
                let image = LMInput.ProcessedImage(pixels: pixels, frames: [THW(g[0], g[1], g[2])])
                refInput = LMInput(text: refInput.text, image: image)
                r["visionCosine"] = Self.cosine(
                    try model.encodeImages(image), ref["vision_embeddings"]!)
                if let ourPixels = ours.image?.pixels, ourPixels.shape == pixels.shape {
                    r["pixelsMaxAbs"] = abs(ourPixels - pixels).max().item(Float.self)
                } else {
                    r["pixelsShape"] = "\(ours.image?.pixels.shape ?? []) vs \(pixels.shape)"
                }
            }
            if let features = ref["input_features"], let lengths = ref["feature_lengths"] {
                let audio = LMInput.ProcessedAudio(
                    features: features,
                    lengths: lengths.flattened().asArray(Int32.self).map(Int.init))
                refInput = LMInput(text: refInput.text, audio: audio)
                r["audioCosine"] = Self.cosine(model.encodeAudio(audio), ref["audio_embeddings"]!)
                if let ourFeatures = ours.audio?.features, ourFeatures.shape == features.shape {
                    r["melMaxAbs"] = abs(ourFeatures - features).max().item(Float.self)
                } else {
                    r["melShape"] = "\(ours.audio?.features.shape ?? []) vs \(features.shape)"
                }
                // The front end on the reference's own samples.
                if let samples = ref["audio_samples"] {
                    let mel = LFM3OmniAudioFeatures.logMel(samples.asArray(Float.self))
                    let refMel = features[0, 0..., 0 ..< mel.dim(-1)]
                    r["melFromRefSamplesMaxAbs"] = abs(mel - refMel).max().item(Float.self)
                }
            }

            // The whole prompt, cold.
            let cache = model.newCache(parameters: nil)
            guard case .logits(let out) = try model.prepare(refInput, cache: cache, windowSize: nil)
            else { return XCTFail("no logits") }
            let last = out.logits[0, -1]
            let refLast = ref["last_logits"]!
            r["logitsCosine"] = Self.cosine(last, refLast)
            r["top10Identical"] = Self.topK(last, 10) == Self.topK(refLast, 10)

            let refGreedy = ref["greedy_tokens"]!.asArray(Int32.self).map(Int.init)
            let fromRef = try Self.greedy(model, refInput, eos: eos)
            let fromOurs = try Self.greedy(model, ours, eos: eos)
            r["greedyFromRefInputsIdentical"] = fromRef == refGreedy
            r["greedyFromOurInputsIdentical"] = fromOurs == refGreedy
            r["greedyText"] = context.tokenizer.decode(tokens: fromOurs)
            r["referenceText"] = meta["greedy_text"] as? String ?? ""
            report[name] = r
        }

        let printable = report.mapValues { r in
            r.mapValues { v -> any Sendable in
                if let f = v as? Float, !f.isFinite { return "\(f)" }
                return v
            }
        }
        let data = try JSONSerialization.data(
            withJSONObject: printable, options: [.prettyPrinted, .sortedKeys])
        if let out = env.out { try data.write(to: out) }
        print(String(decoding: data, as: UTF8.self))

        for (name, r) in report {
            XCTAssertEqual(r["greedyFromRefInputsIdentical"] as? Bool, true, name)
            XCTAssertGreaterThan(r["logitsCosine"] as? Float ?? 0, 0.999, name)
        }
    }
}

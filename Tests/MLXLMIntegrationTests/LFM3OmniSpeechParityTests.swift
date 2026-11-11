// Parity of LFM3-Omni's speech head against
// mlx-vlm 0.7.12 on the same 4-bit weights and bf16 speech modules.
//
// Skipped unless both are set:
//   LFM3_OMNI_DIR         an LFM3-Omni-2B-MLX-4bit folder with audio_tokenizer/
//   LFM3_OMNI_SPEECH_REF  mlx-vlm's dump (sr2_ref.safetensors)
// LFM3_OMNI_OUT, if set, receives the measurements as JSON and the WAVs.
//
// Stages, each on the reference's own inputs: the prefill hidden state
// (cosine), depth-decoder logits on a fixed state (cosine, argmax agreement),
// greedy frames with a fixed reveal order (codes identical), and the codec on
// fixed codes (max abs error, SNR).

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class LFM3OmniSpeechParityTests: XCTestCase {

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a = a.asType(.float32).flattened()
        let b = b.asType(.float32).flattened()
        return ((a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))).item(Float.self)
    }

    static func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = UInt32(samples.count * 2)
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes)
        data.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(bytes)
        for s in samples { u16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
        try data.write(to: url)
    }

    func testSpeechParityWithMlxVlm() async throws {
        let e = ProcessInfo.processInfo.environment
        guard let dir = e["LFM3_OMNI_DIR"], let refPath = e["LFM3_OMNI_SPEECH_REF"] else {
            throw XCTSkip("set LFM3_OMNI_DIR and LFM3_OMNI_SPEECH_REF")
        }
        let modelDir = URL(fileURLWithPath: dir)
        let out = e["LFM3_OMNI_OUT"].map { URL(fileURLWithPath: $0) }
        let ref = try loadArrays(url: URL(fileURLWithPath: refPath))

        let context = try await VLMModelFactory.shared.load(
            configuration: ModelConfiguration(directory: modelDir))
        let model = try XCTUnwrap(context.model as? LFM3Omni)
        let speech = try LFM3OmniSpeech.load(directory: modelDir)
        var report = [String: Any]()

        // 1. Prefill hidden state.
        let ids = ref["prompt_ids"]!.asArray(Int32.self).map(Int.init)
        let cache = model.newCache(parameters: nil)
        let hidden = model.speechHidden(
            model.speechEmbedTokens(MLXArray(ids.map(Int32.init)).reshaped(1, -1)), cache: cache)
        let hiddenCos = Self.cosine(hidden, ref["hidden"]!)
        report["hiddenCosine"] = hiddenCos
        XCTAssertGreaterThan(hiddenCos, 0.999)

        // 2. Depth logits on the reference's own hidden state.
        let logits = speech.depthDecoder(
            ref["hidden"]!.asType(.bfloat16), codes: ref["depth_codes"]!,
            masked: ref["depth_masked"]!.asType(.bool))
        let depthCos = Self.cosine(logits, ref["depth_logits"]!)
        let agree = (argMax(logits, axis: -1) .== argMax(ref["depth_logits"]!, axis: -1))
            .asType(.int32).sum().item(Int.self)
        report["depthLogitCosine"] = depthCos
        report["depthArgmaxAgree"] = "\(agree)/12"
        XCTAssertGreaterThan(depthCos, 0.999)

        // 3. Greedy frames with the reference's reveal order.
        let refGreedy = ref["greedy_codes"]!  // [N, 12]
        let order = ref["frame_order"]!
        let n = refGreedy.dim(0)
        let orders = (0 ..< n).map { order[$0].asArray(Int32.self).map(Int.init) }
        var params = LFM3OmniSpeechParameters()
        params.temperature = 0
        params.maxFrames = n
        var stats = LFM3OmniSpeechStats()
        let ours = speech.generateCodes(
            model: model, promptTokens: Array(ids.dropLast()), parameters: params,
            frameOrder: orders, stats: &stats)  // [12, T]
        let oursRows = ours.transposed(1, 0).asArray(Int32.self)
        let refRows = refGreedy.asArray(Int32.self)
        var firstDiff: Int? = nil
        for f in 0 ..< min(ours.dim(1), n) {
            if Array(oursRows[(f * 12) ..< (f * 12 + 12)]) != Array(refRows[(f * 12) ..< (f * 12 + 12)]) {
                firstDiff = f
                break
            }
        }
        var codeAgree = 0
        for i in 0 ..< min(oursRows.count, refRows.count) where oursRows[i] == refRows[i] {
            codeAgree += 1
        }
        report["greedyFrames"] = "\(ours.dim(1))/\(n)"
        report["greedyFirstDifferentFrame"] = firstDiff ?? -1
        report["greedyCodesAgree"] = "\(codeAgree)/\(n * 12)"

        // 3b. Teacher-forced: each frame from the reference's earlier frames, so a
        // near-tie cannot compound.
        do {
            let cache = model.newCache(parameters: nil)
            var h = model.speechHidden(
                model.speechEmbedTokens(MLXArray(ids.map(Int32.init)).reshaped(1, -1)),
                cache: cache)
            var forcedAgree = 0
            var framesExact = 0
            for f in 0 ..< n {
                let refFrame = refGreedy[f]
                let hist: MLXArray? =
                    f == 0 ? nil
                    : refGreedy[max(0, f - 64) ..< f].expandedDimensions(axis: 0)
                let c = speech.sampleFrame(
                    h, parameters: params, history: hist,
                    order: MLXArray(orders[f].map(Int32.init)).reshaped(1, -1))[0]
                let same = (c .== refFrame).asType(.int32).sum().item(Int.self)
                forcedAgree += same
                if same == 12 { framesExact += 1 }
                h = model.speechHidden(speech.embedFrames(refFrame).reshaped(1, 1, -1), cache: cache)
            }
            report["forcedCodesAgree"] = "\(forcedAgree)/\(n * 12)"
            report["forcedFramesExact"] = "\(framesExact)/\(n)"
        }

        // 4. Codec on fixed codes.
        let audio = speech.decode(ref["codec_codes"]!)
        let refAudio = ref["codec_audio"]!
        let maxAbs = abs(audio - refAudio).max().item(Float.self)
        let noise = ((audio - refAudio) * (audio - refAudio)).mean().item(Float.self)
        let signal = (refAudio * refAudio).mean().item(Float.self)
        let snr = 10 * log10(signal / max(noise, 1e-20))
        report["codecSamples"] = "\(audio.dim(0))/\(refAudio.dim(0))"
        report["codecMaxAbs"] = maxAbs
        report["codecSNRdB"] = snr
        XCTAssertEqual(audio.dim(0), refAudio.dim(0))
        XCTAssertGreaterThan(snr, 20)

        print("SR2 parity: \(report)")
        if let out {
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: out.appending(path: "speech_parity.json"))
            try Self.writeWAV(
                audio.asArray(Float.self), sampleRate: speech.sampleRate,
                to: out.appending(path: "codec_fixed.wav"))
        }
    }
}

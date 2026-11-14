// Parity of the d1 port (`D1Decider`) against mlx-vlm 0.7.12 on the same
// LiquidAI/d1-2B-MLX-4bit weights.
//
// Skipped unless both are set:
//   D1_DIR     a local LiquidAI/d1-2B-MLX-4bit checkout
//   D1_GOLDEN  mlx-vlm's cases, written by Mortise's
//              tool/decision_bench/golden.py (test/fixtures/d1_golden.json)
// D1_OUT, if set, receives the measurements as JSON.
//
// Each case is one prompt mlx-vlm answered: a state as text, one or two
// questions, the packed ids, and each answer's probabilities.
//
// The ids, the state's length and the question groups must be identical. Until
// the fork required swift-transformers 1.4.0, which honours LFM3's BPE
// `ignore_merges` (#397), one case split " Genehmigung" into three tokens where
// Python kept one, and this test let a same-text split through. It no longer does.
//
// Every answer must pick the same option. Each probability must be within 5e-3
// of mlx-vlm's, and the median within 5e-4. The two sides run different MLX
// cores (mlx-swift 0.31.6 against Python mlx 0.32.4, the oldest mlx-vlm 0.7.12
// accepts), so kernels can differ in the last bits of fp16. 1e-3 was the
// plan's guess; the measured median is about 1e-4.

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import XCTest

final class D1DeciderParityTests: XCTestCase {

    func testParityWithMlxVlm() async throws {
        let e = ProcessInfo.processInfo.environment
        guard let dir = e["D1_DIR"], let golden = e["D1_GOLDEN"] else {
            throw XCTSkip("set D1_DIR and D1_GOLDEN")
        }
        let decider = try await D1Decider.load(directory: URL(fileURLWithPath: dir))
        let data = try Data(contentsOf: URL(fileURLWithPath: golden))
        let cases = try JSONDecoder().decode(Golden.self, from: data).cases

        var idMismatches = [String]()
        var pickMismatches = [String]()
        var maxDelta = 0.0
        var worst = ""
        var deltas = [Double]()
        var ms = [Double]()
        for c in cases {
            let questions = c.questions.map(\.d1)
            let row = try decider.pack(stateText: c.state_text, questions: questions)
            if row.ids != c.ids || row.stateLength != c.state_len || row.groups != c.groups {
                let at =
                    (0 ..< min(row.ids.count, c.ids.count)).first { row.ids[$0] != c.ids[$0] }
                    ?? min(row.ids.count, c.ids.count)
                let ours = row.ids[at ..< min(at + 4, row.ids.count)].map { $0 }
                let theirs = c.ids[at ..< min(at + 4, c.ids.count)].map { $0 }
                idMismatches.append(
                    "\(c.id)@\(at): ours \(ours) \(decider.tokenizer.decode(tokens: ours).debugDescription)"
                        + " mlx-vlm \(theirs) \(decider.tokenizer.decode(tokens: theirs).debugDescription)")
                continue
            }
            let start = Date()
            let (answers, _) = try decider.decide(stateText: c.state_text, questions: questions)
            ms.append(Date().timeIntervalSince(start) * 1000)
            for (q, a) in zip(c.questions, answers) {
                let ref = c.answers[q.name]!
                let refProbs = ref.probabilities(labels: a.labels)
                let delta = zip(a.probabilities, refProbs).map { abs($0 - $1) }.max()!
                let refPick = a.labels[refProbs.indices.max { refProbs[$0] < refProbs[$1] }!]
                if a.choice != refPick { pickMismatches.append("\(c.id).\(q.name)") }
                deltas.append(delta)
                if delta > maxDelta { worst = "\(c.id).\(q.name)" }
                maxDelta = max(maxDelta, delta)
            }
        }
        ms.sort()
        deltas.sort()
        let report: [String: Any] = [
            "cases": cases.count,
            "id_mismatches": idMismatches,
            "pick_mismatches": pickMismatches,
            "max_probability_delta": maxDelta,
            "worst_case": worst,
            "deltas_over_1e-3": deltas.filter { $0 > 1e-3 }.count,
            "median_probability_delta": deltas.isEmpty ? 0 : deltas[deltas.count / 2],
            "median_ms": ms.isEmpty ? 0 : ms[ms.count / 2],
        ]
        print("D1 parity:", report)
        if let out = e["D1_OUT"] {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: out))
        }
        XCTAssertEqual(idMismatches, [], "prompt ids differ from mlx-vlm's")
        XCTAssertEqual(pickMismatches, [], "a different option won")
        XCTAssertLessThanOrEqual(maxDelta, 5e-3, "a probability moved more than 5e-3")
        XCTAssertLessThanOrEqual(
            deltas.isEmpty ? 1 : deltas[deltas.count / 2], 5e-4, "the median moved more than 5e-4")
    }
}

/// The golden file, decoded with each question's options in criteria order.
struct Golden: Decodable {
    struct Question {
        let name: String
        let d1: D1Question
    }

    struct Answer: Decodable {
        let type: String
        let yes: Double?
        let probabilities: [String: Double]?

        func probabilities(labels: [String]) -> [Double] {
            if type == "noul" { return [yes!, 1 - yes!] }
            return labels.map { probabilities![$0]! }
        }
    }

    struct Case: Decodable {
        let id: String
        let state_text: String
        let questions: [Question]
        let ids: [Int]
        let state_len: Int
        let groups: [Int]
        let answers: [String: Answer]

        enum Keys: String, CodingKey {
            case id, state_text, questions, ids, state_len, groups, answers
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: Keys.self)
            id = try c.decode(String.self, forKey: .id)
            state_text = try c.decode(String.self, forKey: .state_text)
            ids = try c.decode([Int].self, forKey: .ids)
            state_len = try c.decode(Int.self, forKey: .state_len)
            groups = try c.decode([Int].self, forKey: .groups)
            answers = try c.decode([String: Answer].self, forKey: .answers)
            questions = try OrderedQuestions(from: c.superDecoder(forKey: .questions)).items
        }
    }

    let cases: [Case]
}

/// Questions and criteria in file order. A keyed container lists `allKeys` in an
/// unspecified order, so the order is read back from the raw JSON text.
struct OrderedQuestions: Decodable {
    let items: [Golden.Question]

    struct Key: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    struct Raw: Decodable {
        let type: String
        let instructions: String
        let criteria: [String: String?]?
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        let order = Self.questionOrder
        let keys = c.allKeys.sorted {
            (order[$0.stringValue] ?? 99) < (order[$1.stringValue] ?? 99)
        }
        items = try keys.map { key in
            let raw = try c.decode(Raw.self, forKey: key)
            let d1: D1Question
            if raw.type == "noul" {
                d1 = .noul(instructions: raw.instructions)
            } else {
                let labels = Self.criteriaOrder.filter { raw.criteria?[$0] != nil }
                precondition(labels.count == raw.criteria?.count, "unknown criteria label")
                d1 = .choice(
                    instructions: raw.instructions,
                    options: labels.map { ($0, raw.criteria![$0]!) })
            }
            return Golden.Question(name: key.stringValue, d1: d1)
        }
    }

    /// The bank's names and criteria, in the order `build_bank.py` writes them.
    static let questionOrder = ["decision": 0, "needs_documents": 0, "ask_kind": 1]
    static let criteriaOrder = [
        "whole_document", "enumeration", "specific", "follow_up", "social",
    ]
}

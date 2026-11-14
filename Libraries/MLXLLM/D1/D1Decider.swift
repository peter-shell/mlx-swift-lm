// d1's System One on the fork's LFM3: named yes/no and choice questions over a
// state, answered in one forward pass with no tokens generated.
//
// Ported from mlx-vlm 0.7.12 (`mlx_vlm/models/d1_lfm3_vl`), which adapts the
// system_one code LiquidAI ships with its d1 checkpoints under the LFM Open
// License v1.0. Only what a text decision needs is here:
// - the `noul` and single `choice` questions;
// - the `json_only` state style with no system turn, as the checkpoint
//   defaults;
// - `set` attention over a packed row.
//
// Reads, estimates, boxes, multiple choice, scores and media are not ported.
// `D1DeciderParityTests` holds every answer to mlx-vlm's on the same weights.

import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

/// One question for d1: a yes/no (`noul`) or a pick of one option (`choice`).
public enum D1Question: Sendable, Equatable {
    /// `criteria`, when given, says what yes and no mean.
    case noul(instructions: String, yes: String? = nil, no: String? = nil)
    /// Options in order: a label and an optional description, which the prompt
    /// shows in place of the label.
    case choice(instructions: String, options: [(label: String, description: String?)])

    public static func == (a: D1Question, b: D1Question) -> Bool {
        switch (a, b) {
        case let (.noul(i, y, n), .noul(j, z, m)): return i == j && y == z && n == m
        case let (.choice(i, o), .choice(j, p)):
            return i == j && o.map(\.label) == p.map(\.label)
                && o.map(\.description) == p.map(\.description)
        default: return false
        }
    }
}

/// d1's answer to one question.
public struct D1Answer: Sendable, Equatable {
    /// The options' probabilities in question order: yes then no for a `noul`.
    public let labels: [String]
    public let probabilities: [Double]

    /// A `noul`'s P(yes); nil for a choice.
    public var yes: Double? { labels == ["yes", "no"] ? probabilities[0] : nil }

    /// The likeliest option's label (`yes` or `no` for a noul).
    public var choice: String { labels[best] }

    /// The likeliest option's probability.
    public var confidence: Double { probabilities[best] }

    var best: Int { probabilities.indices.max { probabilities[$0] < probabilities[$1] }! }
}

/// A packed prompt: what `D1Decider.decide` runs, exposed for the parity test.
public struct D1Row: Sendable {
    public let ids: [Int]
    public let stateLength: Int
    /// Each token's question: −1 on the state, 1... after it.
    public let groups: [Int]
    /// Each question's answer position.
    public let slots: [Int]
}

public enum D1Error: Error, LocalizedError {
    case notD1(String)
    case noSingleTokenCode(Int)
    case noYesNo

    public var errorDescription: String? {
        switch self {
        case .notD1(let why): return "not a d1 LFM3 checkpoint: \(why)"
        case .noSingleTokenCode(let n): return "no single-token code left for \(n) options"
        case .noYesNo: return "the tokenizer has no single-token yes/no"
        }
    }
}

/// `calibration.json`: a temperature per question bucket, and the default.
struct D1Calibration {
    var temperatures: [String: Double] = [:]
    var defaultTemperature: Double = 1.0

    init() {}

    init(url: URL) throws {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let d = object as? [String: Any] ?? [:]
        temperatures = (d["temperatures"] as? [String: Any] ?? [:]).compactMapValues {
            ($0 as? NSNumber)?.doubleValue
        }
        defaultTemperature = (d["default"] as? NSNumber)?.doubleValue ?? 1.0
    }

    static func bucket(_ q: D1Question) -> String {
        switch q {
        case .noul: return "noul"
        case .choice(_, let options):
            switch options.count {
            case ...3: return "choice:small"
            case ...8: return "choice:mid"
            case ...26: return "choice:large"
            default: return "choice:inventory"
            }
        }
    }

    func temperature(_ q: D1Question) -> Double {
        temperatures[Self.bucket(q)] ?? defaultTemperature
    }
}

public final class D1Decider: @unchecked Sendable {
    public let model: LFM3Model
    public let tokenizer: Tokenizer
    let calibration: D1Calibration
    let setSize: Int
    let slidingWindow: Int
    let codeScheme: Int
    let bos: String
    /// Special tokens' strings, longest first: broken with a zero-width space
    /// wherever the caller's text holds one (`plain`).
    let specials: [String]

    static let imStart = "<|im_start|>"
    static let imEnd = "<|im_end|>"
    /// What LFM3's template writes before a non-thinking answer.
    static let lead = "<think></think>"
    static let header = "QUESTION:\n"
    static let zeroWidthSpace = "\u{200B}"
    static let yesForms = ["yes", "Yes", "YES"]
    static let noForms = ["no", "No", "NO"]

    init(
        model: LFM3Model, tokenizer: Tokenizer, calibration: D1Calibration, setSize: Int,
        slidingWindow: Int, codeScheme: Int, specials: [String]
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.calibration = calibration
        self.setSize = setSize
        self.slidingWindow = slidingWindow
        self.codeScheme = codeScheme
        self.bos = tokenizer.bosToken ?? ""
        self.specials = specials
    }

    /// Loads a d1 LFM3 checkpoint from a local directory: its text model only.
    /// The vision tower and projector are left unread.
    public static func load(directory: URL) async throws -> D1Decider {
        let configURL = directory.appending(component: "config.json")
        let data = try Data(contentsOf: configURL)
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = config["text_config"]
        else { throw D1Error.notD1("config.json has no text_config") }
        guard (config["architectures"] as? [String])?.contains("D1Model") == true else {
            throw D1Error.notD1("architectures does not name D1Model")
        }
        if let attention = config["d1_attention"] as? String, attention != "set" {
            throw D1Error.notD1("d1_attention is \(attention), not set")
        }
        let textData = try JSONSerialization.data(withJSONObject: text)
        let args = try JSONDecoder().decode(LFM3Configuration.self, from: textData)
        let model = LFM3Model(args)

        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try loadWeights(
            modelDirectory: directory, model: model, quantization: base.quantization,
            perLayerQuantization: base.perLayerQuantization)

        let tokenizer = try await loadTokenizer(
            configuration: ModelConfiguration(directory: directory), hub: HubApi())

        let calibrationURL = directory.appending(component: "calibration.json")
        let calibration =
            FileManager.default.fileExists(atPath: calibrationURL.path)
            ? try D1Calibration(url: calibrationURL) : D1Calibration()

        return D1Decider(
            model: model, tokenizer: tokenizer, calibration: calibration,
            setSize: (config["d1_set_size"] as? Int) ?? 16,
            slidingWindow: args.slidingWindow,
            codeScheme: (config["d1_codes"] as? Int) ?? 1,
            specials: try specialTokens(directory: directory))
    }

    /// The special tokens' strings from `tokenizer.json`, longest first.
    static func specialTokens(directory: URL) throws -> [String] {
        let url = directory.appending(component: "tokenizer.json")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let added = (object as? [String: Any])?["added_tokens"] as? [[String: Any]] ?? []
        let names = added.compactMap { t -> String? in
            guard t["special"] as? Bool == true, let s = t["content"] as? String,
                s.count > 1, !s.contains(zeroWidthSpace)
            else { return nil }
            return s
        }
        return Array(Set(names)).sorted { $0.count != $1.count ? $0.count > $1.count : $0 < $1 }
    }

    // MARK: - Prompt

    /// `text` with every special token's string broken after its first
    /// character, so the caller's text can never close a turn.
    func plain(_ text: String) -> String {
        guard !specials.isEmpty else { return text }
        var current = text
        while true {
            var out = ""
            var changed = false
            var i = current.startIndex
            outer: while i < current.endIndex {
                for name in specials where current[i...].hasPrefix(name) {
                    let first = name.index(after: name.startIndex)
                    out += name[..<first] + Self.zeroWidthSpace + name[first...]
                    i = current.index(i, offsetBy: name.count)
                    changed = true
                    continue outer
                }
                out.append(current[i])
                i = current.index(after: i)
            }
            if !changed { return out }
            current = out
        }
    }

    func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// `texts` that are one token each, first come.
    func singleTokenIds(_ texts: [String]) -> [Int] {
        var out = [Int]()
        for t in texts {
            let ids = encode(t)
            if ids.count == 1, !out.contains(ids[0]) { out.append(ids[0]) }
        }
        return out
    }

    static let letters = (0 ..< 26).map { String(UnicodeScalar(UInt8(65 + $0))) }

    /// Native letters, else A–Z, else the checkpoint's scheme (`option_codes`).
    func optionCodes(_ labels: [String]) -> [String] {
        let labs = labels.map { $0.trimmingCharacters(in: .whitespaces) }
        if !labs.isEmpty, labs.allSatisfy({ $0.count == 1 && $0.first!.isLetter }) {
            return labs
        }
        if labs.count <= 26 { return Array(Self.letters.prefix(labs.count)) }
        if codeScheme == 2 {
            let pairs = Self.letters.flatMap { a in Self.letters.map { a + $0 } }
            return Array((Self.letters + pairs).prefix(labs.count))
        }
        return (0 ..< labs.count).map { String(format: "%02d", $0) }
    }

    static let fallbackPool: [String] =
        letters + (0 ..< 100).map { String(format: "%02d", $0) }
        + (0 ..< 26).map { String(UnicodeScalar(UInt8(97 + $0))) }
        + (0 ..< 200).map { "#\($0)" }
        + letters.flatMap { a in letters.map { a + $0 } }

    /// Each label's distinct single-token code and its id (`aliases`).
    func aliases(_ labels: [String]) throws -> [(code: String, id: Int)] {
        var used = Set<Int>()
        var out = [(code: String, id: Int)]()
        func take(_ raw: String) -> Bool {
            let ids = encode(raw)
            guard ids.count == 1, !used.contains(ids[0]) else { return false }
            out.append((raw, ids[0]))
            used.insert(ids[0])
            return true
        }
        for code in optionCodes(labels) where !take(code) {
            if !Self.fallbackPool.contains(where: take) {
                throw D1Error.noSingleTokenCode(labels.count)
            }
        }
        return out
    }

    /// The state block and the opening of the first question.
    func prefix(stateText: String) -> String {
        "\(bos)\(Self.imStart)user\n\(plain(stateText + "\n\n"))\n\(Self.header)"
    }

    /// The question and the assistant header, up to the answer slot.
    func suffix(_ q: D1Question) throws -> String {
        let body: String
        switch q {
        case let .noul(instructions, yes, no):
            let extra = (yes != nil || no != nil) ? "\nYes: \(yes ?? "None")\nNo: \(no ?? "None")" : ""
            body = "\(instructions)\(extra)\n\nReply with yes or no only."
        case let .choice(instructions, options):
            let codes = try aliases(options.map(\.label))
            let lines = zip(codes, options).map { code, option in
                "\(code.code) \(option.description ?? option.label.replacingOccurrences(of: "_", with: " "))"
            }
            body = "\(instructions)\n\nOptions:\n\(lines.joined(separator: "\n"))\n\nReply with the option code only."
        }
        return "\(plain(body))\(Self.imEnd)\n\(Self.imStart)assistant\n\(Self.lead)"
    }

    /// The token groups scored at a question's slot, one per option, max-pooled.
    func readoutGroups(_ q: D1Question) throws -> [[Int]] {
        switch q {
        case .noul:
            let yes = singleTokenIds(Self.yesForms)
            let no = singleTokenIds(Self.noForms)
            guard !yes.isEmpty, !no.isEmpty else { throw D1Error.noYesNo }
            return [yes, no]
        case .choice(_, let options):
            return try aliases(options.map(\.label)).map { code in
                [code.id] + singleTokenIds([" " + code.code]).filter { $0 != code.id }
            }
        }
    }

    /// One row for up to `setSize` questions: the state and the first question
    /// tokenized as one string, each further one as its own `QUESTION:` block.
    public func pack(stateText: String, questions: [D1Question]) throws -> D1Row {
        precondition(!questions.isEmpty, "a question set cannot be empty")
        let first = try suffix(questions[0])
        var ids = encode(prefix(stateText: stateText) + first)
        let stateLength = ids.count - encode(first).count
        var groups = Array(repeating: -1, count: stateLength)
            + Array(repeating: 1, count: ids.count - stateLength)
        var slots = [ids.count - 1]
        for (k, q) in questions.dropFirst().enumerated() {
            let own = encode(Self.header + (try suffix(q)))
            ids += own
            groups += Array(repeating: k + 2, count: own.count)
            slots.append(ids.count - 1)
        }
        return D1Row(ids: ids, stateLength: stateLength, groups: groups, slots: slots)
    }

    // MARK: - Forward pass and readout

    /// Global layers read the state and a question's own tokens both ways, and
    /// the state reads everything. Sliding layers stay causal in their window.
    func masks(_ row: D1Row) -> (
        full: MLXFast.ScaledDotProductAttentionMaskMode,
        sliding: MLXFast.ScaledDotProductAttentionMaskMode
    ) {
        let n = row.ids.count
        let pos = MLXArray(Int32(0) ..< Int32(n))
        let group = MLXArray(row.groups.map(Int32.init))
        let s = Int32(row.stateLength)
        let full =
            (pos[0..., .newAxis] .< s) .|| (pos[.newAxis] .< s)
            .|| (group[0..., .newAxis] .== group[.newAxis])
        let sliding: MLXFast.ScaledDotProductAttentionMaskMode =
            n <= slidingWindow
            ? .causal : .array(createCausalMask(n: n, offset: 0, windowSize: slidingWindow))
        return (.array(full), sliding)
    }

    /// Each question's probabilities, for a row built by `pack`.
    func read(_ row: D1Row, questions: [D1Question]) throws -> [D1Answer] {
        let (full, sliding) = masks(row)
        let inputs = MLXArray(row.ids.map(Int32.init)).expandedDimensions(axis: 0)
        let hidden = model.model.hidden(inputs, fullMask: full, slidingMask: sliding)[0]

        var answers = [D1Answer]()
        for (q, slot) in zip(questions, row.slots) {
            let groups = try readoutGroups(q)
            var tokens = [Int]()
            for id in groups.joined() where !tokens.contains(id) { tokens.append(id) }
            let rows = model.model.embedTokens(MLXArray(tokens.map(Int32.init)))
            let logits = matmul(hidden[slot ..< slot + 1], rows.transposed())[0]
                .asType(.float32).asArray(Float.self)
            let t = calibration.temperature(q)
            let scores = groups.map { g in
                Double(g.map { logits[tokens.firstIndex(of: $0)!] }.max()!) / t
            }
            let m = scores.max()!
            let exps = scores.map { exp($0 - m) }
            let total = exps.reduce(0, +)
            let labels: [String]
            switch q {
            case .noul: labels = ["yes", "no"]
            case .choice(_, let options): labels = options.map(\.label)
            }
            answers.append(D1Answer(labels: labels, probabilities: exps.map { $0 / total }))
        }
        return answers
    }

    /// Answers every question about `stateText` in rows of up to `setSize`
    /// questions, returning the answers in order and the tokens read.
    ///
    /// `stateText` is the state as the prompt shows it. A JSON state is
    /// rendered as Python's `json.dumps(state, ensure_ascii=False, indent=2)`
    /// renders it, the form the checkpoint was trained on.
    public func decide(stateText: String, questions: [D1Question]) throws -> (
        answers: [D1Answer], inputTokens: Int
    ) {
        var answers = [D1Answer]()
        var read = 0
        for start in stride(from: 0, to: questions.count, by: setSize) {
            let group = Array(questions[start ..< min(start + setSize, questions.count)])
            let row = try pack(stateText: stateText, questions: group)
            read += row.ids.count
            answers += try self.read(row, questions: group)
        }
        return (answers, read)
    }
}

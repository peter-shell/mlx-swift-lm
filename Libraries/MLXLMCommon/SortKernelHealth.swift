// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Whether this device's Metal `argSort` / `argPartition` can be trusted.
///
/// On Apple GPU family 7 — an A14 iPad, measured — both return wrong indices:
/// out-of-range values in the billions at 1,024 elements, and in-range but
/// mis-ordered values at 151,936. Crucially the wrong answer **differs on every
/// invocation** for a byte-identical input, which points at a race or
/// uninitialized memory rather than a logic error. Every other op in the
/// sampling path (`cumsum`, `takeAlong`, `putAlong`, `logSoftmax`, `argMax`,
/// `categorical`) is correct on the same device, and all of them including
/// these two are correct on M-series.
///
/// The user-visible consequence is severe and silent: `TopPSampler` is built on
/// exactly these two ops, so a model producing a perfectly correct logit
/// distribution emits garbage tokens, with nothing thrown and nothing logged.
///
/// This exists so the sampler can route around them rather than the whole
/// device losing sampled generation.
public enum SortKernelHealth {

    /// Evaluated once, lazily, on first use.
    ///
    /// A `let` rather than a computed property on purpose: the check costs a
    /// GPU round trip, and it cannot change for the lifetime of the process.
    public static let isTrustworthy: Bool = check()

    /// Sorts a small known array and verifies the result is actually ordered.
    ///
    /// 1,024 elements is deliberate — the failure is present at that size, and
    /// checking there costs microseconds rather than sorting a whole
    /// vocabulary. Values are distinct so a correct sort has exactly one
    /// answer.
    private static func check() -> Bool {
        let count = 1024
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        let values = (0 ..< count).map { _ -> Float in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((state >> 40) & 0xFF_FFFF) / Float(0x100_0000)
        }

        let indices = argSort(MLXArray(values, [1, count]), axis: -1)
        eval(indices)
        let order = indices.asArray(UInt32.self)
        guard order.count == count else { return false }

        for i in 0 ..< (count - 1) {
            let a = Int(order[i]), b = Int(order[i + 1])
            guard a < count, b < count else { return false }
            if values[a] > values[b] { return false }
        }
        return true
    }
}

/// Top-k / top-p selection done without `argSort` or `argPartition`.
///
/// Used only where [SortKernelHealth] reports those kernels are broken. The
/// selection runs on the CPU; the draw itself stays on the GPU via
/// `categorical`, which is correct on the affected hardware, so sampling
/// semantics and the seeded random state are preserved.
///
/// **This is exact, not an approximation, whenever top-k is in play.** Nucleus
/// sampling keeps a prefix of the descending order, and top-k keeps the first
/// `k` of that same order, so their intersection is a prefix of the top `k` —
/// which is computable from the top `k` alone. When `topK` is 0 there is no such
/// bound and the caller must not use this path.
enum UnsortedSelection {

    /// Indices of the `k` largest values, descending, via a bounded insertion.
    ///
    /// O(n·k) with k small (20 in practice) and no allocation per element,
    /// which beats sorting 151,936 floats to keep 20 of them.
    static func topIndices(_ values: [Float], k: Int) -> [Int] {
        guard k > 0 else { return [] }
        var best = [Int]()
        best.reserveCapacity(k)

        for i in values.indices {
            let v = values[i]
            if best.count == k, v <= values[best[best.count - 1]] { continue }
            var position = best.count
            while position > 0, values[best[position - 1]] < v {
                position -= 1
            }
            best.insert(i, at: position)
            if best.count > k { best.removeLast() }
        }
        return best
    }

    /// Applies top-p and min-p to `candidates` (already descending by value).
    ///
    /// Mirrors `apply_top_p` / `apply_min_p` from `mlx_lm/sample_utils.py`:
    /// a token survives top-p when the probability mass of everything strictly
    /// better than it is still below `topP`, which keeps the crossing token.
    static func filter(
        _ candidates: [Int],
        logprobs: [Float],
        topP: Float?,
        minP: Float?
    ) -> [Int] {
        guard let first = candidates.first else { return candidates }
        var kept = candidates

        if let minP {
            let threshold = logprobs[first] + Foundation.log(minP)
            kept = kept.filter { logprobs[$0] >= threshold }
        }

        if let topP, topP > 0, topP < 1 {
            var massBefore: Float = 0
            var cut = kept.count
            for (offset, index) in kept.enumerated() {
                if massBefore >= topP {
                    cut = offset
                    break
                }
                massBefore += Foundation.exp(logprobs[index])
            }
            kept = Array(kept.prefix(max(1, cut)))
        }

        return kept
    }
}

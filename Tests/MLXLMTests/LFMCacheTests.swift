import Foundation
import MLXLLM
import MLXLMCommon
import Testing

/// `newCache` delegates to `makeCache`. Building the model allocates weights
/// and loads the GPU metallib, which `swift test` does not package, so these
/// tests stop at the decoded configuration.
@Test func lfm2NewCacheHonorsMaxKVSize() throws {
    let config = try lfm2Config()

    let unbounded = config.makeCache(parameters: nil)
    #expect(unbounded[0] is KVCacheSimple)
    #expect(unbounded[0].maxSize == nil)
    #expect(unbounded[1] is MambaCache)

    let capped = config.makeCache(parameters: GenerateParameters(maxKVSize: 512))
    #expect(capped[0] is RotatingKVCache)
    #expect(capped[0].maxSize == 512)
    #expect(capped[1] is MambaCache)
    #expect(capped[1].maxSize == nil)
}

@Test func lfm2MoENewCacheHonorsMaxKVSize() throws {
    let config = try lfm2MoEConfig()

    let unbounded = config.makeCache(parameters: nil)
    #expect(unbounded[0] is KVCacheSimple)
    #expect(unbounded[1] is MambaCache)

    let capped = config.makeCache(parameters: GenerateParameters(maxKVSize: 256))
    #expect(capped[0] is RotatingKVCache)
    #expect(capped[0].maxSize == 256)
    #expect(capped[1] is MambaCache)
}

@Test func lfm3NewCacheHonorsMaxKVSize() throws {
    let config = try lfm3Config(slidingWindow: 128)

    let unbounded = config.makeCache(parameters: nil)
    #expect(unbounded[0] is KVCacheSimple)
    #expect(unbounded[0].maxSize == nil)
    #expect(unbounded[1] is RotatingKVCache)
    #expect(unbounded[1].maxSize == 128)
    #expect(unbounded[2] is MambaCache)

    let capped = config.makeCache(parameters: GenerateParameters(maxKVSize: 512))
    #expect(capped[0] is RotatingKVCache)
    #expect(capped[0].maxSize == 512)
    // The sliding window is tighter than the requested ceiling, so it stays.
    #expect(capped[1] is RotatingKVCache)
    #expect(capped[1].maxSize == 128)
    #expect(capped[2] is MambaCache)

    let tighter = try lfm3Config(slidingWindow: 128)
        .makeCache(parameters: GenerateParameters(maxKVSize: 64))
    #expect(tighter[0].maxSize == 64)
    #expect(tighter[1].maxSize == 64)
}

private func lfm2Config() throws -> LFM2Configuration {
    let json = """
        {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "norm_eps": 1e-5,
            "layer_types": ["full_attention", "conv"]
        }
        """
    return try JSONDecoder().decode(LFM2Configuration.self, from: Data(json.utf8))
}

private func lfm2MoEConfig() throws -> LFM2MoEConfiguration {
    let json = """
        {
            "model_type": "lfm2_moe",
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "norm_topk_prob": true,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "use_expert_bias": false,
            "num_dense_layers": 0,
            "norm_eps": 1e-5,
            "conv_bias": false,
            "conv_L_cache": 3,
            "layer_types": ["full_attention", "conv"]
        }
        """
    return try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(json.utf8))
}

private func lfm3Config(slidingWindow: Int) throws -> LFM3Configuration {
    let json = """
        {
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "norm_eps": 1e-5,
            "layer_types": ["full_attention", "sliding_attention", "conv"],
            "sliding_window": \(slidingWindow),
            "rope_parameters": {
                "full_attention": {"rope_theta": 10000.0},
                "sliding_attention": {"rope_theta": 10000.0}
            }
        }
        """
    return try JSONDecoder().decode(LFM3Configuration.self, from: Data(json.utf8))
}

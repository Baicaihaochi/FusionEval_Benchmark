from ..decoder_linear import (
    ATTN_LEAVES,
    MLP_LEAVES,
    assert_decoder_linears,
    count_decoder_linears,
    decoder_linear_kind,
    is_decoder_linear,
    optimize_matrix,
)

QWEN3_4B_LAYER_COUNT = 36
EXPECTED_ATTN_MATRICES = QWEN3_4B_LAYER_COUNT * len(ATTN_LEAVES)
EXPECTED_MLP_MATRICES = QWEN3_4B_LAYER_COUNT * len(MLP_LEAVES)

qwen_linear_kind = decoder_linear_kind
is_qwen_decoder_linear = is_decoder_linear
count_qwen_linears = count_decoder_linears

def assert_qwen3_4b_decoder_linears(keys):
    return assert_decoder_linears(keys, QWEN3_4B_LAYER_COUNT)

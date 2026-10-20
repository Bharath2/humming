import dataclasses
import math

import pytest
import torch

import humming.schema.humming as schema_module
from humming import dtypes, ops
from humming.schema.humming import HummingWeightSchema

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA weight packing")


@pytest.fixture
def small_conversion_chunks(monkeypatch):
    max_unpacked_bytes = 256
    monkeypatch.setattr(schema_module, "_MAX_UNPACKED_WEIGHT_BYTES", max_unpacked_bytes)
    unpack_weight = ops.unpack_weight
    unpacked_sizes = []

    def checked_unpack_weight(inputs, num_bits, outputs=None):
        result = unpack_weight(inputs, num_bits, outputs=outputs)
        unpacked_sizes.append(result.numel() * result.element_size())
        assert unpacked_sizes[-1] <= max_unpacked_bytes
        return result

    monkeypatch.setattr(ops, "unpack_weight", checked_unpack_weight)
    return unpacked_sizes


@pytest.mark.parametrize("num_bits", range(2, 9))
@pytest.mark.parametrize("shape", [(5, 32), (3, 160), (2, 3, 96), (1, 2048)])
@pytest.mark.parametrize("zero_position", [None, 0, -1])
def test_symmetric_integer_conversion(num_bits, shape, zero_position, small_conversion_chunks):
    schema = HummingWeightSchema(b_dtype=f"uint{num_bits}")
    codes = torch.arange(math.prod(shape), device="cuda", dtype=torch.int32)
    codes = (codes % ((1 << num_bits) - 1) + 1).view(shape)
    if zero_position is not None:
        codes.view(-1)[zero_position] = 0
    packed = ops.pack_weight(codes, num_bits)
    packed_before = packed.clone()
    tensors = {
        "weight": packed,
        "weight_scale": torch.ones(shape[:-1] + (1,), device="cuda", dtype=torch.float16),
        "bias": torch.ones(shape[:-1], device="cuda", dtype=torch.float16),
    }
    expected_codes = codes - (1 << (num_bits - 1))
    expected_codes = expected_codes.abs() | ((expected_codes < 0).to(torch.int32) << (num_bits - 1))
    expected_weight = ops.pack_weight(expected_codes, num_bits)

    converted_schema, converted_tensors = schema.convert_symmetric_integer_weight(tensors)

    assert torch.equal(packed, packed_before)
    assert converted_tensors.keys() == tensors.keys()
    for name in ("weight_scale", "bias"):
        assert converted_tensors[name] is tensors[name]
    if zero_position is not None:
        assert converted_schema is schema
        assert converted_tensors is tensors
    else:
        expected_dtype = dtypes.FloatingPointType(
            num_bits=num_bits, exponent_bits=0, mantissa_bits=num_bits - 1
        )
        assert converted_schema == dataclasses.replace(schema, b_dtype=expected_dtype)
        assert converted_tensors is not tensors
        assert converted_tensors["weight"].shape == packed.shape
        assert converted_tensors["weight"].dtype == packed.dtype
        assert converted_tensors["weight"].device == packed.device
        assert torch.equal(converted_tensors["weight"], expected_weight)
    assert len(small_conversion_chunks) > 1 or zero_position == 0


@pytest.mark.parametrize("num_bits", range(2, 9))
@pytest.mark.parametrize("zero_point_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_representable", [False, True])
def test_symmetric_integer_conversion_float_zero_point(
    num_bits, zero_point_dtype, is_representable, small_conversion_chunks
):
    schema = HummingWeightSchema(b_dtype=f"uint{num_bits}", has_zero_point=True, is_fp_zero_point=True)
    codes = torch.ones((5, 64), dtype=torch.int32, device="cuda")
    packed = ops.pack_weight(codes, num_bits)
    midpoint = 1 << (num_bits - 1)
    zero_point_value = midpoint + 0.5 if is_representable else torch.finfo(zero_point_dtype).tiny
    zero_point = torch.tensor([[zero_point_value]], dtype=zero_point_dtype, device="cuda")
    zero_point_before = zero_point.clone()
    tensors = {"weight": packed, "zero_point": zero_point}

    converted_schema, converted_tensors = schema.convert_symmetric_integer_weight(tensors)

    assert torch.equal(zero_point, zero_point_before)
    if is_representable:
        assert converted_schema.b_dtype.is_floating_point_type
        assert converted_tensors is not tensors
        assert torch.equal(converted_tensors["zero_point"], zero_point - midpoint)
        expected_code = (midpoint - 1) | (1 << (num_bits - 1))
        expected_weight = ops.pack_weight(torch.full_like(codes, expected_code), num_bits)
        assert torch.equal(converted_tensors["weight"], expected_weight)
    else:
        assert converted_schema is schema
        assert converted_tensors is tensors


@pytest.mark.parametrize(
    "schema",
    [
        HummingWeightSchema(b_dtype=dtypes.uint1),
        HummingWeightSchema(b_dtype=dtypes.IntegerType(num_bits=9, is_signed=False)),
        HummingWeightSchema(b_dtype=dtypes.float4e2m1),
        HummingWeightSchema(b_dtype=dtypes.uint4, has_zero_point=True),
    ],
)
def test_symmetric_integer_conversion_unsupported_schema(schema, monkeypatch):
    def unexpected_unpack(*args, **kwargs):
        pytest.fail("unsupported schemas must not unpack weights")

    monkeypatch.setattr(ops, "unpack_weight", unexpected_unpack)
    tensors = {}

    converted_schema, converted_tensors = schema.convert_symmetric_integer_weight(tensors)

    assert converted_schema is schema
    assert converted_tensors is tensors


@pytest.mark.parametrize("shape", [(2, 5), (2, 3, 5)])
def test_symmetric_integer_conversion_rejects_invalid_packed_width(shape):
    schema = HummingWeightSchema(b_dtype=dtypes.uint2)
    tensors = {"weight": torch.ones(shape, dtype=torch.int32, device="cuda")}

    with pytest.raises(ValueError, match="Packed weight width must be divisible by the bit width"):
        schema.convert_symmetric_integer_weight(tensors)


def test_symmetric_integer_conversion_rejects_noncontiguous_weight():
    schema = HummingWeightSchema(b_dtype=dtypes.uint2)
    tensors = {"weight": torch.ones((4, 8), dtype=torch.int32, device="cuda").transpose(0, 1)}

    with pytest.raises(ValueError, match="Packed weights must be contiguous"):
        schema.convert_symmetric_integer_weight(tensors)


def test_symmetric_integer_conversion_rejects_non_cuda_weight():
    schema = HummingWeightSchema(b_dtype=dtypes.uint2)
    tensors = {"weight": torch.ones((2, 4), dtype=torch.int32)}

    with pytest.raises(ValueError, match="Packed weights must be CUDA tensors"):
        schema.convert_symmetric_integer_weight(tensors)


@pytest.mark.parametrize("dtype", [torch.int16, torch.int64, torch.float32])
def test_symmetric_integer_conversion_rejects_invalid_weight_dtype(dtype):
    schema = HummingWeightSchema(b_dtype=dtypes.uint2)
    tensors = {"weight": torch.ones((2, 4), dtype=dtype, device="cuda")}

    with pytest.raises(ValueError, match="Packed weights must have dtype torch.int32"):
        schema.convert_symmetric_integer_weight(tensors)


@pytest.mark.parametrize("zero_codes", [False, True])
def test_symmetric_integer_conversion_peak_memory(zero_codes):
    schema = HummingWeightSchema(b_dtype=dtypes.uint2)
    packed = torch.full((131072, 64), 0x55555555, dtype=torch.int32, device="cuda")
    if zero_codes:
        packed[-1, -1] = 0
    tensors = {"weight": packed}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    converted_schema, converted_tensors = schema.convert_symmetric_integer_weight(tensors)
    torch.cuda.synchronize()
    transient_memory = torch.cuda.max_memory_allocated() - initial_memory

    full_unpacked_bytes = packed.numel() * 16 * packed.element_size()
    assert transient_memory < full_unpacked_bytes
    if zero_codes:
        assert converted_schema is schema
        assert converted_tensors is tensors
    else:
        assert converted_schema.b_dtype.is_floating_point_type
        assert torch.all(ops.unpack_weight(converted_tensors["weight"], 2) == 3)

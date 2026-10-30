"""CPU checkpoint-source gates for optional composite FP8 integration."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization import Float8WeightOnlyConfig, quantize_
from diffusers.quantizers.torchao.torchao_quantizer import TorchAoHfQuantizer
from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from diffusers import TorchAoConfig

from diffusers_dynamic_offloader.dynamic_offload import _safetensor_file_slices
from diffusers_dynamic_offloader.quantized_sources import describe_float8_sources_cpu


@pytest.fixture(autouse=True)
def no_cuda():
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
        yield
    assert not torch.cuda.is_initialized()


def checkpoint(tmp_path, split=False):
    model = torch.nn.Sequential(torch.nn.Linear(32, 16, bias=False, dtype=torch.bfloat16)).eval()
    model.requires_grad_(False)
    quantize_(model, Float8WeightOnlyConfig(set_inductor_config=False))
    tensors, metadata = TorchAoHfQuantizer.get_state_dict_and_metadata(
        None, model.state_dict(), safe_serialization=True)
    assert set(tensors) == {"0._weight_qdata", "0._weight_scale"}
    paths = []
    batches = [{key: value} for key, value in tensors.items()] if split else [tensors]
    for index, batch in enumerate(batches):
        path = (tmp_path / f"shard{index}.safetensors").resolve()
        save_file(batch, str(path), metadata=metadata)
        paths.append(path)
    available = _safetensor_file_slices(paths)
    mapping = {"qdata": available["0._weight_qdata"], "scale": available["0._weight_scale"]}
    return model[0].weight, paths, mapping


@pytest.mark.parametrize("split", [False, True])
def test_actual_diffusers_serialization_sources_and_roundtrip(tmp_path, split):
    weight, paths, mapping = checkpoint(tmp_path, split)
    contract = describe_float8_sources_cpu(weight, mapping)
    assert contract.physical_bytes == 576
    assert len(contract.sources) == 2
    quantizer = TorchAoHfQuantizer(TorchAoConfig(Float8WeightOnlyConfig(set_inductor_config=False)))
    quantizer.set_metadata([str(path) for path in paths])
    diffusers_restored = {}
    data, metadata = {}, {}
    for path in paths:
        shard = load_file(str(path), device="cpu")
        data.update(shard)
        diffusers_restored.update(quantizer.maybe_update_state_dict(shard))
        with safe_open(str(path), framework="pt", device="cpu") as file:
            metadata.update(file.metadata())
    restored, remaining = unflatten_tensor_state_dict(data, metadata)
    assert not remaining
    copy = restored["0.weight"]
    assert not quantizer._pending_flattened_state_dict
    diffusers_copy = diffusers_restored["0.weight"]
    assert torch.equal(diffusers_copy.qdata.view(torch.uint8), weight.qdata.view(torch.uint8))
    assert torch.equal(diffusers_copy.scale, weight.scale)
    assert torch.equal(copy.qdata.view(torch.uint8), weight.qdata.view(torch.uint8))
    assert torch.equal(copy.scale, weight.scale)
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(torch.nn.functional.linear(inputs, copy),
                               torch.nn.functional.linear(inputs, weight), rtol=0, atol=0)


@pytest.mark.parametrize("change", ["missing", "bytes", "dtype", "shape", "offset", "relative", "overlap"])
def test_incomplete_or_conflicting_source_refused(tmp_path, change):
    weight, _paths, mapping = checkpoint(tmp_path)
    source = mapping["scale"]
    if change == "missing":
        mapping.pop("scale")
    elif change == "bytes":
        mapping["scale"] = replace(source, size_bytes=source.size_bytes + 4)
    elif change == "dtype":
        mapping["scale"] = replace(source, dtype="BF16")
    elif change == "shape":
        mapping["scale"] = replace(source, shape=(16,))
    elif change == "offset":
        mapping["scale"] = replace(source, offset_bytes=-1)
    elif change == "relative":
        mapping["scale"] = replace(source, path=type(source.path)("relative.safetensors"))
    else:
        mapping["scale"] = replace(source, offset_bytes=mapping["qdata"].offset_bytes)
    with pytest.raises(ValueError):
        describe_float8_sources_cpu(weight, mapping)

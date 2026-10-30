import gc
import json
import weakref
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.core_model_loading import Transpose, WeightConverter

from diffusers_dynamic_offloader import from_pretrained_with_dynamic_offload
from diffusers_dynamic_offloader.dynamic_offload import _attach_safetensor_file_slices, _safetensor_file_slices
from diffusers_dynamic_offloader.weight_source_metadata import associate_source_metadata


class TinyModel(PreTrainedModel):
    config_class = PretrainedConfig
    base_model_prefix = "model"

    def __init__(self):
        super().__init__(PretrainedConfig())
        self.projection = torch.nn.Linear(3, 2, bias=False)


def checkpoint(tmp_path, tensors, filename="weights.safetensors"):
    path = tmp_path / filename
    save_file(tensors, str(path))
    return path


def test_exact_name_descriptor_bytes_and_metadata(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    path = checkpoint(tmp_path, {"weight": model.weight.detach()})
    with patch("torch.cuda.init", side_effect=AssertionError("CUDA")), \
         patch("torch.Tensor.to", side_effect=AssertionError("copy")):
        assert _attach_safetensor_file_slices(model, [path]) == (1, 1)
    descriptor = model._ddo_safetensor_slices.lookup(model.weight)
    assert descriptor.shape == (2, 3)
    assert descriptor.dtype == "F32"
    with path.open("rb") as source:
        source.seek(descriptor.offset_bytes)
        payload = source.read(descriptor.size_bytes)
    assert payload == model.weight.detach().numpy().tobytes()


def test_transformers_mapping_uses_loader_api_without_architecture_prefixes(tmp_path):
    model = TinyModel()
    path = checkpoint(tmp_path, {"old_arbitrary_name": model.projection.weight.detach()})
    assert _attach_safetensor_file_slices(model, [path], key_mapping={
        r"^old_arbitrary_name$": "projection.weight"}) == (1, 1)
    assert model._ddo_safetensor_slice_summary["resolver"] == "transformers_name_mapping"
    assert model._ddo_safetensor_slices.lookup(model.projection.weight).path == path


def test_loader_wrapper_forwards_same_mapping_to_binding(tmp_path):
    model = TinyModel()
    checkpoint(tmp_path, {"source_weight": model.projection.weight.detach()})
    mapping = {r"^source_weight$": "projection.weight"}
    calls = []

    def loader(path, **kwargs):
        calls.append(kwargs)
        return model

    result = from_pretrained_with_dynamic_offload(tmp_path, model_loader=loader, key_mapping=mapping)
    assert calls == [{"key_mapping": mapping}]
    assert result.module._ddo_safetensor_slices.lookup(model.projection.weight) is not None
    with patch("torch.cuda.init", side_effect=AssertionError("CUDA")):
        report = result.metrics_snapshot()
    assert report.as_dict()["setup"]["weight_source"]["bound"] == 1
    assert "not native execution support" in report.format_profile()
    report.setup["weight_source"]["header"]["invalid_files"] = -1
    assert result.metrics_snapshot().as_dict()["setup"]["weight_source"]["header"]["invalid_files"] == 0
    detached = report.as_dict()
    detached["setup"]["weight_source"]["bound"] = -1
    assert result.metrics_snapshot().as_dict()["setup"]["weight_source"]["bound"] == 1


@pytest.mark.parametrize("source_name", ["embedding.weight", "head.weight"])
def test_shared_parameter_alias_can_supply_source(tmp_path, source_name):
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(5, 3)
    model.head = torch.nn.Linear(3, 5, bias=False)
    model.head.weight = model.embedding.weight
    path = checkpoint(tmp_path, {source_name: model.head.weight.detach()})
    assert _attach_safetensor_file_slices(model, [path]) == (1, 1)
    assert model._ddo_safetensor_slices.lookup(model.embedding.weight) == model._ddo_safetensor_slices.lookup(model.head.weight)


def test_conflicting_shared_parameter_aliases_are_rejected(tmp_path):
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.ones(4))
    model.second = model.first
    path = checkpoint(tmp_path, {"first": torch.ones(4), "second": torch.zeros(4)})
    assert _attach_safetensor_file_slices(model, [path]) == (0, 2)
    assert model._ddo_safetensor_slice_summary["rejected_storage_groups"] == 1
    assert not hasattr(model, "_ddo_safetensor_slices")


def test_distinct_parameters_sharing_storage_cannot_hide_conflicting_sources(tmp_path):
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.ones(4))
    model.second = torch.nn.Parameter(model.first.detach())
    path = checkpoint(tmp_path, {"first": torch.ones(4), "second": torch.zeros(4)})
    assert _attach_safetensor_file_slices(model, [path])[0] == 0


@pytest.mark.parametrize("change", ["dtype", "shape"])
def test_equal_byte_count_is_not_sufficient(tmp_path, change):
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    source = torch.zeros(2, 2, dtype=torch.float16) if change == "dtype" else torch.zeros(4, dtype=torch.bfloat16)
    path = checkpoint(tmp_path, {"weight": source})
    assert _attach_safetensor_file_slices(model, [path])[0] == 0
    assert model._ddo_safetensor_slice_summary["rejected_storage_groups"] == 1


def test_two_loader_keys_cannot_silently_converge(tmp_path):
    model = TinyModel()
    path = checkpoint(tmp_path, {"one": torch.ones(2, 3), "two": torch.zeros(2, 3)})
    mapping = {r"^one$": "projection.weight", r"^two$": "projection.weight"}
    assert _attach_safetensor_file_slices(model, [path], key_mapping=mapping)[0] == 0


def test_payload_conversion_refuses_direct_file_binding(tmp_path):
    model = TinyModel()
    path = checkpoint(tmp_path, {"projection.weight": torch.ones(2, 3)})
    transform = WeightConverter(source_patterns=r"projection.weight", target_patterns="projection.weight",
                                operations=[Transpose(0, 1)])
    with patch("transformers.conversion_mapping.get_model_conversion_mapping", return_value=[transform]):
        assert _attach_safetensor_file_slices(model, [path])[0] == 0
    assert model._ddo_safetensor_slice_summary["resolver"] == "unsupported_payload_transform"


def test_unavailable_loader_mapping_is_fail_closed(tmp_path):
    model = TinyModel()
    path = checkpoint(tmp_path, {"projection.weight": torch.ones(2, 3)})
    with patch("transformers.conversion_mapping.get_model_conversion_mapping", side_effect=TypeError("unsupported API")):
        assert _attach_safetensor_file_slices(model, [path])[0] == 0
    assert model._ddo_safetensor_slice_summary["resolver"] == "unsupported_loader_mapping"


def test_quantized_payload_and_unknown_loader_mapping_are_refused(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    path = checkpoint(tmp_path, {"weight": torch.ones(2, 3)})
    assert _attach_safetensor_file_slices(model, [path], key_mapping={"weight": "weight"})[0] == 0
    model.is_quantized = True
    assert _attach_safetensor_file_slices(model, [path])[0] == 0
    assert model._ddo_safetensor_slice_summary["resolver"] == "unsupported_quantized_payload"


def test_duplicate_shard_keys_refuse_entire_source_set(tmp_path):
    first = checkpoint(tmp_path, {"weight": torch.zeros(2, 3)}, "first.safetensors")
    second = checkpoint(tmp_path, {"weight": torch.ones(2, 3)}, "second.safetensors")
    assert _safetensor_file_slices([first, second]) == {}
    model = torch.nn.Linear(3, 2, bias=False)
    assert _attach_safetensor_file_slices(model, [first, second]) == (0, 0)
    assert model._ddo_safetensor_slice_summary["header"]["ambiguous_key_count"] == 1


def test_meta_parameters_are_not_queried_for_storage_addresses(tmp_path):
    path = checkpoint(tmp_path, {"weight": torch.zeros(2, 3)})
    with torch.device("meta"):
        model = torch.nn.Linear(3, 2, bias=False)
    with patch("torch.Tensor.data_ptr", side_effect=AssertionError("meta pointer")):
        assert _attach_safetensor_file_slices(model, [path])[0] == 0


def test_association_retains_no_tensor_or_open_file(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    path = checkpoint(tmp_path, {"weight": model.weight.detach()})
    reference = weakref.ref(model.weight)
    registry, summary = associate_source_metadata(model, _safetensor_file_slices([path]))
    json.dumps(summary)
    del model
    gc.collect()
    assert reference() is None
    assert registry.descriptor_count == 1
    registry.close()
    path.unlink()


def test_rebinding_cannot_leave_or_replace_existing_source_metadata(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    path = checkpoint(tmp_path, {"weight": model.weight.detach()})
    _attach_safetensor_file_slices(model, [path])
    original = model._ddo_safetensor_slices
    with pytest.raises(RuntimeError, match="already associated"):
        _attach_safetensor_file_slices(model, [])
    assert model._ddo_safetensor_slices is original
    assert not original.closed


@pytest.mark.parametrize("offsets", [[-1, 24], [0, 100000], [True, 24]])
def test_invalid_header_ranges_are_not_descriptors(tmp_path, offsets):
    path = tmp_path / "invalid.safetensors"
    header = json.dumps({"weight": {"shape": [2, 3], "dtype": "F32", "data_offsets": offsets}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(24))
    assert _safetensor_file_slices([path]) == {}

"""Optional CPU-only TorchAO storage gate; not a native runtime adapter."""

import importlib.metadata
import io
import json

import torch
from torchao.quantization import Float8WeightOnlyConfig, quantize_


def main():
    assert not torch.cuda.is_initialized()
    torch.manual_seed(43)
    model = torch.nn.Linear(32, 16, dtype=torch.bfloat16).eval()
    forward = type(model).forward
    source_bytes = model.weight.numel() * model.weight.element_size()
    inputs = torch.randn(4, 32, dtype=torch.bfloat16)
    quantize_(model, Float8WeightOnlyConfig(set_inductor_config=False))
    weight = model.weight
    names, metadata = weight.__tensor_flatten__()
    assert set(names) == {"qdata", "scale"}
    payload = {name: getattr(weight, name) for name in names}
    assert all(item.device.type == "cpu" for item in payload.values())
    assert weight.qdata.dtype == torch.float8_e4m3fn
    assert weight.dtype == torch.bfloat16
    assert weight.act_quant_kwargs is None
    assert type(model).forward is forward
    rebuilt = type(weight).__tensor_unflatten__(
        payload, metadata, weight.size(), weight.stride()
    )
    with torch.inference_mode():
        output = model(inputs)
        # Match upstream weight-only bias ordering, rather than fused addmm.
        reference = torch.matmul(inputs, weight.t().dequantize()) + model.bias
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        torch.testing.assert_close(rebuilt.dequantize(), weight.dequantize(), rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=True)
    copy = torch.nn.Linear(32, 16, dtype=torch.bfloat16).eval()
    copy.load_state_dict(restored, assign=True)
    assert type(copy.weight) is type(weight)
    assert torch.equal(copy.weight.qdata.view(torch.uint8), weight.qdata.view(torch.uint8))
    assert torch.equal(copy.weight.scale, weight.scale)
    with torch.inference_mode():
        torch.testing.assert_close(copy(inputs), output, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
    report = {
        "status": "passed", "scope": "small_linear_cpu_only",
        "torch": torch.__version__, "torchao": importlib.metadata.version("torchao"),
        "forward_preserved": True, "serialization": "torch_state_dict_weights_only",
        "serialization_roundtrip_exact": True, "cuda_initialized": False,
        "logical_dtype": str(weight.dtype), "original_weight_bytes": source_bytes,
        "payloads": {name: {"dtype": str(item.dtype), "shape": list(item.shape),
                             "bytes": item.numel() * item.element_size()}
                     for name, item in payload.items()},
        "native_transport_validated": False, "diffusers_checkpoint_validated": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

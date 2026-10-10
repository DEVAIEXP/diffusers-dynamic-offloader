"""Small explicitly approved Windows FP8/native gate; not a model benchmark."""

import sys

if "--cpu-preflight" not in sys.argv:
    from ddo_vmm_bootstrap import ensure_vmm_fast_process

    ensure_vmm_fast_process(allocator_cache_mib=4)

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from diffusers import TorchAoConfig
from diffusers.quantizers.torchao.torchao_quantizer import TorchAoHfQuantizer
from safetensors.torch import load_file, save_file
from torchao.quantization import Float8WeightOnlyConfig, quantize_

from ddo_vmm_allocator.native import NativeVmmLibrary
from ddo_vmm_allocator.component_arena_adapter import attach_component_parameters, discover_component_layout
from ddo_vmm_allocator.quantized_arena_adapter import attach_float8_component_parameters, discover_float8_component_layout
from diffusers_dynamic_offloader.dynamic_offload import _safetensor_file_slices
from diffusers_dynamic_offloader.vmm_weight_store import SafetensorSliceRegistry

WIDTH = 512
LAYERS = 8
PAGE_BYTES = 2 * 1024**2
WEIGHT_CEILING = 6 * 1024**2
HOST_CACHE_BYTES = 6 * 1024**2
HOST_STAGING_BYTES = 4 * 1024**2


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(WIDTH, WIDTH, dtype=torch.bfloat16)
        self.interruption = None
        self.capture = False
        self.observed = None

    def forward(self, value):
        output = self.proj(value)
        if self.capture:
            # Diagnostic reads occur inside the managed unit, never after its
            # native views have been unpinned or restored to unfaulted addresses.
            self.observed = (self.proj.weight.qdata.view(torch.uint8).cpu(),
                             self.proj.weight.scale.cpu())
        if self.interruption is not None:
            raise self.interruption("intentional FP8 gate interruption")
        return output


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(LAYERS)])

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


def load_checkpoint(path, quantized=True):
    model = Model().eval().requires_grad_(False)
    state = load_file(str(path), device="cpu")
    if quantized:
        quantizer = TorchAoHfQuantizer(TorchAoConfig(Float8WeightOnlyConfig(set_inductor_config=False)))
        quantizer.set_metadata([str(path)])
        state = quantizer.maybe_update_state_dict(state)
    model.load_state_dict(state, assign=True)
    return model


def cpu_preflight(output_dir):
    """Validate actual serialized layout without loading a DLL or CUDA."""
    assert not torch.cuda.is_initialized()
    reports = []
    for quantized in (False, True):
        torch.manual_seed(43)
        source = Model().eval().requires_grad_(False)
        if quantized:
            quantize_(source, Float8WeightOnlyConfig(set_inductor_config=False))
        flat, metadata = TorchAoHfQuantizer.get_state_dict_and_metadata(
            None, source.state_dict(), safe_serialization=True)
        path = output_dir / ("fp8.safetensors" if quantized else "bf16.safetensors")
        save_file(flat, str(path), metadata=metadata)
        del source, flat, metadata
        model = load_checkpoint(path, quantized)
        available = _safetensor_file_slices([path])
        registry = SafetensorSliceRegistry()
        for index, block in enumerate(model.blocks):
            if quantized:
                for payload in ("qdata", "scale"):
                    assert registry.bind(getattr(block.proj.weight, payload),
                                         available[f"blocks.{index}.proj._weight_{payload}"])
            else:
                assert registry.bind(block.proj.weight, available[f"blocks.{index}.proj.weight"])
            assert registry.bind(block.proj.bias, available[f"blocks.{index}.proj.bias"])
        if quantized:
            layout, _ = discover_float8_component_layout(model, registry)
        else:
            layout = discover_component_layout(model, registry, buffer_limit_bytes=1024**2)
        observations = layout.capacity_snapshot()
        largest = observations["execution_slot_payload_bytes"]
        slot_bytes = ((largest + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
        assert slot_bytes == PAGE_BYTES
        assert 2 * slot_bytes < WEIGHT_CEILING
        prefix = WEIGHT_CEILING - 2 * slot_bytes
        streamed_units = [name for name, _, indices in layout.units if any(
            layout.parameters[index][1] + layout.parameters[index][0].nbytes > prefix
            for index in indices)]
        assert streamed_units and len(layout.units) == LAYERS
        assert sum(group.nbytes for group, _, _ in layout.parameters) <= HOST_CACHE_BYTES
        assert path.stat().st_size < HOST_CACHE_BYTES
        with torch.inference_mode():
            assert torch.isfinite(model(torch.randn(4, WIDTH, dtype=torch.bfloat16))).all()
        reports.append({"case": "fp8" if quantized else "bf16_control",
                        "layout": observations, "slot_bytes": slot_bytes,
                        "faultable_prefix_bytes": prefix, "streaming_units": streamed_units,
                        "serialized_file_bytes": path.stat().st_size})
    assert not torch.cuda.is_initialized()
    report = {"status": "passed", "scope": "cpu_geometry_only", "cuda_initialized": False,
              "weight_ceiling_bytes": WEIGHT_CEILING, "host_cache_bytes": HOST_CACHE_BYTES,
              "host_staging_bytes": HOST_STAGING_BYTES,
              "host_process_limit_bytes": HOST_CACHE_BYTES + HOST_STAGING_BYTES, "cases": reports}
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def run_case(output_dir, *, quantized):
    if os.name != "nt":
        raise RuntimeError("This selected gate is scoped to Windows")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    library_path = Path(os.environ["DDO_VMM_LIBRARY_PATH"]).resolve(strict=True)
    binary_hash = hashlib.sha256(library_path.read_bytes()).hexdigest().upper()
    expected_hash = "EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6"
    if binary_hash != expected_hash:
        raise RuntimeError("The DLL differs from the preserved Windows checkpoint")
    report = {"status": "running", "scope": "eight_small_layers_native_cuda",
              "case": "fp8" if quantized else "bf16_control",
              "dll_sha256": binary_hash, "torch": torch.__version__,
              "budget_bytes": WEIGHT_CEILING, "host_cache_bytes": HOST_CACHE_BYTES,
              "host_staging_bytes": HOST_STAGING_BYTES,
              "host_process_limit_bytes": HOST_CACHE_BYTES + HOST_STAGING_BYTES,
              "layers": LAYERS, "width": WIDTH,
              "allocator_cache_mib": 4, "cleanup_completed": False}
    adapter = store = None
    try:
        torch.manual_seed(43)
        source = Model().eval().requires_grad_(False)
        if quantized:
            quantize_(source, Float8WeightOnlyConfig(set_inductor_config=False))
        flat, metadata = TorchAoHfQuantizer.get_state_dict_and_metadata(
            None, source.state_dict(), safe_serialization=True)
        path = output_dir / "weights.safetensors"
        save_file(flat, str(path), metadata=metadata)
        del source, flat, metadata
        model = load_checkpoint(path, quantized)
        reference = load_checkpoint(path, quantized).cuda()
        value = torch.randn(4, WIDTH, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            target = reference(value).cpu()
        del reference
        available = _safetensor_file_slices([path])
        registry = SafetensorSliceRegistry()
        expected_payloads = []
        for index, block in enumerate(model.blocks):
            weight = block.proj.weight
            if quantized:
                for payload in ("qdata", "scale"):
                    name = f"blocks.{index}.proj._weight_{payload}"
                    assert registry.bind(getattr(weight, payload), available[name])
                expected_payloads.append((weight.qdata.view(torch.uint8).clone(), weight.scale.clone()))
            else:
                assert registry.bind(weight, available[f"blocks.{index}.proj.weight"])
            assert registry.bind(block.proj.bias, available[f"blocks.{index}.proj.bias"])
        del weight, block
        library = NativeVmmLibrary(library_path)
        attach = attach_float8_component_parameters if quantized else attach_component_parameters
        attach_report, store = attach(
            model, library, resident_budget_bytes=WEIGHT_CEILING, page_size_bytes=PAGE_BYTES,
            component_buffer_limit_bytes=1024**2,
            file_slices=registry, native_transport_slot_size_bytes=2 * 1024**2,
            native_transport_slots=2, native_source_cache_slot_bytes=2 * 1024**2,
            # The process ceiling includes both bounded native host owners.
            native_source_cache_process_limit_bytes=HOST_CACHE_BYTES + HOST_STAGING_BYTES,
            native_source_cache_slots=3, native_source_cache_registered_limit_bytes=HOST_CACHE_BYTES,
            native_source_cache_model_windows=True)
        adapter = attach_report.adapter
        reads = None
        for iteration in range(3):
            for block in model.blocks:
                block.capture = quantized and iteration == 0
            with torch.inference_mode():
                actual = model(value).cpu()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
            if iteration == 0:
                for block, expected in zip(model.blocks, expected_payloads):
                    assert torch.equal(block.observed[0], expected[0])
                    assert torch.equal(block.observed[1], expected[1])
                    block.observed = None
            adapter._collect_temporary_lifetimes(wait=True)
            stats = store.runtime_stats()
            report["native_stats"] = stats
            report["last_exact_output_iteration"] = iteration
            assert stats["cpu_source_bindings"] == 0
            assert stats["file_backed_bindings"] == LAYERS * (3 if quantized else 2)
            assert stats["fault_failures"] == 0 and stats["temporary_tensors"] == 0
            assert stats["total_weight_resident_bytes"] <= WEIGHT_CEILING
            assert stats["execution_buffer_uploads"] > 0
            assert stats["source_cache_admission_denials"] == 0
            # Open batches retain completed events until the next submission
            # or close. Zero ownership is a post-close contract, not a snapshot
            # contract while reusable batches remain open.
            assert stats["source_cache_transfer_owned_events"] <= LAYERS * (6 if quantized else 4)
            assert stats["source_cache_batch_completion_events"] <= LAYERS
            current_reads = stats["source_cache_source_read_bytes"]
            reads = current_reads if reads is None else reads
            assert current_reads == reads
        for error_type in (RuntimeError, KeyboardInterrupt):
            model.blocks[-1].interruption = error_type
            try:
                with torch.inference_mode():
                    model(value)
            except error_type as error:
                assert "intentional FP8 gate interruption" in str(error)
            else:
                raise AssertionError("The intentional interruption was not observed")
            model.blocks[-1].interruption = None
            with torch.inference_mode():
                torch.testing.assert_close(model(value).cpu(), target, rtol=0, atol=0)
        report["native_stats"] = store.runtime_stats()
        report["physical_payloads_verified"] = True if quantized else None
        report["repeat_output_exact"] = True
        report["interruption_recovery"] = ["RuntimeError", "KeyboardInterrupt"]
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if adapter is not None:
                adapter.close()
            if store is not None:
                transport = store._native_shard_transport
                for key, batch in tuple(transport._batches.items()):
                    batch.close()
                    del transport._batches[key]
                final_stats = store.runtime_stats()
                report["after_batch_close_stats"] = final_stats
                assert final_stats["source_cache_transfer_owned_events"] == 0
                assert final_stats["source_cache_batch_completion_events"] == 0
                assert final_stats["source_cache_live_tickets"] == 0
                store.close()
            if adapter is not None:
                assert all(block.proj.weight.device.type == "meta" for block in model.blocks)
                if quantized:
                    assert all(block.proj.weight.qdata.device.type == "meta" and
                               block.proj.weight.scale.device.type == "meta" for block in model.blocks)
                report["cleanup_completed"] = True
            assert hashlib.sha256(library_path.read_bytes()).hexdigest().upper() == binary_hash
        except BaseException as error:
            report["status"] = "failed"
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.cpu_preflight:
        cpu_preflight(output_dir)
        return
    # Stop at the first failed case. No model benchmark follows this gate.
    run_case(output_dir / "bf16_control", quantized=False)
    run_case(output_dir / "fp8", quantized=True)


if __name__ == "__main__":
    main()

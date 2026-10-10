"""Selected public DDO load/attach/cleanup gate, not a real-model benchmark."""

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
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers_dynamic_offloader import (
    DynamicOffloadConfig, enable_offload, from_pretrained_with_dynamic_offload, remove_dynamic_offload,
)
from convert_fp8_weight_only import convert_component
from probe_fp8_native_cuda import Block, LAYERS, WIDTH, WEIGHT_CEILING, PAGE_BYTES, HOST_STAGING_BYTES


class Component(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, root_state=False):
        super().__init__()
        self.root_state = root_state
        if root_state:
            self.root_gain = torch.nn.Parameter(torch.linspace(0.95, 1.05, 8 * WIDTH,
                dtype=torch.bfloat16).reshape(8, WIDTH), requires_grad=False)
            self.root_bias = torch.nn.Parameter(torch.randn(16, WIDTH,
                dtype=torch.bfloat16) * 0.001, requires_grad=False)
        self.blocks = torch.nn.ModuleList([Block() for _ in range(LAYERS)])

    def forward(self, value):
        if self.root_state:
            value = value * self.root_gain.mean(dim=0)
        for block in self.blocks:
            value = block(value)
        if self.root_state:
            value = value + self.root_bias.sum(dim=0)
        return value


def prepare(root, root_state=False):
    torch.manual_seed(43)
    source = root / "bf16_checkpoint"
    output = root / "fp8_checkpoint"
    model = Component().eval().requires_grad_(False)
    model.save_pretrained(source)
    del model
    conversion_source = source
    if root_state:
        conversion_source = root / "bf16_root_checkpoint"
        Component(root_state=True).eval().requires_grad_(False).save_pretrained(conversion_source)
    convert_component(conversion_source, output, model_class=Component)
    return source, output


def weight_ceiling(root_state):
    # Only this opt-in small fixture reserves an additional native page for
    # root state. Production budgets and the original gate remain unchanged.
    return WEIGHT_CEILING + PAGE_BYTES if root_state else WEIGHT_CEILING


def config_for(quantized, root_state=False):
    return DynamicOffloadConfig(
        execution_mode="vmm_fast_runtime", execution_device="cuda:0", offload_device="cpu",
        vmm_float8_weight_only=quantized, vmm_component_tensor_ownership=True,
        vmm_native_transport=True, vmm_module_pattern=".*", vmm_page_mb=PAGE_BYTES // 1024**2,
        vmm_resident_weight_budget_gb=weight_ceiling(root_state) / 1024**3,
        vmm_staging_buffer_gb=HOST_STAGING_BYTES / 1024**3,
        vmm_source_cache_gb=64 / 1024, vmm_source_registered_gb=64 / 1024,
        vmm_native_source_cache_slot_mb=0, vmm_native_source_cache_slots=0,
        cache_plan=False, verbose=False)


def load(path, quantized, root_state=False):
    return from_pretrained_with_dynamic_offload(
        path, model_loader=Component, dynamic_offload_config=config_for(quantized, root_state),
        torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True, apply_dynamic=False)


def preflight(root, paths, root_state=False):
    assert not torch.cuda.is_initialized()
    reports = []
    for quantized, path in zip((False, True), paths):
        case_root = root_state and quantized
        loaded = load(path, quantized, case_root)
        assert loaded.module.root_state is case_root
        report = loaded.inspect_vmm_component(component="public_gate").as_dict()
        capacity = report["setup"]["component_capacity"]
        assert capacity["execution_units"] == LAYERS
        assert capacity["execution_slot_payload_bytes"] < PAGE_BYTES
        assert capacity["payload_bytes"] > PAGE_BYTES
        assert capacity["unique_parameter_bytes"] < 64 * 1024**2
        if case_root:
            assert capacity["component_lifetime_parameter_groups"] == 2
            assert capacity["component_lifetime_parameter_bytes"] == 24 * 1024
        with torch.inference_mode():
            assert torch.isfinite(loaded.module(torch.randn(4, WIDTH, dtype=torch.bfloat16))).all()
        reports.append(report)
    assert not torch.cuda.is_initialized()
    result = {"status": "passed", "scope": "public_api_cpu_preflight_only", "cases": reports,
              "root_state_requested": root_state,
              "weight_ceiling_bytes": weight_ceiling(root_state), "host_cache_bytes": 64 * 1024**2,
              "host_staging_bytes": HOST_STAGING_BYTES, "host_process_ceiling_bytes": 68 * 1024**2,
              "cuda_initialized": False}
    (root / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def run_case(root, path, quantized, root_state=False, compile_fp8_linear=False):
    library_path = Path(os.environ["DDO_VMM_LIBRARY_PATH"]).resolve(strict=True)
    binary_hash = hashlib.sha256(library_path.read_bytes()).hexdigest().upper()
    assert binary_hash == "EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6"
    report = {"status": "running", "case": "fp8" if quantized else "bf16_control",
              "scope": "public_api_small_cuda_not_ltx", "dll_sha256": binary_hash,
              "root_state_requested": root_state,
              "weight_ceiling_bytes": weight_ceiling(root_state), "host_cache_bytes": 64 * 1024**2,
              "host_staging_bytes": HOST_STAGING_BYTES, "allocator_cache_mib": 4,
              "cleanup_completed": False,
              "compile_fp8_linear_requested": compile_fp8_linear,
              "compile_options": None,
              "compiled_linear_count": 0}
    model = result = None
    compile_handle = None
    try:
        loaded = load(path, quantized, root_state)
        model = loaded.module
        assert model.root_state is root_state
        reference = load(path, quantized, root_state).module.cuda()
        torch.manual_seed(43)
        value = torch.randn(4, WIDTH, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            expected = reference(value).cpu()
        del reference
        payloads = [(block.proj.weight.qdata.view(torch.uint8).clone(), block.proj.weight.scale.clone())
                    for block in model.blocks] if quantized else []
        root_payloads = {name: parameter.detach().clone() for name, parameter in
                        model.named_parameters(recurse=False)} if root_state else {}
        if compile_fp8_linear:
            if not quantized:
                raise ValueError("The compiled candidate must not alter the BF16 control")
            from diffusers_dynamic_offloader.float8_compile import configure_float8_linear_compile
            compile_report, compile_handle = configure_float8_linear_compile(model)
            report["compile_options"] = compile_report["options"]
            report["compiled_linear_count"] = compile_report["compiled_linear_count"]
            report["compiled_geometry_count"] = compile_report["compiled_geometry_count"]
            report["compile_cache_scope"] = compile_report["cache_scope"]
        result = enable_offload(model, component="public_gate", config=config_for(quantized, root_state),
                                route="dynamic_offload", use_environment=False)
        reads = None
        with result.cleanup_on_error():
            for iteration in range(3):
                for block in model.blocks:
                    block.capture = quantized and iteration == 0
                with torch.inference_mode():
                    actual = model(value).cpu()
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if iteration == 0 and quantized:
                    for block, original in zip(model.blocks, payloads):
                        assert torch.equal(block.observed[0], original[0])
                        assert torch.equal(block.observed[1], original[1])
                        block.observed = None
                snapshot = result.metrics_snapshot(collect_completed=True).as_dict()
                report["last_snapshot"] = snapshot
                stats = snapshot["runtime"]["vmm"]
                assert stats["fast_execution_buffer_uploads"] > int(root_state)
                assert stats["fast_total_weight_resident_bytes"] <= weight_ceiling(root_state)
                assert stats["fast_cpu_source_bindings"] == 0
                assert stats["fast_file_backed_bindings"] == LAYERS * (3 if quantized else 2) + len(root_payloads)
                if root_state:
                    ownership = snapshot["runtime"]["component_ownership"]
                    assert ownership["component_lifetime_parameter_bytes"] == 24 * 1024
                    assert ownership["component_lifetime_uploads"] == 1
                    assert ownership["component_lifetime_budget_charge_bytes"] == PAGE_BYTES
                    assert ownership["arena_weight_ceiling_bytes"] == WEIGHT_CEILING
                    assert ownership["declared_total_weight_ceiling_bytes"] == weight_ceiling(True)
                    assert stats["fast_component_lifetime_capacity_bytes"] == PAGE_BYTES
                    assert stats["fast_component_lifetime_resident_bytes"] == PAGE_BYTES
                    for name, expected_root in root_payloads.items():
                        torch.testing.assert_close(model._parameters[name].detach().cpu(), expected_root,
                                                   rtol=0, atol=0)
                for name in ("fault_failures", "temporary_tensors", "source_cache_admission_denials"):
                    assert stats["fast_" + name] == 0
                current = stats["fast_source_cache_source_read_bytes"]
                reads = current if reads is None else reads
                assert current == reads
            for error_type in (RuntimeError, KeyboardInterrupt):
                model.blocks[-1].interruption = error_type
                try:
                    with torch.inference_mode():
                        model(value)
                except error_type as error:
                    assert "intentional FP8 gate interruption" in str(error)
                else:
                    raise AssertionError("Intentional interruption was not raised")
                model.blocks[-1].interruption = None
                with torch.inference_mode():
                    torch.testing.assert_close(model(value).cpu(), expected, rtol=0, atol=0)
            report.update(status="passed", repeated_output_exact=True,
                          physical_payloads_verified=True if quantized else None,
                          interruption_recovery=["RuntimeError", "KeyboardInterrupt"])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if model is not None:
                remove_dynamic_offload(model)
                if compile_handle is not None:
                    compile_handle.remove()
                if result is not None:
                    assert all(block.proj.weight.device.type == "meta" for block in model.blocks)
                if quantized and result is not None:
                    assert all(block.proj.weight.qdata.device.type == "meta" and
                               block.proj.weight.scale.device.type == "meta" for block in model.blocks)
                if result is not None:
                    assert not hasattr(model, "_ddo_safetensor_slices")
                    if root_state:
                        assert all(parameter.device.type == "meta" for parameter in
                                   model.parameters(recurse=False))
                    report["cleanup_completed"] = True
                else:
                    report["attachment_completed"] = False
            assert hashlib.sha256(library_path.read_bytes()).hexdigest().upper() == binary_hash
        except BaseException as error:
            report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            (root / ("fp8_report.json" if quantized else "bf16_report.json")).write_text(
                json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "last_snapshot"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--compile-fp8-linear", action="store_true",
                        help="Opt-in small CUDA Inductor candidate; BF16 control stays eager")
    parser.add_argument("--root-state", action="store_true",
                        help="Keep the BF16 control unchanged; FP8 additionally exercises 24 KiB root state")
    parser.add_argument("--checkpoint-root", type=Path,
                        help="Previously prepared CPU fixture root; required for CUDA execution")
    args = parser.parse_args()
    if args.cpu_preflight and args.compile_fp8_linear:
        parser.error("The existing CPU preflight does not validate CUDA Inductor")
    if not args.cpu_preflight and args.checkpoint_root is None:
        parser.error("CUDA execution requires --checkpoint-root from a completed CPU preflight")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.cpu_preflight:
        paths = prepare(root, args.root_state)
        preflight(root, paths, args.root_state)
    else:
        checkpoint_root = args.checkpoint_root.resolve(strict=True)
        evidence = json.loads((checkpoint_root / "preflight.json").read_text(encoding="utf-8"))
        if evidence.get("status") != "passed" or evidence.get("cuda_initialized") is not False:
            raise ValueError("A passing CPU-only checkpoint preflight is required")
        if evidence.get("root_state_requested", False) is not args.root_state:
            raise ValueError("Prepared root-state fixture does not match the requested gate")
        paths = (checkpoint_root / "bf16_checkpoint", checkpoint_root / "fp8_checkpoint")
        for path in paths:
            if not path.is_dir():
                raise ValueError("Prepared checkpoint component is missing")
        for quantized, path in zip((False, True), paths):
            run_case(root, path, quantized, args.root_state and quantized,
                     args.compile_fp8_linear and quantized)


if __name__ == "__main__":
    main()

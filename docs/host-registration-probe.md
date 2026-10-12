# Host registration capacity diagnostic

`ddo-vmm-allocator/scripts/host_registration_probe.py` consolidates manual CUDA
host-registration controls into one explicitly approved, bounded sequence.
It does not tune a running model or change presets, cache ceilings, kernel
limits, or allocator configuration. No native rebuild is required.

Each attempt runs in a fresh child process with normal PyTorch allocation and
registers native host-ring windows. The parent does not import PyTorch or
initialize CUDA. The search uses the requested window size, never exceeds the
explicit total ceiling, and repeats the selected passing observation. Only a
host-registration CUDA OOM is treated as a capacity rejection. Other errors,
timeouts, or an unstable final observation stop the diagnostic. Each child has
a timeout; timeout cleanup kills and waits for that child.

Example from the WSL example-host directory, using the approved 1.5 GiB ceiling:

```bash
DDO_VMM_LIBRARY_PATH="$PWD/../diffusers-dynamic-offloader/ddo-vmm-allocator/build-wsl-idle-trim-20261012/libddo_vmm.so" \
.venv-wsl-gpu/bin/python -B \
  ../diffusers-dynamic-offloader/ddo-vmm-allocator/scripts/host_registration_probe.py \
  --max-mib 1536 --window-mib 64 --timeout-sec 120 \
  --output outputs/wsl_host_registration_probe.json
```

Before execution, approve the command, ceiling, hypothesis, and acceptance
criterion. Run without a competing model workload. The JSON records all
attempts, platform, interpreter, library hash, and selected passing capacity.
If the ceiling passes, the maximum remains unknown. Binary search assumes
monotonic admission and reports observations, not a guaranteed physical limit.

This is a registration/closure control, not a DMA, performance, memory-pressure,
or native-model-bootstrap gate. Available capacity can change with other
processes and driver state. Staging and source-cache registrations coexist and
consume the same host registration resources: do not assign the observed total
to each independently. Keep margin and separately approve a complete model
validation before adopting settings. Never increase runtime ceilings silently.

CPU-only search validation:

Native Linux review on 2026-10-12: six search contracts passed in 0.02 s, with
CUDA hidden. This is diagnostic algorithm acceptance, not a repeated CUDA
registration-capacity probe or an instruction to change model ceilings.

```bash
python -B -m pytest -q -p no:cacheprovider \
  ddo-vmm-allocator/tests/test_host_registration_probe_cpu.py
```

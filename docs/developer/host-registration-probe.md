# Host registration capacity diagnostic

`ddo-vmm-allocator/scripts/host_registration_probe.py` consolidates manual CUDA
host-registration controls into one bounded sequence.
It does not tune a running model or change presets, cache ceilings, kernel
limits, or allocator configuration. No native rebuild is required.

Each attempt runs in a fresh child process with normal PyTorch allocation and
registers native host-ring windows. The parent does not import PyTorch or
initialize CUDA. The search uses the requested window size, never exceeds the
explicit total ceiling, and repeats the selected passing observation. Only a
host-registration CUDA OOM is treated as a capacity rejection. Other errors,
timeouts, or an unstable final observation stop the diagnostic. Each child has
a timeout; timeout cleanup kills and waits for that child.

Example from the DDO checkout, with a CUDA-ready environment and a reviewed
platform library. The 1536 MiB ceiling is an explicit diagnostic limit, not a
recommended runtime capacity:

```bash
python -B ddo-vmm-allocator/scripts/host_registration_probe.py \
  --library /path/to/reviewed/libddo_vmm.so \
  --max-mib 1536 --window-mib 64 --timeout-sec 120 \
  --output host_registration_probe.json
```

Run without a competing model workload. The JSON records all
attempts, platform, interpreter, library hash, and selected passing capacity.
If the ceiling passes, the maximum remains unknown. Binary search assumes
monotonic admission and reports observations, not a guaranteed physical limit.

This is a registration/closure control, not a DMA, performance, memory-pressure,
or native-model-bootstrap gate. Available capacity can change with other
processes and driver state. Staging and source-cache registrations coexist and
consume the same host registration resources: do not assign the observed total
to each independently. Keep margin and separately perform complete model
validation before adopting settings. Never increase runtime ceilings silently.

CPU-only search validation:

```bash
python -B -m pytest -q -p no:cacheprovider \
  ddo-vmm-allocator/tests/test_host_registration_probe_cpu.py
```

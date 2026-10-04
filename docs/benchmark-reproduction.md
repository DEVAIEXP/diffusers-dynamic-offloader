# Native benchmark reproduction manifest

This manifest describes the selected staged runs in the
[public results](../experiments/dynamic_offload_results.md). It is not a
standalone benchmark shipped with the DDO wheel. The model-specific fixture
remains in the sibling `ltx2.3-image-diffusers` repository, revision `fbea833`;
script `run_dynamic_modular_vmm_distilled.py` and its custom blocks are required.
Do not move architecture-specific logic into DDO to reproduce this fixture.

## Fixed workload

Model: `elismasilva/ltx2.3-image-distilled-1.1`; Linux local snapshot
`5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9`. The Windows gate used its existing
local checkpoint; its exact snapshot ID was not recorded in the gate document.
Do not infer equal checkpoint hashes from the same model label alone.

BF16 encoder and transformer; real prompt; empty negative prompt; 1280×704;
seed43; eight steps; one generation; no PAG. Host defaults at the recorded
fixture revision: guidance scale1.0, guidance rescale0.7, VAE decode timestep0.0,
decode noise scaleNone. Prompt:

> Fisheye close-up of a calico cat wearing a tiny flower crown, sniffing the camera lens in a sunny park, with bright colors, realistic fur detail, and playful viral-pet energy.

Native encoder → readback/unload → connectors → native transformer → VAE/save.
Native active budget4GiB, total staging1.5GiB, source and registration limits
24.625GiB, allocator cache512MiB; encoder page2MiB, buffer ceiling1MiB.
Those values describe this test, not a recommendation to raise limits on another
machine. No automatic memory-capacity expansion or kernel-policy change.

## Environment and artifact identity

Use the DDO environment, not a separate host interpreter:

- Windows: `.venv-vmm`, optional `.[vmm-dev]` development profile.
- Linux: `.venv-linux`, optional `.[vmm-dev-linux]`; setup/build commands in
  [Linux environment](linux-environment.md).
- Profiles pin Torch2.12.1+cu132, torchvision0.27.1+cu132, transformers5.15.1,
  accelerate1.14.0, peft0.20.0, einops0.8.2 and custom Diffusers revision
  `e71ebc5858961e2a37d6a7b161d00ae0063c3ab7`. Linux pins safetensors0.8.0.
  Profile pins are not substitutes for a recorded installed dependency inventory.
- Linux observed: Ubuntu26.04.1 x86_64, Python3.12.14, RTX3060Ti8192MiB,
  driver595.91.07, GCC15.2.0, CMake4.2.3, Ninja1.13.2. Toolkit package13.2.1,
  nvcc13.2.78. Exact Windows driver/OS build were not recorded in this manifest.
- Linux closure: DDO `473b605`, allocator `433d3e4`. Documentation-only
  changes after this checkpoint do not change the validated binary.

| Platform | Selected library SHA256 |
| --- | --- |
| Windows candidate, not default DLL promotion | `755fcdd8dd1a72b371402750c071ca4b2146c0ac6bfe6b94c9b78f50f1ebb729` |
| Linux `build-linux/libddo_vmm.so` | `a87979f8773a010ddc2c781a37f3ef55112ca558a9d93083b972f51246a2b245` |

Linux build: span pool ON, dispatcher/test hooks/probes OFF. Build flags and
allocator source are part of reproduction; a suffix change is not portability.
Neither Python wheel embeds these binaries. Use the submodule checkout/build
and select the library explicitly. New builds do not inherit old acceptance
merely because their source revision matches.

## Model invocation

Run from the sibling host. Set `DDO_VMM_LIBRARY_PATH` to the selected platform
binary; `DDO_VMM_TORCH_CACHE_MIB=512`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Linux also used `HF_HUB_OFFLINE=1`, a populated local `HF_HOME` and host-local
`HF_MODULES_CACHE`. Replace paths only with equivalent local assets.

The shared model arguments were:

```text
run_dynamic_modular_vmm_distilled.py
  --ddo-public-native-runtime --ddo-backend vmm --ddo-preset auto
  --text-encoder-native-vmm
  --vmm-resident-weight-budget-gb 4 --vmm-staging-buffer-gb 0.5
  --ddo-vmm-source-cache-gb 24.625 --ddo-vmm-source-registered-gb 24.625
  --width 1280 --height 704 --seed 43 --steps 8 --generation-repeats 1
  --no-pag --local-files-only --metrics-level 2 --memory-monitor rss
  --ddo-gpu-timing --output-dir <new-output-directory>
```

Linux additionally supplied `--model-path <snapshot-directory>`, `--no-fake-prompt`
and `--save-metrics`. Windows real prompt was the fixture default. These are
host-fixture arguments: its `--vmm-staging-buffer-gb 0.5` becomes1.5GiB total.
The generic DDO `--ddo-vmm-staging-buffer-gb` means total capacity instead;
do not transpose the fixture value unchanged into another runner.

Fresh bootstrap installs the native allocator before Torch/Diffusers CUDA use.
Do not invoke this fixture inside a process that has already initialized CUDA.
This document is a reproduction recipe, not authorization for new GPU runs in
the development workflow; commands/limits and acceptance need user confirmation.

## Artifact lookup and evaluation

Selected sibling output directories:

- Windows: `outputs/ltx_image_modular_vmm/native_encoder_staged_gate_20261003/`.
- Linux: `outputs/ltx_image_modular_vmm/native_linux_staged_gate_20261004/`.

Stem: `ltx23_image_distilled_modular_vmm_bf16_vmm4gb_text_encoder_native_nopag_1280x704_steps8_seed43_ddo_public_cache24.625gb_registered24.625gb`.
PNG under that directory; JSON under `metrics/`.
Linux PNG SHA256 `dfbf135a833b5a77f4574a98cebd3be84d19fc569b79de8dd5f18457c52d64c1`;
JSON SHA256 `07c14628d7982f2319ec5f9b37ea9684c1d0c7a09f00910221e97d6fbdc707f4`.
Raw artifacts are local, ignored and not embedded in the Python package.

A self-contained public fixture/artifact bundle and matched official Diffusers
comparison remain publication work, not completed validations. Missing metadata,
different cache states and different timing scopes must remain visible in any
review. Do not republish historical data as a new matched comparison.

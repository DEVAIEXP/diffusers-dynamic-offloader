# Native delivery review (2026-10-10)

Development acceptance and public distribution are separate. This review adds
no CUDA execution, binary promotion, dependency installation or history rewrite.

## Preserved development state

- Generic saved FP8 adapter: allocator commit `3670f8d`; DDO integration
  `42ed5bf`; optional compute compilation is separate in `3dff1f1`.
- Documentation/evidence checkpoint: DDO `88edcf7`; host responsibility update
  `d227914`. The parent pins the allocator revision. Commits are local, not a
  claim that those revisions are available from the public remotes.
- Selected Windows DLL SHA256 remains
  `EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
  Latest review: 118 selected CPU contracts passed in 14.19 s; no rebuild.
- Selected BF16 Windows/Linux acceptance and newer Windows FP8 acceptance
  retain their distinct workload/binary boundaries. Linux FP8 remains unvalidated.
- Linear and NVMe defaults are preserved. LTX-specific compute changes stay
  in the host; generic FP8 transport does not require those changes.

## Concrete distribution work

1. **Installed-package bootstrap:** `ensure_vmm_fast_process()` currently
   requires `_default_allocator_root()` even when `ddo_vmm_allocator` is
   importable and an explicit library is supplied. A CPU-only review reproduced
   the refusal with package discovery true and checkout discovery disabled.
   Source-checkout integration works; standalone wheel integration is not ready.
   Support the installed package without importing CUDA-owning modules in the
   parent, preserve checkout precedence, and add bounded CPU relaunch contracts.
2. **Native artifact selection:** wheels do not bundle DLL/.so files. Choose
   source-build-only delivery or separately distributed binaries explicitly.
   Record source revision, build options, dependency environment and hash for
   each selected artifact. A default CMake build is not automatically the
   accepted joined-span build. Do not copy/promote or rebuild binaries silently.
3. **Clean installation:** run an isolated wheel/sdist install and bootstrap
   smoke check outside the source tree after item 1. Verify public module and
   license contents and optional dependency instructions. Native adapter and
   TorchAO are not mandatory ordinary DDO dependencies; TorchAO 0.18.0 and the
   tested Diffusers serialization stack are required for the selected FP8 recipe.
   Do not claim latest development gates from an untested clean environment.
4. **Publication review:** audit active deliverable files, engineering records,
   private history and remote reachability before publishing. Preserve truthful
   licensing, attribution and raw acceptance evidence. Any history rewrite,
   compatibility removal or publication requires separate explicit approval.
   Removing provenance references cannot establish license compatibility.

The allocator README also contains historical core-only/future-adapter wording;
its standalone installation/support overview needs reconciliation with current
native transport before release. Parent-relative evidence links assume a DDO
checkout and need a deliberate standalone-documentation destination.

This is a finite delivery checklist, not authorization for another performance
search or an open-ended model/platform test matrix. See
[native support](native-support-status.md) and [generic FP8](fp8-weight-only.md)
for actual accepted scope.

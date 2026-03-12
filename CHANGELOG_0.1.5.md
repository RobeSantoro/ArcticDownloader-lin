# Arctic ComfyUI Helper 0.1.5

## Fixes

- Fixed a PyTorch runtime warning during ComfyUI launch by updating the environment compatibility handling for allocator settings.
- Improved the ComfyUI runtime log text so launching without Sage/Flash/Nunchaku now reads as `PyTorch attention` instead of the more confusing `none`.

## ComfyUI Launch Controls

- Added a new `Flags` section in the ComfyUI area.
- Separated installed add-ons from launch-time flags so users can keep an add-on installed but temporarily launch ComfyUI without using it.
- Added launch flag controls for:
  - `--use-sage-attention`
  - `--use-flash-attention`
  - `--listen`
- Simplified the Sage UI so SageAttention and SageAttention3 share one visible Sage launch flag instead of showing the same flag twice.

## ComfyUI Listen Option

- Added support for starting ComfyUI with `--listen` directly from the app.
- The setting is saved and reused on future launches.

## Linux AMD ROCm Support

- Added a Linux install option for `Torch 2.9.1 + ROCm 6.4`.
- Added AMD GPU detection on Linux.
- If AMD is detected, the installer now auto-selects the ROCm torch profile.
- If AMD is detected, NVIDIA/CUDA torch profiles are disabled in the installer UI.
- If NVIDIA is detected, the ROCm torch profile is disabled in the installer UI.

## ROCm Guardrails

- Added compatibility checks so CUDA-focused add-ons are blocked when the ROCm profile is selected.
- This currently affects:
  - SageAttention
  - SageAttention3
  - FlashAttention
  - Nunchaku
  - Trellis2

## Notes

- The Linux AMD/ROCm flow was validated at compile level and integrated into the installer logic, but it still should be tested on a real AMD ROCm system.

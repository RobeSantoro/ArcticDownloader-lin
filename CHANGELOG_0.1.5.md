# Arctic ComfyUI Helper 0.1.5

## Highlights

- Added a new `Flags` section for ComfyUI launch options.
- Added support for starting ComfyUI with `--listen`.
- Improved launch controls so installed add-ons and launch-time flags are handled separately.
- Fixed the PyTorch allocator warning shown during some ComfyUI launches.

## Linux AMD / ROCm

- Added `Torch 2.9.1 + ROCm 6.4` as a Linux install option.
- Added AMD GPU detection with automatic ROCm profile selection.
- Added guided ROCm setup for:
  - Debian-based distros
  - Fedora-based distros
  - Arch-based distros
- Added ROCm readiness checks and clearer setup guidance in the UI.
- Guided ROCm setup now shows progress directly in the app logs.

## Installer Improvements

- Improved ROCm setup messaging, logging, and responsiveness.
- Reduced repeated sudo prompts during guided ROCm setup.
- Added better handling for group updates and post-install logout/login guidance.
- Hid ROCm setup controls automatically once the system is ready.

## Notes

- CUDA-only add-ons remain blocked when the ROCm profile is selected.
- AMD/ROCm support has been implemented across supported Linux distro families, but real-world testing on more AMD systems is still recommended.

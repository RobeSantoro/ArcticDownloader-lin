# Arctic ComfyUI Helper 0.1.7

## What's New

- Added Linux Intel GPU support for ComfyUI installs with a new `Torch 2.9.1 + XPU` option.
- Added guided Intel setup on Linux for Arch, Fedora, and Debian-based distributions.
- Added new optional ComfyUI startup flags:
  - `--lowvram`
  - `--bf16-unet`
  - `--async-offload`
  - `--disable-smart-memory`

## Fixes

- Fixed Linux app update detection so installed builds correctly see newer releases.
- Fixed the `Check Updates` button so it gives visible feedback while checking.
- Improved preflight clarity by showing which torch profile is being evaluated.
- Fixed Linux NVIDIA detection on mixed AMD CPU/iGPU systems so NVIDIA GPUs are no longer misidentified as AMD.

## Intel / XPU Notes

- Intel installs now support Triton XPU in the Linux XPU stack.


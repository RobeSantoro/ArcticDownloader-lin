# Arctic ComfyUI Helper 0.1.6

## Fixes

- Fixed Linux GPU detection priority so NVIDIA GPUs are no longer overridden by detected AMD integrated graphics.
- This fixes cases where systems with an AMD CPU/iGPU and an NVIDIA card could incorrectly appear as AMD in the app.
- NVIDIA systems should now keep the correct GPU detection and torch recommendation behavior.

# Arctic ComfyUI Helper 0.1.8

## Fixes

- Fixed a Linux Intel XPU readiness-check issue where the app could keep telling users to log out and back in even after a reboot.
- The Intel XPU runtime check now treats the `render` group as required and the `video` group as optional, which better matches real Linux Intel GPU setups.


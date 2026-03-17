# Arctic ComfyUI Helper 0.1.9

## Fixes

- Fixed a Linux Intel XPU readiness-check issue where the app could keep telling users to log out and back in even after a reboot.
- The Intel XPU runtime check now validates actual access to `/dev/dri/renderD*` instead of relying only on Linux group-name checks.
- This should stop false Intel XPU "logout/login or reboot" warnings on systems where render-node access is already available.


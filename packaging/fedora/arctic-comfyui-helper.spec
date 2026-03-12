Name:           arctic-comfyui-helper
Version:        0.1.5
Release:        1%{?dist}
Summary:        ComfyUI installer and model manager
%global debug_package %{nil}
%global _debugsource_packages 0

License:        Proprietary
URL:            https://github.com/ArcticLatent/ArcticDownloader-lin
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  cargo
BuildRequires:  rust
BuildRequires:  pkgconfig(gtk+-3.0)
BuildRequires:  pkgconfig(webkit2gtk-4.1)
BuildRequires:  pkgconfig(ayatana-appindicator3-0.1)
BuildRequires:  openssl-devel

Requires:       gtk3
Requires:       webkit2gtk4.1
Requires:       libappindicator-gtk3

%description
Native Linux desktop app for installing and managing ComfyUI,
models, and custom nodes.

%prep
%autosetup -n %{name}-%{version}

%build
# Reuse build artifacts across rpmbuild runs for faster iterative packaging.
export CARGO_TARGET_DIR="%{_topdir}/cargo-target"
# Allow reuse of incremental state for local packaging speed.
export CARGO_INCREMENTAL=1
# Prefer fast linker when available; fall back to bfd to avoid missing mold/lld issues.
if command -v mold >/dev/null 2>&1; then
  case " ${RUSTFLAGS:-} " in
    *" -C link-arg=-fuse-ld=mold "*) ;;
    *) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C link-arg=-fuse-ld=mold" ;;
  esac
elif command -v ld.bfd >/dev/null 2>&1; then
  case " ${RUSTFLAGS:-} " in
    *" -C link-arg=-fuse-ld=bfd "*) ;;
    *) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C link-arg=-fuse-ld=bfd" ;;
  esac
fi

cargo build --release --jobs "%{?_smp_build_ncpus}" --manifest-path src-tauri/Cargo.toml

%install
install -Dpm0755 "%{_topdir}/cargo-target/release/Arctic-ComfyUI-Helper" \
  %{buildroot}%{_bindir}/arctic-comfyui-helper
install -Dpm0644 packaging/linux/io.github.ArcticHelper.desktop \
  %{buildroot}%{_datadir}/applications/io.github.ArcticHelper.desktop
install -Dpm0644 src-tauri/dist/icon.svg \
  %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/io.github.ArcticHelper.svg

%files
%license README.public.md
%doc README.md
%{_bindir}/arctic-comfyui-helper
%{_datadir}/applications/io.github.ArcticHelper.desktop
%{_datadir}/icons/hicolor/scalable/apps/io.github.ArcticHelper.svg

%changelog
* Sat Feb 14 2026 Arctic Latent <contact@arcticlatent.com> - 0.1.0-1
- Initial Fedora package

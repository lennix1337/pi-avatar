fn main() {
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        embed_windows_resources();
    }
}

#[cfg(windows)]
fn embed_windows_resources() {
    use std::{env, path::PathBuf, process::Command};

    println!("cargo:rerun-if-changed=assets/pi-avatar.rc");
    println!("cargo:rerun-if-changed=assets/pi-avatar.ico");

    let resource = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("pi-avatar.res");
    let rc = find_msvc_tools::find_windows_sdk(env::consts::ARCH)
        .and_then(|sdk| sdk.path().map(|dir| dir.join("rc.exe")).find(|file| file.is_file()))
        .unwrap_or_else(|| "rc.exe".into());
    let components = [
        env!("CARGO_PKG_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH"),
        "0",
    ];
    let status = Command::new(rc)
        .args(["/nologo", "/D"])
        .arg(format!("APP_VERSION={}", components.join(",")))
        .arg("/D")
        .arg(format!("APP_VERSION_TEXT=\"{}\"", env!("CARGO_PKG_VERSION")))
        .arg("/fo")
        .arg(&resource)
        .arg("pi-avatar.rc")
        .current_dir("assets")
        .status()
        .expect("run rc.exe from the Windows SDK to compile Pi Avatar resources");
    assert!(status.success(), "rc.exe failed to compile Pi Avatar resources");
    println!("cargo:rustc-link-arg-bin=pi-avatar={}", resource.display());
}

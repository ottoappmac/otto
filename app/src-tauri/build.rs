fn main() {
    // Without this, changing files under `icons/` does not rerun the build script,
    // so `cargo tauri dev` keeps embedding the previous ICNS/PNG set (Dock shows the old icon).
    fn rerun_if_icons_changed(dir: &std::path::Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                rerun_if_icons_changed(&path);
            } else if path.is_file() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    rerun_if_icons_changed(std::path::Path::new("icons"));
    println!("cargo:rerun-if-changed=tauri.conf.json");

    tauri_build::build()
}

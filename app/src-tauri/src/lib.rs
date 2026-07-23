#[cfg(target_os = "macos")]
mod stealth;
mod tray;

use std::process::Child;
use std::sync::Mutex;
#[cfg(target_os = "macos")]
use tauri::Emitter;
use tauri::Manager;


struct BackendState(Mutex<Option<Child>>);

fn stop_backend(child: &mut Child) {
    #[cfg(unix)]
    {
        use std::time::{Duration, Instant};

        let pid = child.id() as i32;

        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                _ => {
                    eprintln!("[backend] graceful shutdown timed out, sending SIGKILL");
                    unsafe {
                        libc::kill(-pid, libc::SIGKILL);
                    }
                    let _ = child.wait();
                    return;
                }
            }
        }
    }

    #[cfg(not(unix))]
    {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn shutdown_backend(app: &tauri::AppHandle) {
    let state = app.state::<BackendState>();
    let mut guard = state.0.lock().unwrap();
    if let Some(mut child) = guard.take() {
        eprintln!("[backend] stopping backend");
        stop_backend(&mut child);
    }
}

#[tauri::command]
fn kill_backend(state: tauri::State<'_, BackendState>) -> Result<String, String> {
    let mut guard = state.0.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = guard.take() {
        eprintln!("[backend] kill_backend command invoked from frontend");
        stop_backend(&mut child);
        Ok("stopped".into())
    } else {
        Ok("not_running".into())
    }
}

/// Update the ambient suggestion count in the tray menu label.
/// Called from the frontend's `useAmbientHints` hook whenever the count changes.
#[tauri::command]
fn set_ambient_count(count: u32, app_handle: tauri::AppHandle) {
    tray::update_ambient_count(&app_handle, count);
}

/// Toggle Otto's anti-proctoring "stealth mode" on macOS.
///
/// When enabled this (a) excludes the main window from *all* screen-capture
/// paths — the legacy `NSWindow.sharingType` flag plus the private
/// `CGSSetWindowCaptureExcludeShape` window-server call that also defeats
/// ScreenCaptureKit / browser `getDisplayMedia` (what CoderPad proctoring uses)
/// — and (b) keeps the (still normal-sized) main window floating above other
/// windows and off the Dock/menu bar. It does *not* switch Otto into the
/// compact overlay panels — that's the separate `set_compact_mode` preference,
/// which is only offered once stealth is on. See `stealth.rs` for the caveats.
///
/// Must run on the main thread since it touches AppKit. No-op off macOS.
#[tauri::command]
fn set_hidden_from_capture(hidden: bool, app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let handle = app.clone();
        app.run_on_main_thread(move || apply_hidden_from_capture(&handle, hidden))
            .map_err(|e| e.to_string())?;

        // macOS needs a beat after a `.accessory` -> `.regular` policy change
        // before it will actually redraw the Dock icon / honor activation, so
        // re-assert foreground shortly after. (This is also why activation is
        // flaky under `tauri dev`: the terminal/cargo parent holds focus.)
        if !hidden {
            let handle = app.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(200));
                let inner = handle.clone();
                let _ = handle.run_on_main_thread(move || {
                    force_foreground(&inner);
                    // Re-apply the unread badge only after the final icon
                    // re-assert, so it isn't wiped by the icon reset.
                    let _ = inner.emit("reassert-badge", ());
                });
            });
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (hidden, app);
    }
    Ok(())
}

/// Promote Otto to a regular foreground app and bring its window forward so the
/// Dock icon and Cmd-Tab entry reappear. Must run on the main thread.
#[cfg(target_os = "macos")]
fn force_foreground(app: &tauri::AppHandle) {
    let Some(mtm) = objc2::MainThreadMarker::new() else {
        return;
    };
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
    let ns_app = NSApplication::sharedApplication(mtm);
    let _ = ns_app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    // After an accessory -> regular round-trip the Dock redraws the icon from
    // the process image, which for a bare dev executable falls back to the
    // generic "exec" icon. Re-assert the real icon explicitly.
    set_dock_icon(&ns_app);
    #[allow(deprecated)]
    ns_app.activateIgnoringOtherApps(true);
    if let Some(window) = app.get_webview_window("main") {
        // Return to normal stacking now that the app is a regular foreground app.
        let _ = window.set_always_on_top(false);
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Set the Dock icon to the bundled app icon (embedded at compile time).
#[cfg(target_os = "macos")]
fn set_dock_icon(ns_app: &objc2_app_kit::NSApplication) {
    use objc2::AllocAnyThread;
    use objc2_app_kit::NSImage;
    use objc2_foundation::NSData;

    static ICON_PNG: &[u8] = include_bytes!("../icons/128x128@2x.png");
    let data = NSData::with_bytes(ICON_PNG);
    if let Some(image) = NSImage::initWithData(NSImage::alloc(), &data) {
        unsafe { ns_app.setApplicationIconImage(Some(&image)) };
    }
}

#[cfg(target_os = "macos")]
fn apply_hidden_from_capture(app: &tauri::AppHandle, hidden: bool) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };

    // Exclude (or restore) the main window across every capture path: the
    // legacy `sharingType` flag and the private capture-exclude-shape that also
    // hides the window from ScreenCaptureKit / browser screen sharing.
    stealth::apply_capture_exclusion(app, "main", hidden);

    // Also hide Otto from the menu bar (tray) and Dock while hidden, so a
    // full-screen share doesn't reveal its icons. Restored when shown again.
    tray::set_tray_visible(app, !hidden);

    // Toggle the Dock icon via the app's activation policy. We drive this
    // directly rather than using Tauri's `set_dock_visibility`, which relies on
    // `TransformProcessType` — an asynchronous call that is unreliable at
    // bringing the Dock icon back.
    if hidden {
        if let Some(mtm) = objc2::MainThreadMarker::new() {
            use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
            let ns_app = NSApplication::sharedApplication(mtm);
            let _ = ns_app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
        }
        // Stealth alone keeps the normal-sized main window around — just
        // floating above other windows and off the Dock/menu bar — rather than
        // switching to the compact overlay panels. Compact is a separate,
        // stealth-gated preference applied via `set_compact_mode`.
        let _ = window.set_always_on_top(true);
    } else {
        // Stealth off implies compact off too: dismiss the overlay panels
        // regardless of their state and restore the normal main window. The
        // caller also re-asserts foreground after a short delay since macOS may
        // not redraw the Dock icon on the first attempt.
        stealth::hide_overlay(app);
        let _ = window.set_always_on_top(false);
        force_foreground(app);
    }
}

/// Toggle Otto's "compact" overlay UI on macOS — swaps the normal main window
/// for two small, transparent, non-activating panels (Chat + Live Capture).
/// Only meaningful while stealth mode is on (the frontend gates the setting on
/// that), but this is otherwise independent of `set_hidden_from_capture` so
/// stealth can run with the normal window when compact is off.
///
/// Must run on the main thread since it touches AppKit. No-op off macOS.
#[tauri::command]
fn set_compact_mode(compact: bool, app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let handle = app.clone();
        app.run_on_main_thread(move || apply_compact_mode(&handle, compact))
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (compact, app);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn apply_compact_mode(app: &tauri::AppHandle, compact: bool) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };

    if compact {
        // Bring up the overlay panels first and only hide the normal (decorated,
        // focus-stealing) main window if they actually showed — otherwise a
        // panel failure would leave nothing visible at all.
        match stealth::show_overlay(app, true) {
            Ok(()) => {
                let _ = window.hide();
            }
            Err(e) => {
                eprintln!("[stealth] compact overlay failed to show, keeping main window: {e}");
            }
        }
    } else {
        stealth::hide_overlay(app);
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Set the Dock tile badge label and force a redraw. Returning to the Dock
/// (accessory -> regular) creates a fresh tile, so the badge must be re-applied
/// with an explicit `display()` for it to reappear.
#[tauri::command]
fn set_dock_badge(label: Option<String>, app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        app.run_on_main_thread(move || {
            let Some(mtm) = objc2::MainThreadMarker::new() else {
                return;
            };
            use objc2_app_kit::NSApplication;
            use objc2_foundation::NSString;
            let ns_app = NSApplication::sharedApplication(mtm);
            let dock_tile = ns_app.dockTile();
            let ns_label = label.as_ref().map(|l| NSString::from_str(l));
            dock_tile.setBadgeLabel(ns_label.as_deref());
            dock_tile.display();
        })
        .map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (label, app);
    }
    Ok(())
}

/// Show the non-activating stealth overlay panel. `make_key` requests keyboard
/// focus so the overlay input can be typed into. No-op off macOS.
#[tauri::command]
fn show_stealth_overlay(make_key: Option<bool>, app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let handle = app.clone();
        let make_key = make_key.unwrap_or(false);
        app.run_on_main_thread(move || {
            let _ = stealth::show_overlay(&handle, make_key);
        })
        .map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (make_key, app);
    }
    Ok(())
}

/// Hide all stealth panels. No-op off macOS.
#[tauri::command]
fn hide_stealth_overlay(app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let handle = app.clone();
        app.run_on_main_thread(move || stealth::hide_overlay(&handle))
            .map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
    }
    Ok(())
}

/// Bring a single stealth panel to the front (creating it if needed). Used by
/// the "Show Chat" / "Show Live Capture" buttons so each panel can summon the
/// other. `panel` is the window label ("overlay" for chat, "capture" for Live
/// Capture); `make_key` grabs keyboard focus (for the chat composer). No-op off
/// macOS.
#[tauri::command]
fn focus_stealth_panel(
    panel: String,
    make_key: Option<bool>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let handle = app.clone();
        let make_key = make_key.unwrap_or(false);
        app.run_on_main_thread(move || {
            let _ = stealth::focus_panel(&handle, &panel, make_key);
        })
        .map_err(|e| e.to_string())?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (panel, make_key, app);
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[allow(unused_mut)]
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_notification::init());

    // Global hotkey (Cmd+Shift+\) to toggle the stealth overlay from any app.
    #[cfg(target_os = "macos")]
    {
        use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut, ShortcutState};
        let toggle_shortcut = Shortcut::new(
            Some(Modifiers::SUPER | Modifiers::SHIFT),
            Code::Backslash,
        );
        builder = builder
            .plugin(tauri_nspanel::init())
            .plugin(
                tauri_plugin_global_shortcut::Builder::new()
                    .with_handler(move |app, shortcut, event| {
                        if shortcut == &toggle_shortcut
                            && event.state() == ShortcutState::Pressed
                        {
                            let handle = app.clone();
                            let _ = app.run_on_main_thread(move || {
                                stealth::toggle_overlay(&handle);
                            });
                        }
                    })
                    .build(),
            );
    }

    builder
        .manage(BackendState(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            kill_backend,
            set_ambient_count,
            set_hidden_from_capture,
            set_compact_mode,
            set_dock_badge,
            show_stealth_overlay,
            hide_stealth_overlay,
            focus_stealth_panel
        ])
        .setup(|app| {
            tray::create_tray(app)?;

            // Register the overlay toggle hotkey now that the app is running.
            #[cfg(target_os = "macos")]
            {
                use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};
                let toggle_shortcut = Shortcut::new(
                    Some(Modifiers::SUPER | Modifiers::SHIFT),
                    Code::Backslash,
                );
                let _ = app.global_shortcut().register(toggle_shortcut);

                // Guard against ⌘M losing stealth panels into the Dock.
                let handle = app.handle().clone();
                let _ = handle.run_on_main_thread(stealth::install_miniaturize_guard);
            }

            #[cfg(not(debug_assertions))]
            {
                spawn_backend(app)?;
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            match event {
                tauri::RunEvent::WindowEvent {
                    label,
                    event: tauri::WindowEvent::CloseRequested { api, .. },
                    ..
                } => {
                    api.prevent_close();
                    if let Some(window) = app_handle.get_webview_window(&label) {
                        let _ = window.hide();
                    }
                }
                tauri::RunEvent::Exit => {
                    shutdown_backend(app_handle);
                }
                _ => {}
            }
        });
}

#[cfg(not(debug_assertions))]
fn spawn_backend(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let resource_dir = app
        .path()
        .resource_dir()
        .expect("failed to resolve resource directory");

    let backend_exe = resource_dir.join("backend").join(backend_exe_name());

    // Don't hard-crash if the bundled backend is missing (e.g. a dev/test
    // build that skipped the PyInstaller step). Log and continue — the app can
    // still connect to a backend already listening on the expected port.
    if !backend_exe.exists() {
        eprintln!("[backend] no bundled backend at {backend_exe:?}; skipping spawn");
        return Ok(());
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&backend_exe) {
            let mut perms = meta.permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(&backend_exe, perms);
        }
    }

    eprintln!("[backend] launching {:?}", backend_exe);

    #[allow(unused_mut)]
    let mut cmd = Command::new(&backend_exe);
    cmd.args(["--port", "18081"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setpgid(0, 0);
                Ok(())
            });
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            eprintln!("[backend] failed to spawn backend ({backend_exe:?}): {e}");
            return Ok(());
        }
    };

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            eprintln!("[backend stdout] {}", line);
        }
    });

    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("[backend stderr] {}", line);
        }
    });

    let state = app.state::<BackendState>();
    *state.0.lock().unwrap() = Some(child);

    Ok(())
}

#[cfg(not(debug_assertions))]
fn backend_exe_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "otto-backend.exe"
    } else {
        "otto-backend"
    }
}

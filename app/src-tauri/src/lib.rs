mod tray;

use std::process::Child;
use std::sync::Mutex;
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_notification::init())
        .manage(BackendState(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![kill_backend, set_ambient_count])
        .setup(|app| {
            tray::create_tray(app)?;

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

    let mut child = cmd.spawn().unwrap_or_else(|e| {
        panic!("failed to spawn backend: {} (path: {:?})", e, backend_exe)
    });

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

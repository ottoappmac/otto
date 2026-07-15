use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Emitter, Manager, Theme,
};

use crate::shutdown_backend;

const TRAY_ICON_LIGHT: &[u8] = include_bytes!("../icons/tray-light.png");
const TRAY_ICON_DARK: &[u8] = include_bytes!("../icons/tray-dark.png");
const TRAY_ID: &str = "main";
const AMBIENT_MENU_ID: &str = "ambient";

fn tray_icon_for(theme: Theme) -> Image<'static> {
    let bytes = match theme {
        Theme::Light => TRAY_ICON_LIGHT,
        _ => TRAY_ICON_DARK,
    };
    Image::from_bytes(bytes).expect("failed to decode tray icon")
}

fn ambient_label(count: u32) -> String {
    if count == 0 {
        "Suggestions".into()
    } else {
        format!("Suggestions ({})", count)
    }
}

pub fn create_tray(app: &mut tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let show = MenuItem::with_id(app, "show", "Show Otto", true, None::<&str>)?;
    let ambient = MenuItem::with_id(app, AMBIENT_MENU_ID, &ambient_label(0), true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &ambient, &quit])?;

    let initial_theme = app
        .get_webview_window("main")
        .and_then(|w| w.theme().ok())
        .unwrap_or(Theme::Dark);

    TrayIconBuilder::with_id(TRAY_ID)
        .icon(tray_icon_for(initial_theme))
        .icon_as_template(true)
        .tooltip("Otto")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_tray_icon_event(|tray, event| {
            if let tauri::tray::TrayIconEvent::DoubleClick { .. } = event {
                show_main_window(tray.app_handle());
            }
        })
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => show_main_window(app),
            AMBIENT_MENU_ID => {
                show_main_window(app);
                // Emit an event the webview listens for to navigate to /ambient.
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.emit("navigate", "/ambient");
                }
            }
            "quit" => {
                shutdown_backend(app);
                app.exit(0);
            }
            _ => {}
        })
        .build(app)?;

    if let Some(window) = app.get_webview_window("main") {
        let app_handle = app.handle().clone();
        window.on_window_event(move |event| {
            if let tauri::WindowEvent::ThemeChanged(theme) = event {
                if let Some(tray) = app_handle.tray_by_id(TRAY_ID) {
                    let _ = tray.set_icon(Some(tray_icon_for(*theme)));
                }
            }
        });
    }

    Ok(())
}

/// Called from the frontend via `invoke("set_ambient_count", { count })`.
/// Rebuilds the tray menu with an updated "Suggestions (N)" label.
/// Tauri 2's `TrayIcon` has `set_menu` but no `menu()` getter, so we
/// reconstruct the full menu and swap it in — the `on_menu_event` handler
/// registered on the TrayIcon itself is unaffected by menu replacement.
pub fn update_ambient_count(app: &tauri::AppHandle, count: u32) {
    let Some(tray) = app.tray_by_id(TRAY_ID) else { return };
    let Ok(show)    = MenuItem::with_id(app, "show",            "Show Otto",              true, None::<&str>) else { return };
    let Ok(ambient) = MenuItem::with_id(app, AMBIENT_MENU_ID,  &ambient_label(count),    true, None::<&str>) else { return };
    let Ok(quit)    = MenuItem::with_id(app, "quit",            "Quit",                   true, None::<&str>) else { return };
    let Ok(menu)    = Menu::with_items(app, &[&show, &ambient, &quit])                                       else { return };
    let _ = tray.set_menu(Some(menu));
}

/// Show or hide the menu-bar (status item) tray icon. Used by the
/// "hide from screen share" toggle so Otto can vanish from the menu bar too.
pub fn set_tray_visible(app: &tauri::AppHandle, visible: bool) {
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        let _ = tray.set_visible(visible);
    }
}

fn show_main_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

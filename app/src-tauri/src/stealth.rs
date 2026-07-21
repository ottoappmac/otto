//! Anti-proctoring "stealth" support (macOS only).
//!
//! Two capabilities live here:
//!
//! 1. **True capture exclusion.** `NSWindow.sharingType = .none` only hides a
//!    window from the legacy CoreGraphics capture path; on macOS 15+ browser
//!    `getDisplayMedia` and other ScreenCaptureKit consumers still see it. The
//!    private `CGSSetWindowCaptureExcludeShape` window-server call punches a hole
//!    in the captured framebuffer for the window, which does defeat conformant
//!    ScreenCaptureKit capturers (this is what CoderPad's in-browser proctoring
//!    uses). We apply both so every capture path is covered.
//!
//! 2. **Non-activating overlay panel.** A borderless, transparent
//!    `NSPanel` (swizzled from a Tauri window via `tauri-nspanel`) with the
//!    non-activating style mask. Because it never activates Otto, clicking or
//!    typing in it does not deactivate the frontmost app, so the CoderPad
//!    browser tab never fires the `blur`/`visibilitychange` "focus lost" event.
//!
//! Caveat: `CGSSetWindowCaptureExcludeShape` is a private, undocumented API. It
//! is App Store-ineligible and may change in future macOS releases. Apple's own
//! QuickTime / blessed conferencing partners can still capture the window; the
//! browser capture path CoderPad relies on cannot.

use block2::RcBlock;
use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use objc2_foundation::NSString;
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_nspanel::{ManagerExt, WebviewWindowExt};

/// Chat panel — the compact composer + expandable message history. Keeps the
/// historical "overlay" label so existing preferences/capabilities still match.
pub const CHAT_LABEL: &str = "overlay";
/// Live Capture panel — audio transcription + screenshots, its own movable
/// window so the two surfaces no longer share (and squash) one panel.
pub const CAPTURE_LABEL: &str = "capture";

/// All stealth panels, in the order we create/show them.
const STEALTH_LABELS: [&str; 2] = [CHAT_LABEL, CAPTURE_LABEL];

/// `NSWindowStyleMaskNonactivatingPanel` — the panel can accept key events
/// without activating (and thus without deactivating the frontmost app).
const NS_NONACTIVATING_PANEL: i32 = 1 << 7;
/// `NSWindowStyleMaskResizable` — keep edge-drag resizing on the borderless
/// panel so the user can expand it to see more output.
const NS_RESIZABLE: i32 = 1 << 3;

/// Gap below the system menu bar / from screen edges when first placed.
const EDGE_MARGIN: f64 = 16.0;
const TOP_INSET: f64 = 28.0;

/// Default logical sizes for each panel when first shown. Both stay resizable
/// and movable, so these are only the initial frame. Both panels start
/// *compact* — the chat panel shows the input composer with a peek of the last
/// message; the capture panel shows the record controls with the last
/// transcribed line. The user expands either (via the title-bar control) to the
/// fuller view. Keep the compact heights in sync with `EXPAND_CONFIG` in
/// `StealthTitlebar.tsx`.
const CHAT_W: f64 = 600.0;
const CHAT_H: f64 = 300.0;
const CAPTURE_W: f64 = 440.0;
const CAPTURE_H: f64 = 320.0;

/// Collection behavior so the overlay follows the user across Spaces and stays
/// visible over another app's full-screen window (CoderPad forces the browser
/// full-screen): canJoinAllSpaces (1) | stationary (1<<4) | fullScreenAuxiliary (1<<8).
const NS_COLLECTION_BEHAVIOR: usize = (1 << 0) | (1 << 4) | (1 << 8);

/// `NSFloatingWindowLevel` — hover above ordinary windows.
const NS_FLOATING_WINDOW_LEVEL: isize = 3;

// --- Private CoreGraphics (SkyLight) window-server API -----------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct CgPoint {
    x: f64,
    y: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CgSize {
    width: f64,
    height: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CgRect {
    origin: CgPoint,
    size: CgSize,
}

type CgsConnectionId = u32;
type CgsWindowId = u32;
#[repr(C)]
struct CgsRegionObject {
    _private: [u8; 0],
}
type CgsRegionRef = *mut CgsRegionObject;

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGSMainConnectionID() -> CgsConnectionId;
    fn CGSNewRegionWithRect(rect: *const CgRect, out_region: *mut CgsRegionRef) -> i32;
    fn CGSSetWindowCaptureExcludeShape(
        cid: CgsConnectionId,
        wid: CgsWindowId,
        region: CgsRegionRef,
    ) -> i32;
    fn CGSReleaseRegion(region: CgsRegionRef) -> i32;
}

/// Exclude (or restore) a specific NSWindow from the captured framebuffer.
///
/// Uses an oversized region so the whole window is excluded regardless of its
/// current size — no need to re-apply on resize. Passing a null region clears
/// the exclusion. Must run on the main thread.
///
/// SAFETY: `ns_window` must be a valid `NSWindow*` and we must be on the main
/// thread (callers guarantee this via `run_on_main_thread`).
unsafe fn set_window_capture_excluded(ns_window: *mut AnyObject, excluded: bool) {
    if ns_window.is_null() {
        return;
    }
    let window_number: isize = msg_send![&*ns_window, windowNumber];
    if window_number <= 0 {
        return;
    }
    let wid = window_number as CgsWindowId;
    let cid = CGSMainConnectionID();

    if excluded {
        // Window-local coordinates; oversized so any window size is covered.
        let rect = CgRect {
            origin: CgPoint { x: 0.0, y: 0.0 },
            size: CgSize {
                width: 1_000_000.0,
                height: 1_000_000.0,
            },
        };
        let mut region: CgsRegionRef = std::ptr::null_mut();
        if CGSNewRegionWithRect(&rect, &mut region) == 0 && !region.is_null() {
            let _ = CGSSetWindowCaptureExcludeShape(cid, wid, region);
            CGSReleaseRegion(region);
        }
    } else {
        let _ = CGSSetWindowCaptureExcludeShape(cid, wid, std::ptr::null_mut());
    }
}

/// Install a process-wide guard that instantly un-miniaturizes any stealth
/// panel. A borderless `NSPanel` has no minimize button, and we clear the
/// miniaturizable style bit — but the app's **Window ▸ Minimize** menu item
/// (⌘M) sends `miniaturize:`, which ignores `isMiniaturizable` and would
/// otherwise shrink the panel into the Dock's minimized tray. With the Dock
/// icon hidden in stealth mode that panel is effectively lost. Rather than
/// rebuild the whole app menu just to drop ⌘M (which would also affect the
/// normal main window), we listen for `NSWindowDidMiniaturizeNotification` and
/// immediately restore any window carrying the non-activating-panel style mask
/// (i.e. our stealth panels — never the main window).
///
/// Registers exactly once; must run on the main thread.
pub fn install_miniaturize_guard() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }

    // Fires *after* the miniaturize completes; deminiaturize reverses it. The
    // window is the notification's `object`.
    let block = RcBlock::new(|note: *mut AnyObject| {
        if note.is_null() {
            return;
        }
        unsafe {
            let win: *mut AnyObject = msg_send![&*note, object];
            if win.is_null() {
                return;
            }
            let mask: usize = msg_send![&*win, styleMask];
            if mask & (NS_NONACTIVATING_PANEL as usize) != 0 {
                let _: () = msg_send![&*win, deminiaturize: std::ptr::null_mut::<AnyObject>()];
            }
        }
    });

    unsafe {
        let center: *mut AnyObject = msg_send![class!(NSNotificationCenter), defaultCenter];
        let name = NSString::from_str("NSWindowDidMiniaturizeNotification");
        let _: *mut AnyObject = msg_send![
            center,
            addObserverForName: &*name,
            object: std::ptr::null_mut::<AnyObject>(),
            queue: std::ptr::null_mut::<AnyObject>(),
            usingBlock: &*block,
        ];
    }
    // The observer retains a copy of the block; leak our handle so the closure
    // stays valid for the whole app lifetime.
    std::mem::forget(block);
}

/// Apply both capture-exclusion mechanisms (legacy `sharingType` + the private
/// exclude-shape) to the window with `label`. No-op if the window is absent.
/// Must be called on the main thread.
pub fn apply_capture_exclusion(app: &AppHandle, label: &str, excluded: bool) {
    use objc2_app_kit::NSWindowSharingType;

    let Some(window) = app.get_webview_window(label) else {
        return;
    };
    let ns_window = match window.ns_window() {
        Ok(ptr) if !ptr.is_null() => ptr as *mut AnyObject,
        _ => return,
    };
    // NSWindowSharingNone (0) covers the legacy CoreGraphics path; 2 =
    // NSWindowSharingReadWrite (the default). The named constants are
    // deprecated in this SDK, so the default is passed as a raw value.
    let sharing_type = if excluded {
        NSWindowSharingType::None
    } else {
        NSWindowSharingType(2)
    };
    unsafe {
        let _: () = msg_send![&*ns_window, setSharingType: sharing_type];
        set_window_capture_excluded(ns_window, excluded);
    }
}

/// Ensure both stealth panels (chat + Live Capture) exist, are capture-excluded,
/// and are visible. Creates them lazily on first use and lays them out
/// side-by-side. When `make_key` is true the chat panel takes key focus (so its
/// composer can be typed into) — because these are non-activating panels this
/// still does not activate Otto or deactivate the frontmost app.
pub fn show_overlay(app: &AppHandle, make_key: bool) -> Result<(), String> {
    let mut created_any = false;
    for label in STEALTH_LABELS {
        if app.get_webview_panel(label).is_err() {
            create_stealth_panel(app, label)?;
            created_any = true;
        }
    }
    // Only auto-arrange the very first time we create the panels; afterwards we
    // respect wherever the user has dragged them.
    if created_any {
        layout_panels(app);
    }

    for label in STEALTH_LABELS {
        apply_capture_exclusion(app, label, true);
        // Recover a panel that was miniaturized (e.g. via ⌘M before we disabled
        // it) so the hotkey always brings the panels back.
        if let Some(win) = app.get_webview_window(label) {
            let _ = win.unminimize();
        }
    }

    // Live Capture comes forward without stealing key focus…
    if let Ok(panel) = app.get_webview_panel(CAPTURE_LABEL) {
        panel.order_front_regardless();
    }
    // …then the chat panel, which optionally takes key focus for typing.
    if let Ok(panel) = app.get_webview_panel(CHAT_LABEL) {
        if make_key {
            panel.show();
        } else {
            panel.order_front_regardless();
        }
    }
    Ok(())
}

/// Bring a single stealth panel to the front (creating it if needed). Used by
/// the "Show Chat" / "Show Live Capture" buttons so each panel can summon the
/// other. `make_key` grabs keyboard focus (for the chat composer).
pub fn focus_panel(app: &AppHandle, label: &str, make_key: bool) -> Result<(), String> {
    if app.get_webview_panel(label).is_err() {
        create_stealth_panel(app, label)?;
        layout_panels(app);
    }
    apply_capture_exclusion(app, label, true);
    if let Some(win) = app.get_webview_window(label) {
        let _ = win.unminimize();
    }
    if let Ok(panel) = app.get_webview_panel(label) {
        if make_key {
            panel.show();
        } else {
            panel.order_front_regardless();
        }
    }
    Ok(())
}

/// Place the two panels side-by-side near the top of the primary screen: chat
/// on the left, Live Capture on the right. Called once when the panels are
/// first created; both stay resizable/movable so the user can rearrange them.
fn layout_panels(app: &AppHandle) {
    let logical_screen_width = app
        .primary_monitor()
        .ok()
        .flatten()
        .map(|m| m.size().width as f64 / m.scale_factor())
        .unwrap_or(1440.0);

    let capture_x = (logical_screen_width - CAPTURE_W - EDGE_MARGIN).max(EDGE_MARGIN);
    // Chat fills the space to the left of the capture panel, within sane bounds.
    let chat_w = (capture_x - EDGE_MARGIN * 2.0).clamp(480.0, CHAT_W);

    place_panel(app, CHAT_LABEL, EDGE_MARGIN, TOP_INSET, chat_w, CHAT_H);
    place_panel(app, CAPTURE_LABEL, capture_x, TOP_INSET, CAPTURE_W, CAPTURE_H);
}

/// Set a panel's frame from logical coordinates relative to the primary screen.
fn place_panel(app: &AppHandle, label: &str, x: f64, y: f64, w: f64, h: f64) {
    use tauri::{PhysicalPosition, PhysicalSize};

    let Some(win) = app.get_webview_window(label) else {
        return;
    };
    let monitor = win
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| app.primary_monitor().ok().flatten());
    let Some(monitor) = monitor else {
        return;
    };
    let scale = monitor.scale_factor();
    let m_pos = monitor.position();

    let _ = win.set_size(PhysicalSize::new(
        (w * scale) as u32,
        (h * scale) as u32,
    ));
    let _ = win.set_position(PhysicalPosition::new(
        m_pos.x + (x * scale) as i32,
        m_pos.y + (y * scale) as i32,
    ));
}

/// Hide all stealth panels (kept alive for fast re-show). No-op if absent.
pub fn hide_overlay(app: &AppHandle) {
    for label in STEALTH_LABELS {
        if let Ok(panel) = app.get_webview_panel(label) {
            panel.order_out(None);
        }
    }
}

/// Toggle stealth-panel visibility as a group; used by the global hotkey. If the
/// chat panel is visible we hide everything, otherwise we show everything.
pub fn toggle_overlay(app: &AppHandle) {
    let chat_visible = matches!(
        app.get_webview_panel(CHAT_LABEL),
        Ok(panel) if panel.is_visible()
    );
    if chat_visible {
        hide_overlay(app);
    } else {
        let _ = show_overlay(app, true);
    }
}

/// Build a stealth Tauri window and swizzle it into a non-activating NSPanel.
///
/// The window must be created *without* decorations — converting a titled
/// window to a non-activating panel crashes AppKit. Both the chat and capture
/// panels load the same app root; the frontend routes on the window label.
fn create_stealth_panel(app: &AppHandle, label: &str) -> Result<(), String> {
    // Load the app root with no hash/query — a `#`/`?` in an `App` URL is
    // treated as part of the file path and percent-encoded, which 404s to a
    // blank (transparent) panel. The frontend detects the panel (and which one)
    // by its window label instead.
    let (w, h) = if label == CAPTURE_LABEL {
        (CAPTURE_W, CAPTURE_H)
    } else {
        (CHAT_W, CHAT_H)
    };

    let window = WebviewWindowBuilder::new(app, label, WebviewUrl::App("index.html".into()))
        .title("")
        // Fallback frame only — `layout_panels` positions it right after.
        .inner_size(w, h)
        .min_inner_size(320.0, 160.0)
        .decorations(false)
        .transparent(true)
        .resizable(true)
        // Not minimizable: a borderless panel has no title bar, but ⌘M would
        // still miniaturize it — and with the Dock icon hidden in stealth the
        // window would be effectively lost. Keep it always reachable instead.
        .minimizable(false)
        .skip_taskbar(true)
        .always_on_top(true)
        .visible(false)
        .build()
        .map_err(|e| e.to_string())?;

    let panel = window.to_panel().map_err(|e| e.to_string())?;
    // Borderless + non-activating, but keep it resizable so the user can drag
    // it larger to see more output.
    panel.set_style_mask(NS_NONACTIVATING_PANEL | NS_RESIZABLE);
    // Only grab key focus when the user actually needs to type.
    panel.set_becomes_key_only_if_needed(true);
    // Keep the panel alive across close so it can be re-shown quickly.
    panel.set_released_when_closed(false);
    panel.set_level(NS_FLOATING_WINDOW_LEVEL as i32);

    // Follow across Spaces and float over full-screen apps. Set directly on the
    // NSWindow so we don't need to import AppKit collection-behavior types.
    if let Ok(ptr) = window.ns_window() {
        if !ptr.is_null() {
            let ns_window = ptr as *mut AnyObject;
            unsafe {
                let _: () = msg_send![&*ns_window, setCollectionBehavior: NS_COLLECTION_BEHAVIOR];
            }
        }
    }

    Ok(())
}

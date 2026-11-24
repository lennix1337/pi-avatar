//! Minimal Windows platform helpers: Per-Monitor V2 DPI awareness, exact monitor work area in
//! physical pixels, live per-frame window verification (exists, visible, not minimized, not
//! cloaked, not occluded, current DWM bounds), cursor position, WS_EX_TOOLWINDOW, 1 ms timer
//! resolution, and process Working Set.

use crate::pet_state::{Rect, WindowCandidate};

#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
mod imp {
    use super::{Rect, WindowCandidate};
    use crate::pet_state::is_valid_sit_target;
    use std::ffi::c_void;
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
    use windows_sys::Win32::Graphics::Dwm::{
        DWMWA_CLOAKED, DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute,
    };
    use windows_sys::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITOR_DEFAULTTOPRIMARY, MONITORINFO,
        MonitorFromPoint, MonitorFromWindow,
    };
    use windows_sys::Win32::UI::HiDpi::{
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForMonitor, GetDpiForSystem,
        GetDpiForWindow, MDT_EFFECTIVE_DPI, SetProcessDpiAwarenessContext,
    };
    use windows_sys::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GW_HWNDPREV, GWL_EXSTYLE, GWL_STYLE, GetClassNameW, GetCursorPos,
        GetForegroundWindow, GetWindow, GetWindowLongPtrW, GetWindowRect, GetWindowThreadProcessId,
        IsIconic, IsWindow, IsWindowVisible, SPI_GETWORKAREA, STYLESTRUCT, SWP_FRAMECHANGED, SWP_NOACTIVATE,
        SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, SetWindowLongPtrW, SetWindowPos,
        SystemParametersInfoW, WM_NCDESTROY, WM_STYLECHANGING, WS_CHILD, WS_DISABLED, WS_EX_APPWINDOW, WS_EX_TOOLWINDOW,
        WS_EX_TRANSPARENT,
    };

    const EXCLUDED_CLASSES: &[&str] = &[
        "Progman",
        "WorkerW",
        "Shell_TrayWnd",
        "Shell_SecondaryTrayWnd",
        "NotifyIconOverflowWindow",
        "#32768",
        "tooltips_class32",
        "Windows.UI.Core.CoreWindow",
        "TaskListThumbnailWnd",
        "ForegroundStaging",
        "MultitaskingViewFrame",
        "Xaml_WindowedPopupClass",
        "TopLevelWindowForOverflowXamlIsland",
    ];

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "winmm")]
    unsafe extern "system" {
        fn timeBeginPeriod(u_period: u32) -> u32;
        fn timeEndPeriod(u_period: u32) -> u32;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            ppsmem_counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    /// Requests Per-Monitor V2 DPI awareness early at startup so initial monitor work-area
    /// queries return real physical coordinates before winit creates any window.
    pub fn ensure_dpi_awareness() {
        // SAFETY: safe idempotent process-wide Win32 DPI awareness opt-in.
        unsafe {
            let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
    }

    /// Requests 1 ms Windows timer resolution while alive so 60 FPS waits fire accurately.
    pub struct TimerResolutionGuard {
        active: bool,
    }

    impl TimerResolutionGuard {
        pub fn acquire_1ms() -> Self {
            // SAFETY: winmm timeBeginPeriod(1) is thread-safe and paired with timeEndPeriod(1) in Drop.
            let ok = unsafe { timeBeginPeriod(1) == 0 };
            Self { active: ok }
        }
    }

    impl Drop for TimerResolutionGuard {
        fn drop(&mut self) {
            if self.active {
                // SAFETY: matches timeBeginPeriod(1) above.
                unsafe {
                    timeEndPeriod(1);
                }
            }
        }
    }

    pub fn process_working_set_mb() -> f64 {
        let mut pmc = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            ..Default::default()
        };
        // SAFETY: querying current process memory counters into a sized struct.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) };
        if ok != 0 {
            pmc.working_set_size as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }

    pub fn system_scale_factor() -> f64 {
        // SAFETY: GetDpiForSystem reads system DPI without pointer arguments.
        let dpi = unsafe { GetDpiForSystem() };
        if dpi > 0 { dpi as f64 / 96.0 } else { 1.0 }
    }

    unsafe fn monitor_scale_factor(
        mon: windows_sys::Win32::Graphics::Gdi::HMONITOR,
    ) -> f64 {
        let mut dpi_x: u32 = 0;
        let mut dpi_y: u32 = 0;
        // SAFETY: querying effective DPI for a valid HMONITOR.
        let hr = unsafe { GetDpiForMonitor(mon, MDT_EFFECTIVE_DPI, &mut dpi_x, &mut dpi_y) };
        if hr == 0 && dpi_x > 0 {
            dpi_x as f64 / 96.0
        } else {
            system_scale_factor()
        }
    }

    fn rect_from_win32(r: RECT) -> Rect {
        Rect {
            x: r.left,
            y: r.top,
            width: (r.right - r.left).max(0),
            height: (r.bottom - r.top).max(0),
        }
    }

    fn hwnd_of(win: &rune::WindowHandle) -> Option<HWND> {
        win.with_raw_handle(|handle| {
            if let rune::raw_window_handle::RawWindowHandle::Win32(h) = handle.as_raw() {
                Some(h.hwnd.get() as HWND)
            } else {
                None
            }
        })
        .ok()
        .flatten()
    }

    /// Returns `(work_area_physical_px, dpi_scale)` for the monitor hosting `win` (or primary).
    pub fn window_work_area_and_scale(win: Option<&rune::WindowHandle>) -> (Rect, f64) {
        if let Some(w) = win
            && let Some(hwnd) = hwnd_of(w)
        {
            // SAFETY: reading DPI and monitor work area for our live window.
            unsafe {
                let dpi = GetDpiForWindow(hwnd);
                let mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
                let scale = if dpi > 0 {
                    dpi as f64 / 96.0
                } else if !mon.is_null() {
                    monitor_scale_factor(mon)
                } else {
                    system_scale_factor()
                };
                if !mon.is_null() {
                    let mut info: MONITORINFO = std::mem::zeroed();
                    info.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
                    if GetMonitorInfoW(mon, &mut info) != 0 {
                        return (rect_from_win32(info.rcWork), scale);
                    }
                }
            }
        }
        primary_work_area()
    }

    /// Returns `(work_area_physical_px, dpi_scale)` for the monitor containing `(screen_x, screen_y)`.
    pub fn monitor_work_area_and_scale_at(screen_x: i32, screen_y: i32) -> (Rect, f64) {
        // SAFETY: querying nearest monitor work area and effective DPI for a desktop point.
        unsafe {
            let mon = MonitorFromPoint(
                POINT {
                    x: screen_x,
                    y: screen_y,
                },
                MONITOR_DEFAULTTONEAREST,
            );
            if !mon.is_null() {
                let mut info: MONITORINFO = std::mem::zeroed();
                info.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
                if GetMonitorInfoW(mon, &mut info) != 0 {
                    let scale = monitor_scale_factor(mon);
                    return (rect_from_win32(info.rcWork), scale);
                }
            }
        }
        primary_work_area()
    }

    /// Returns `(primary_work_area_physical_px, primary_dpi_scale)`.
    pub fn primary_work_area() -> (Rect, f64) {
        // SAFETY: querying primary monitor work area and effective DPI.
        unsafe {
            let mon = MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY);
            if !mon.is_null() {
                let mut info: MONITORINFO = std::mem::zeroed();
                info.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
                if GetMonitorInfoW(mon, &mut info) != 0 {
                    let scale = monitor_scale_factor(mon);
                    return (rect_from_win32(info.rcWork), scale);
                }
            }
            let scale = system_scale_factor();
            let mut r = RECT {
                left: 0,
                top: 0,
                right: 1280,
                bottom: 720,
            };
            let ok = SystemParametersInfoW(
                SPI_GETWORKAREA,
                0,
                (&mut r as *mut RECT).cast::<c_void>(),
                0,
            );
            if ok != 0 {
                (rect_from_win32(r), scale)
            } else {
                (
                    Rect {
                        x: 0,
                        y: 0,
                        width: 1280,
                        height: 720,
                    },
                    scale,
                )
            }
        }
    }

    pub fn cursor_position_px() -> Option<(i32, i32)> {
        let mut pt = POINT { x: 0, y: 0 };
        // SAFETY: GetCursorPos writes screen coordinates to `pt`.
        let ok = unsafe { GetCursorPos(&mut pt) };
        if ok != 0 { Some((pt.x, pt.y)) } else { None }
    }

    pub fn window_rect_px(win: &rune::WindowHandle) -> Option<(i32, i32, i32, i32)> {
        let hwnd = hwnd_of(win)?;
        let mut r = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        // SAFETY: querying bounding rectangle of our own live window.
        let ok = unsafe { GetWindowRect(hwnd, &mut r) };
        if ok != 0 && r.right > r.left && r.bottom > r.top {
            Some((r.left, r.top, r.right - r.left, r.bottom - r.top))
        } else {
            None
        }
    }

    unsafe fn is_window_cloaked(hwnd: HWND) -> bool {
        let mut cloaked: u32 = 0;
        let hr = unsafe {
            DwmGetWindowAttribute(
                hwnd,
                DWMWA_CLOAKED as u32,
                (&mut cloaked as *mut u32).cast::<c_void>(),
                std::mem::size_of::<u32>() as u32,
            )
        };
        hr == 0 && cloaked != 0
    }

    unsafe fn window_extended_rect(hwnd: HWND) -> Option<Rect> {
        let mut r = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        let hr = unsafe {
            DwmGetWindowAttribute(
                hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS as u32,
                (&mut r as *mut RECT).cast::<c_void>(),
                std::mem::size_of::<RECT>() as u32,
            )
        };
        if hr != 0 && unsafe { GetWindowRect(hwnd, &mut r) } == 0 {
            return None;
        }
        let rect = rect_from_win32(r);
        if rect.width > 0 && rect.height > 0 {
            Some(rect)
        } else {
            None
        }
    }

    unsafe fn is_excluded_window_class(hwnd: HWND) -> bool {
        let mut buf = [0u16; 128];
        let len = unsafe { GetClassNameW(hwnd, buf.as_mut_ptr(), buf.len() as i32) };
        if len <= 0 {
            return false;
        }
        let name = String::from_utf16_lossy(&buf[..len as usize]);
        EXCLUDED_CLASSES.iter().any(|&c| c == name)
    }

    unsafe fn is_top_edge_occluded(
        hwnd: HWND,
        rect: Rect,
        probe_x: i32,
        ignore_pid: u32,
    ) -> bool {
        let probe_y = rect.y + 4;
        let mut cur = unsafe { GetWindow(hwnd, GW_HWNDPREV) };
        let mut steps = 0usize;
        while !cur.is_null() && steps < 128 {
            steps += 1;
            let visible = unsafe { IsWindowVisible(cur) != 0 };
            let iconic = unsafe { IsIconic(cur) != 0 };
            if visible && !iconic {
                let mut pid: u32 = 0;
                unsafe { GetWindowThreadProcessId(cur, &mut pid) };
                if ignore_pid == 0 || pid != ignore_pid {
                    let ex = unsafe { GetWindowLongPtrW(cur, GWL_EXSTYLE) } as u32;
                    let is_overlay =
                        (ex & (WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT)) != 0;
                    if !is_overlay
                        && !unsafe { is_window_cloaked(cur) }
                        && !unsafe { is_excluded_window_class(cur) }
                        && let Some(r) = unsafe { window_extended_rect(cur) }
                        && probe_x >= r.x
                        && probe_x <= r.right()
                        && probe_y >= r.y
                        && probe_y <= r.bottom()
                    {
                        return true;
                    }
                }
            }
            cur = unsafe { GetWindow(cur, GW_HWNDPREV) };
        }
        false
    }

    unsafe fn inspect_hwnd(
        hwnd: HWND,
        ignore_pid: u32,
        probe_x: Option<i32>,
    ) -> Option<WindowCandidate> {
        if hwnd.is_null() {
            return None;
        }
        let id = hwnd as usize as u64;
        if unsafe { IsWindow(hwnd) } == 0 {
            return Some(WindowCandidate {
                id,
                rect: Rect { x: 0, y: 0, width: 0, height: 0 },
                exists: false,
                visible: false,
                minimized: false,
                cloaked: false,
                occluded: false,
            });
        }
        let mut pid: u32 = 0;
        unsafe { GetWindowThreadProcessId(hwnd, &mut pid) };
        if ignore_pid != 0 && pid == ignore_pid {
            return None;
        }
        let style = unsafe { GetWindowLongPtrW(hwnd, GWL_STYLE) } as u32;
        let ex_style = unsafe { GetWindowLongPtrW(hwnd, GWL_EXSTYLE) } as u32;
        if (style & (WS_CHILD | WS_DISABLED)) != 0 || (ex_style & WS_EX_TOOLWINDOW) != 0 {
            return None;
        }
        if unsafe { is_excluded_window_class(hwnd) } {
            return None;
        }
        let visible = unsafe { IsWindowVisible(hwnd) != 0 };
        let minimized = unsafe { IsIconic(hwnd) != 0 };
        let cloaked = unsafe { is_window_cloaked(hwnd) };
        if !visible || minimized || cloaked {
            return Some(WindowCandidate {
                id,
                rect: Rect { x: 0, y: 0, width: 0, height: 0 },
                exists: true,
                visible,
                minimized,
                cloaked,
                occluded: false,
            });
        }
        let Some(rect) = (unsafe { window_extended_rect(hwnd) }) else {
            return None;
        };
        let px = probe_x
            .unwrap_or(rect.x + rect.width / 2)
            .clamp(rect.x + 2, (rect.right() - 2).max(rect.x + 2));
        let occluded = unsafe { is_top_edge_occluded(hwnd, rect, px, ignore_pid) };
        Some(WindowCandidate {
            id,
            rect,
            exists: true,
            visible: true,
            minimized: false,
            cloaked: false,
            occluded,
        })
    }

    /// Fast per-frame verification of a single tracked HWND.
    pub fn query_window_candidate(
        id: u64,
        ignore_pid: u32,
        probe_x: Option<i32>,
    ) -> Option<WindowCandidate> {
        let hwnd = id as usize as HWND;
        // SAFETY: Win32 read-only queries guarded by IsWindow.
        unsafe { inspect_hwnd(hwnd, ignore_pid, probe_x) }
    }

    struct EnumCtx {
        ignore_pid: u32,
        work_area: Rect,
        scale: f64,
        found: Vec<WindowCandidate>,
        max_count: usize,
    }

    unsafe extern "system" fn enum_top_level_proc(hwnd: HWND, lparam: LPARAM) -> i32 {
        // SAFETY: `lparam` points to stack-owned `EnumCtx` for the duration of `EnumWindows`.
        let ctx = unsafe { &mut *(lparam as *mut EnumCtx) };
        if let Some(cand) = unsafe { inspect_hwnd(hwnd, ctx.ignore_pid, None) }
            && is_valid_sit_target(Some(cand), ctx.work_area, ctx.scale)
            && !ctx.found.iter().any(|existing| existing.id == cand.id)
        {
            ctx.found.push(cand);
            if ctx.found.len() >= ctx.max_count {
                return 0;
            }
        }
        1
    }

    /// Finds top Z-order real visible windows on `work_area` with unoccluded top edges suitable
    /// for sitting (foreground window first if valid).
    pub fn find_sit_candidates(
        ignore_pid: u32,
        work_area: Rect,
        scale: f64,
    ) -> Vec<WindowCandidate> {
        let mut found = Vec::with_capacity(8);
        // SAFETY: read-only Win32 foreground and top-level window enumeration.
        unsafe {
            let fg = GetForegroundWindow();
            if let Some(cand) = inspect_hwnd(fg, ignore_pid, None)
                && is_valid_sit_target(Some(cand), work_area, scale)
            {
                found.push(cand);
            }
            let mut ctx = EnumCtx {
                ignore_pid,
                work_area,
                scale,
                found,
                max_count: 12,
            };
            let _ = EnumWindows(Some(enum_top_level_proc), (&mut ctx as *mut EnumCtx) as LPARAM);
            ctx.found
        }
    }

    /// Finds the highest Z-order real visible window on `work_area` with an unoccluded top edge
    /// suitable for sitting (prefers the foreground window if valid).
    #[allow(dead_code)]
    pub fn find_sit_candidate(
        ignore_pid: u32,
        work_area: Rect,
        scale: f64,
    ) -> Option<WindowCandidate> {
        find_sit_candidates(ignore_pid, work_area, scale)
            .into_iter()
            .next()
    }

    fn tool_window_style(ex: u32) -> u32 {
        (ex | WS_EX_TOOLWINDOW) & !WS_EX_APPWINDOW
    }

    unsafe extern "system" fn tool_window_proc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
        id: usize,
        _: usize,
    ) -> LRESULT {
        // SAFETY: Win32 supplies STYLESTRUCT during this notification and owns the subclass chain.
        unsafe {
            if msg == WM_STYLECHANGING && wparam as i32 == GWL_EXSTYLE {
                // winit rebuilds the extended styles when the transparent-area hit test changes.
                let styles = &mut *(lparam as *mut STYLESTRUCT);
                styles.styleNew = tool_window_style(styles.styleNew);
            } else if msg == WM_NCDESTROY {
                RemoveWindowSubclass(hwnd, Some(tool_window_proc), id);
            }
            DefSubclassProc(hwnd, msg, wparam, lparam)
        }
    }

    pub fn apply_tool_window_style(win: &rune::WindowHandle) {
        if let Some(hwnd) = hwnd_of(win) {
            // SAFETY: updating extended style bits on our own live window on the UI thread.
            unsafe {
                SetWindowSubclass(hwnd, Some(tool_window_proc), 0, 0);
                let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32;
                let next = tool_window_style(ex);
                if next != ex {
                    SetWindowLongPtrW(hwnd, GWL_EXSTYLE, next as isize);
                    SetWindowPos(
                        hwnd,
                        std::ptr::null_mut(),
                        0,
                        0,
                        0,
                        0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
                    );
                }
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod imp {
    use super::{Rect, WindowCandidate};

    pub fn ensure_dpi_awareness() {}

    pub struct TimerResolutionGuard;
    impl TimerResolutionGuard {
        pub fn acquire_1ms() -> Self {
            Self
        }
    }

    pub fn process_working_set_mb() -> f64 {
        0.0
    }

    pub fn window_work_area_and_scale(_win: Option<&rune::WindowHandle>) -> (Rect, f64) {
        primary_work_area()
    }

    pub fn monitor_work_area_and_scale_at(_screen_x: i32, _screen_y: i32) -> (Rect, f64) {
        primary_work_area()
    }

    pub fn primary_work_area() -> (Rect, f64) {
        (
            Rect {
                x: 0,
                y: 0,
                width: 1280,
                height: 720,
            },
            1.0,
        )
    }

    pub fn cursor_position_px() -> Option<(i32, i32)> {
        None
    }

    pub fn window_rect_px(_win: &rune::WindowHandle) -> Option<(i32, i32, i32, i32)> {
        None
    }

    pub fn query_window_candidate(
        _id: u64,
        _ignore_pid: u32,
        _probe_x: Option<i32>,
    ) -> Option<WindowCandidate> {
        None
    }

    pub fn find_sit_candidates(
        _ignore_pid: u32,
        _work_area: Rect,
        _scale: f64,
    ) -> Vec<WindowCandidate> {
        Vec::new()
    }

    #[allow(dead_code)]
    pub fn find_sit_candidate(
        _ignore_pid: u32,
        _work_area: Rect,
        _scale: f64,
    ) -> Option<WindowCandidate> {
        None
    }

    pub fn apply_tool_window_style(_win: &rune::WindowHandle) {}
}

pub use imp::{
    TimerResolutionGuard, apply_tool_window_style, cursor_position_px, ensure_dpi_awareness,
    find_sit_candidates, monitor_work_area_and_scale_at, primary_work_area, process_working_set_mb,
    query_window_candidate, window_rect_px, window_work_area_and_scale,
};

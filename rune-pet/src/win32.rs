//! Minimal Windows platform helpers: per-monitor DPI & work area, foreground window rect,
//! cursor position, window rect, WS_EX_TOOLWINDOW, 1 ms timer resolution, and process Working Set.

use crate::pet_state::Rect;

#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
mod imp {
    use super::Rect;
    use std::ffi::c_void;
    use windows_sys::Win32::Foundation::{HWND, POINT, RECT};
    use windows_sys::Win32::Graphics::Dwm::{DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute};
    use windows_sys::Win32::Graphics::Gdi::{
        GetMonitorInfoW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow,
    };
    use windows_sys::Win32::UI::HiDpi::{GetDpiForSystem, GetDpiForWindow};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GWL_EXSTYLE, GetCursorPos, GetForegroundWindow, GetWindowLongPtrW, GetWindowRect,
        GetWindowThreadProcessId, IsIconic, IsWindowVisible, SPI_GETWORKAREA, SWP_FRAMECHANGED,
        SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, SetWindowLongPtrW, SetWindowPos,
        SystemParametersInfoW, WS_EX_APPWINDOW, WS_EX_TOOLWINDOW,
    };

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

    pub fn window_work_area_and_scale(win: Option<&rune::WindowHandle>) -> (Rect, f64) {
        if let Some(w) = win
            && let Some(hwnd) = hwnd_of(w)
        {
            // SAFETY: reading DPI and monitor work area for our live window.
            unsafe {
                let dpi = GetDpiForWindow(hwnd);
                let scale = if dpi > 0 {
                    dpi as f64 / 96.0
                } else {
                    system_scale_factor()
                };
                let mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
                let mut info: MONITORINFO = std::mem::zeroed();
                info.cbSize = std::mem::size_of::<MONITORINFO>() as u32;
                if GetMonitorInfoW(mon, &mut info) != 0 {
                    let r = info.rcWork;
                    return (
                        Rect {
                            x: (r.left as f64 / scale).round() as i32,
                            y: (r.top as f64 / scale).round() as i32,
                            width: ((r.right - r.left) as f64 / scale).round() as i32,
                            height: ((r.bottom - r.top) as f64 / scale).round() as i32,
                        },
                        scale,
                    );
                }
            }
        }
        primary_work_area()
    }

    pub fn primary_work_area() -> (Rect, f64) {
        let scale = system_scale_factor();
        let mut r = RECT {
            left: 0,
            top: 0,
            right: 1280,
            bottom: 720,
        };
        // SAFETY: `r` is a valid mutable RECT for SPI_GETWORKAREA.
        let ok = unsafe {
            SystemParametersInfoW(SPI_GETWORKAREA, 0, (&mut r as *mut RECT).cast::<c_void>(), 0)
        };
        if ok != 0 {
            let rect = Rect {
                x: (r.left as f64 / scale).round() as i32,
                y: (r.top as f64 / scale).round() as i32,
                width: ((r.right - r.left) as f64 / scale).round() as i32,
                height: ((r.bottom - r.top) as f64 / scale).round() as i32,
            };
            (rect, scale)
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

    pub fn foreground_window_rect(ignore_pid: u32, scale_factor: f64) -> Option<Rect> {
        // SAFETY: standard Win32 read-only window queries.
        unsafe {
            let hwnd = GetForegroundWindow();
            if hwnd.is_null() || IsWindowVisible(hwnd) == 0 || IsIconic(hwnd) != 0 {
                return None;
            }
            let mut pid: u32 = 0;
            GetWindowThreadProcessId(hwnd, &mut pid);
            if ignore_pid != 0 && pid == ignore_pid {
                return None;
            }
            let mut r = RECT {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            };
            let hr = DwmGetWindowAttribute(
                hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS as u32,
                (&mut r as *mut RECT).cast::<c_void>(),
                std::mem::size_of::<RECT>() as u32,
            );
            if hr != 0 && GetWindowRect(hwnd, &mut r) == 0 {
                return None;
            }
            let w = r.right - r.left;
            let h = r.bottom - r.top;
            if w <= 0 || h <= 0 {
                return None;
            }
            let scale = if scale_factor > 0.0 { scale_factor } else { 1.0 };
            Some(Rect {
                x: (r.left as f64 / scale).round() as i32,
                y: (r.top as f64 / scale).round() as i32,
                width: (w as f64 / scale).round() as i32,
                height: (h as f64 / scale).round() as i32,
            })
        }
    }

    pub fn apply_tool_window_style(win: &rune::WindowHandle) {
        if let Some(hwnd) = hwnd_of(win) {
            // SAFETY: updating extended style bits on our own live window on the UI thread.
            unsafe {
                let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
                let next = (ex | (WS_EX_TOOLWINDOW as isize)) & !(WS_EX_APPWINDOW as isize);
                if next != ex {
                    SetWindowLongPtrW(hwnd, GWL_EXSTYLE, next);
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
    use super::Rect;

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

    pub fn foreground_window_rect(_ignore_pid: u32, _scale_factor: f64) -> Option<Rect> {
        None
    }

    pub fn apply_tool_window_style(_win: &rune::WindowHandle) {}
}

pub use imp::{
    TimerResolutionGuard, apply_tool_window_style, cursor_position_px, foreground_window_rect,
    primary_work_area, process_working_set_mb, window_rect_px, window_work_area_and_scale,
};

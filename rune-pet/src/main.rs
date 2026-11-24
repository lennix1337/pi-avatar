#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

//! Native Rust desktop pet v2 built with the Rune GPU UI framework and a VRM 1.0 `Compositor`.
//!
//! Live desktop run:
//!   `pi-avatar.exe`
//!   `pi-avatar.exe --state-file path/to/harness-state.json`
//!   `pi-avatar.exe --measure-secs 25` (or `--force-idle` / `--force-walk`)
//! Headless PNG snapshots (via Rune's built-in tooling):
//!   `pi-avatar.exe --snap docs/rune-shot-idle.png --scene idle --scale 2`

mod animation;
mod delight;
mod harness;
mod hit_mask;
mod pacing;
mod pet_state;
mod poses;
mod renderer;
mod vrm;
mod win32;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use rune::tok::type_;
use rune::{
    App, AppSpec, Color, Cx, El, Phase, Tray, TrayEvent, TrayIcon, TrayItem, TraySpec, Tx, Waker,
    WindowHandle, WindowOptions, col, el, row, text,
};

use crate::delight::DelightController;
use crate::harness::{HarnessMood, HarnessWatcher, MoodPresentation, mood_preset};
use crate::hit_mask::{AvatarHitMask, ClickThroughController, ClickZone};
use crate::pacing::{FramePacer, PaceDecision, TARGET_FPS};
use crate::pet_state::{
    DRAG_THRESHOLD_DIP, ForcedBehavior, PET_WINDOW_HEIGHT, PET_WINDOW_WIDTH, PetMode,
    PetStateMachine, Rect, WindowCandidate, floor_window_y, scaled_avatar_size, scaled_px,
};
pub use crate::poses::EvaluatedPose;
use crate::poses::{BlenderInput, PoseBlender, PoseMap, load_poses};
use crate::renderer::{RenderMetrics, VrmCompositor};
use crate::vrm::VrmModel;

const MEASURE_WARMUP_SEC: f64 = 1.0;

#[derive(Clone, Debug)]
struct CliConfig {
    forced_behavior: ForcedBehavior,
    sit_first: bool,
    state_file: Option<PathBuf>,
    measure_secs: Option<u64>,
    stay_open_ms: Option<u64>,
    stats_file: Option<PathBuf>,
    rune_args: Vec<String>,
}

static CLI: OnceLock<CliConfig> = OnceLock::new();
static WINDOW_SLOT: OnceLock<Arc<Mutex<Option<WindowHandle>>>> = OnceLock::new();
static TRAY_QUEUE: OnceLock<Arc<Mutex<Vec<TrayEvent>>>> = OnceLock::new();
static RENDER_METRICS: OnceLock<Arc<Mutex<RenderMetrics>>> = OnceLock::new();
static HIT_MASK_SLOT: OnceLock<Arc<Mutex<AvatarHitMask>>> = OnceLock::new();

pub struct PetAppState {
    blender: PoseBlender,
    pet: PetStateMachine,
    pacer: FramePacer,
    harness: HarnessWatcher,
    preview_mood: Option<MoodPresentation>,
    delight: DelightController,
    click_through: ClickThroughController,

    scale: f64,
    my_pid: u32,
    last_window_pos: Option<(i32, i32)>,
    last_harness_poll_t: f64,
    last_env_poll_t: f64,
    cached_candidates: Vec<WindowCandidate>,

    forced_preview: bool,
    press_origin_px: Option<(f32, f32)>,
    press_window_pt: Option<(f32, f32)>,
    drag_active: bool,
    tray: Option<Tray>,
}

impl PetAppState {
    fn new(
        poses: PoseMap,
        work_area: Rect,
        scale: f64,
        forced: ForcedBehavior,
        sit_first: bool,
        state_file: Option<PathBuf>,
    ) -> Self {
        let mut pet = PetStateMachine::new(work_area, scale, 0x2026_1123);
        pet.set_forced_behavior(forced);
        pet.set_prefer_sit_first(sit_first);
        let snap = pet.snapshot();
        let blender = PoseBlender::new(poses, snap.pose, snap.expression, snap.facing);

        let mut harness = HarnessWatcher::new(state_file);
        let _ = harness.poll();

        Self {
            blender,
            pet,
            pacer: FramePacer::new(TARGET_FPS),
            harness,
            preview_mood: None,
            delight: DelightController::new(),
            click_through: ClickThroughController::new(),
            scale,
            my_pid: std::process::id(),
            last_window_pos: None,
            last_harness_poll_t: -10.0,
            last_env_poll_t: -10.0,
            cached_candidates: Vec::new(),
            forced_preview: false,
            press_origin_px: None,
            press_window_pt: None,
            drag_active: false,
            tray: None,
        }
    }

    pub fn active_mood(&self) -> Option<&MoodPresentation> {
        self.preview_mood.as_ref().or_else(|| self.harness.current())
    }

    pub fn bubble_text(&self) -> Option<(&str, [u8; 3])> {
        if let Some(m) = self.active_mood() {
            return Some((m.bubble_text.as_str(), m.accent_rgb));
        }
        if self.pet.snapshot().sleeping_local {
            return Some(("Zzz...", [139, 92, 246]));
        }
        None
    }

    pub fn should_render(&self) -> bool {
        if self.forced_preview {
            return false;
        }
        self.pet.snapshot().should_render
    }

    fn apply_scene_preview(
        &mut self,
        pose_name: &'static str,
        expr_name: &'static str,
        walking: bool,
        facing: i8,
        walk_phase: f32,
        mood: Option<MoodPresentation>,
    ) {
        self.forced_preview = true;
        let eyes_closed = mood.as_ref().is_some_and(|m| m.eyes_closed);
        self.preview_mood = mood;
        self.pet.set_forced_behavior(if walking {
            ForcedBehavior::ForceWalk
        } else if pose_name == "cheer" {
            ForcedBehavior::ForceArmsUp
        } else {
            ForcedBehavior::ForceIdle
        });
        self.blender.apply_preview(
            pose_name,
            expr_name,
            walking,
            facing,
            walk_phase,
            eyes_closed,
        );
    }

    fn refresh_live_desktop_environment(&mut self, now_sec: f64) {
        if rune::is_headless() {
            return;
        }
        // Periodically refresh per-monitor work area / DPI and sit candidates (4 Hz).
        if !self.drag_active && now_sec - self.last_env_poll_t >= 0.25 {
            self.last_env_poll_t = now_sec;
            let win_opt = WINDOW_SLOT
                .get()
                .and_then(|s| s.lock().ok())
                .and_then(|g| g.clone());
            let (wa, scale) = win32::window_work_area_and_scale(win_opt.as_ref());
            let env_changed = wa != self.pet.work_area() || (scale - self.pet.scale()).abs() > 1e-4;
            self.scale = scale;
            self.pet.set_environment(wa, scale);
            if env_changed {
                // Force immediate window reposition on DPI or monitor change.
                self.last_window_pos = None;
            }
            if self.pet.tracked_window_id().is_none() {
                self.cached_candidates = win32::find_sit_candidates(self.my_pid, wa, scale);
                self.pet.sync_window_candidates(&self.cached_candidates);
            }
        }

        // Verify the active sit/climb target window on EVERY frame (exists, visible, not minimized,
        // not cloaked, not occluded, current rect) so closed/minimized windows never hold the pet.
        let tracked = self.pet.tracked_window_id().and_then(|id| {
            win32::query_window_candidate(id, self.my_pid, Some(self.pet.sit_anchor_center_x()))
        });
        self.pet
            .sync_window_targets(tracked, self.cached_candidates.first().copied());
    }

    fn step_animation(&mut self, dt_sec: f32, now_sec: f64) -> bool {
        if self.forced_preview {
            return false;
        }

        self.refresh_live_desktop_environment(now_sec);

        // Poll local JSON harness state file every 500 ms (skips disk read when unchanged).
        if now_sec - self.last_harness_poll_t >= 0.5 {
            self.last_harness_poll_t = now_sec;
            let _ = self.harness.poll();
        }

        let mood_override = self
            .harness
            .current()
            .map(|m| (m.mood, m.pose, m.expression, m.eyes_closed));
        if mood_override.is_some() {
            self.pet.notify_user_activity();
        }
        let mut snap = if mood_override.is_some() && !self.drag_active {
            // While reacting to a harness mood, stay attentive in place.
            let mut s = self.pet.snapshot();
            s.walking = false;
            s.walk_speed_factor = 0.0;
            s.facing = 0;
            s
        } else {
            self.pet.step(dt_sec)
        };

        self.sync_window_position(snap.x, snap.y);
        let pat_triggered = self.update_cursor_interactions(dt_sec, snap);
        if pat_triggered {
            self.pet.on_petting();
            snap = self.pet.snapshot();
        }

        let target_pose = mood_override.map_or(snap.pose, |(_, pose, _, _)| pose);
        let target_expr = if self.delight.happy_boost_sec > 0.0 || self.delight.petting_intensity > 0.35 {
            "happy"
        } else {
            mood_override.map_or(snap.expression, |(_, _, expr, _)| expr)
        };
        let eyes_closed = mood_override.map_or(snap.eyes_closed, |(_, _, _, closed)| closed);
        let gaze_active = !snap.walking
            && !eyes_closed
            && !self.drag_active
            && snap.mode != PetMode::Fall;

        self.blender.step(BlenderInput {
            dt_sec,
            target_pose,
            target_expression: target_expr,
            walking: snap.walking && mood_override.is_none(),
            walk_dx_px: snap.dx_px,
            walk_speed_factor: snap.walk_speed_factor,
            scale: self.scale,
            facing: snap.facing,
            eyes_closed,
            carry_sway: snap.carry_sway,
            head_yaw: if gaze_active { self.delight.head_yaw } else { 0.0 },
            head_pitch: if gaze_active { self.delight.head_pitch } else { 0.0 },
            neck_yaw: if gaze_active { self.delight.neck_yaw } else { 0.0 },
            neck_pitch: if gaze_active { self.delight.neck_pitch } else { 0.0 },
            chest_yaw: if gaze_active { self.delight.chest_yaw } else { 0.0 },
            eyes: if gaze_active { self.delight.eyes } else { [0.0; 2] },
            cursor_attention: gaze_active && self.delight.cursor_proximity > 0.1,
            harness_mood: mood_override.map(|(mood, _, _, _)| mood),
            petting_intensity: if !self.drag_active && snap.mode != PetMode::Fall {
                self.delight.petting_intensity
            } else {
                0.0
            },
        });

        snap.should_render
    }

    fn update_cursor_interactions(
        &mut self,
        dt_sec: f32,
        snap: crate::pet_state::PetSnapshot,
    ) -> bool {
        if rune::is_headless() {
            return false;
        }
        let win_opt = WINDOW_SLOT
            .get()
            .and_then(|s| s.lock().ok())
            .and_then(|g| g.clone());
        let Some(win) = win_opt else { return false };

        let cursor_px = win32::cursor_position_px();
        let rect_px = win32::window_rect_px(&win);

        let mut cursor_norm = None;
        let mut cursor_rel_logical = None;
        if let (Some((cx, cy)), Some((wx, wy, ww, wh))) = (cursor_px, rect_px) {
            let ww_f = ww.max(1) as f32;
            let wh_f = wh.max(1) as f32;
            let nx = (cx - wx) as f32 / ww_f;
            let ny = (cy - wy) as f32 / wh_f;
            cursor_rel_logical = Some((nx * PET_WINDOW_WIDTH, ny * PET_WINDOW_HEIGHT));
            if (0.0..=1.0).contains(&nx) && (0.0..=1.0).contains(&ny) {
                cursor_norm = Some((nx, ny));
            }
        }

        let mask = HIT_MASK_SLOT
            .get()
            .and_then(|m| m.lock().ok())
            .map(|g| g.clone())
            .unwrap_or_default();

        // A failed platform update leaves the applied state unchanged and is retried next frame.
        let keep_capture = self.drag_active || self.press_origin_px.is_some();
        let _ = self
            .click_through
            .apply(cursor_norm, keep_capture, &mask, |desired| {
                win.set_click_through(desired)
            });

        let head_norm = mask
            .head_center()
            .unwrap_or(glam::Vec2::new(0.5, 0.30));
        let head_center_px = (
            head_norm.x * PET_WINDOW_WIDTH,
            head_norm.y * PET_WINDOW_HEIGHT,
        );
        let harness_sleeping = self
            .active_mood()
            .is_some_and(|m| m.mood == HarnessMood::Sleep);
        // Allow cursor gaze & slow-stroke petting (including waking local idle sleep).
        let allow_interact =
            !snap.walking && !harness_sleeping && !self.drag_active && snap.mode != PetMode::Fall;
        self.delight
            .update(dt_sec, cursor_rel_logical, head_center_px, allow_interact)
    }

    fn sync_window_position(&mut self, phys_x: i32, phys_y: i32) {
        if self.last_window_pos == Some((phys_x, phys_y)) {
            return;
        }
        self.last_window_pos = Some((phys_x, phys_y));
        if let Some(slot) = WINDOW_SLOT.get()
            && let Ok(guard) = slot.lock()
            && let Some(win) = guard.as_ref()
        {
            win.set_position((phys_x, phys_y));
        }
    }

    pub fn evaluate_frame_pose(&self) -> EvaluatedPose {
        self.blender.evaluate(self.forced_preview)
    }
}

const APP_NAME: &str = "Pi Avatar";

fn make_tray_icon() -> TrayIcon {
    let image = image::load_from_memory_with_format(
        include_bytes!("../assets/pi-avatar-32.png"),
        image::ImageFormat::Png,
    )
    .expect("embedded Pi Avatar icon is a valid PNG")
    .into_rgba8();
    TrayIcon::rgba(image.width(), image.height(), image.into_raw())
}

fn tray_menu(paused: bool) -> Vec<TrayItem> {
    let pause_label = if paused { "Retomar Pi Avatar" } else { "Pausar Pi Avatar" };
    vec![
        TrayItem::action("pause", pause_label),
        TrayItem::Separator,
        TrayItem::action("quit", "Sair"),
    ]
}

fn find_avatar_root() -> PathBuf {
    if let Ok(env_root) = std::env::var("AVATAR_ROOT") {
        let p = PathBuf::from(env_root);
        if p.join("poses.json").exists() {
            return p;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors() {
            if anc.join("poses.json").exists() && anc.join("models/avatar.vrm").exists() {
                return anc.to_path_buf();
            }
            let labs = anc.join("labs/avatar");
            if labs.join("poses.json").exists() && labs.join("models/avatar.vrm").exists() {
                return labs;
            }
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or(manifest)
}

fn parse_cli() -> CliConfig {
    let mut forced_behavior = ForcedBehavior::Auto;
    let mut sit_first = false;
    let mut state_file = None;
    let mut measure_secs = None;
    let mut stay_open_ms = None;
    let mut stats_file = None;
    let mut rune_args = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--force-idle" => forced_behavior = ForcedBehavior::ForceIdle,
            "--force-walk" => forced_behavior = ForcedBehavior::ForceWalk,
            "--force-arms-up" => forced_behavior = ForcedBehavior::ForceArmsUp,
            "--sit-first" => sit_first = true,
            "--state-file" => {
                if let Some(v) = args.next() {
                    state_file = Some(PathBuf::from(v));
                }
            }
            "--measure-secs" => {
                if let Some(v) = args.next() {
                    measure_secs = v.parse::<u64>().ok().filter(|&n| n > 0);
                }
            }
            "--stay-open-ms" => {
                if let Some(v) = args.next() {
                    stay_open_ms = v.parse::<u64>().ok();
                }
            }
            "--stats-file" => {
                if let Some(v) = args.next() {
                    stats_file = Some(PathBuf::from(v));
                }
            }
            _ => rune_args.push(arg),
        }
    }

    CliConfig {
        forced_behavior,
        sit_first,
        state_file,
        measure_secs,
        stay_open_ms,
        stats_file,
        rune_args,
    }
}

fn emit_measurement_report(print_stdout: bool) {
    let Some(metrics) = RENDER_METRICS.get().and_then(|m| m.lock().ok()) else {
        return;
    };
    let s = metrics.summary();
    let ws_mb = win32::process_working_set_mb();
    let mode = CLI
        .get()
        .map(|c| match c.forced_behavior {
            ForcedBehavior::Auto => "auto",
            ForcedBehavior::ForceIdle => "idle",
            ForcedBehavior::ForceWalk => "walk",
            ForcedBehavior::ForceArmsUp => "arms-up",
        })
        .unwrap_or("auto");

    let payload = serde_json::json!({
        "mode": mode,
        "frames": s.frames,
        "total_frames": metrics.total_frames,
        "duration_sec": (s.duration_sec * 100.0).round() / 100.0,
        "avg_fps": (s.avg_fps * 100.0).round() / 100.0,
        "min_fps": (s.min_fps * 100.0).round() / 100.0,
        "p5_fps": (s.p5_fps * 100.0).round() / 100.0,
        "working_set_mb": (ws_mb * 100.0).round() / 100.0,
    });
    let text = serde_json::to_string_pretty(&payload).unwrap_or_default();
    if print_stdout {
        println!("{text}");
    }
    if let Some(path) = CLI.get().and_then(|c| c.stats_file.as_ref()) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, &text);
    }
}

fn setup(s: &mut PetAppState, waker: Waker) {
    if rune::is_headless() {
        return;
    }

    let tray_queue = TRAY_QUEUE.get().unwrap().clone();
    let tray_waker = waker.clone();
    let spec = TraySpec::new("pi-avatar", APP_NAME, make_tray_icon())
        .tooltip(APP_NAME)
        .menu(tray_menu(false));
    if let Ok(tray) = Tray::new(spec, move |ev| {
        if let Ok(mut q) = tray_queue.lock() {
            q.push(ev);
        }
        tray_waker();
    }) {
        s.tray = Some(tray);
    }

    let measure_secs = CLI.get().and_then(|c| c.measure_secs);
    let stay_ms = measure_secs
        .map(|n| ((n as f64 + MEASURE_WARMUP_SEC) * 1000.0).round() as u64)
        .or_else(|| CLI.get().and_then(|c| c.stay_open_ms));

    if let Some(ms) = stay_ms {
        let win_slot = WINDOW_SLOT.get().unwrap().clone();
        let print_report = measure_secs.is_some();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            emit_measurement_report(print_report);
            if let Ok(g) = win_slot.lock()
                && let Some(win) = g.as_ref()
            {
                win.close();
            }
            std::thread::sleep(Duration::from_millis(250));
            std::process::exit(0);
        });
    }
}

fn poll(s: &mut PetAppState, t: f64, tx: &mut Tx) -> bool {
    let mut changed = false;

    // Drain tray events
    if let Some(q) = TRAY_QUEUE.get()
        && let Ok(mut guard) = q.lock()
    {
        for ev in guard.drain(..) {
            match ev {
                TrayEvent::Menu(id) if id == "pause" => {
                    let next = !s.pet.is_paused();
                    s.pet.set_paused(next);
                    if !next {
                        s.pacer.reset();
                    }
                    if let Some(tray) = &s.tray {
                        tray.set_menu(tray_menu(next));
                    }
                    changed = true;
                }
                TrayEvent::Menu(id) if id == "quit" => {
                    emit_measurement_report(CLI.get().and_then(|c| c.measure_secs).is_some());
                    tx.close_window();
                    return true;
                }
                _ => {}
            }
        }
    }

    if s.forced_preview {
        return changed;
    }

    if s.pet.is_paused() {
        // Still keep click-through responsive while paused at 10 Hz
        let snap = s.pet.snapshot();
        let _ = s.update_cursor_interactions(0.1, snap);
        tx.wake_after(Duration::from_millis(100));
        return changed;
    }

    match s.pacer.tick(t) {
        PaceDecision::Wait { next_wake } => {
            tx.wake_after(next_wake);
            changed
        }
        PaceDecision::Render { dt_sec, next_wake } => {
            if s.step_animation(dt_sec, t) {
                changed = true;
            }
            tx.wake_after(next_wake);
            changed
        }
    }
}

fn rgb_u8(c: [u8; 3], a: f32) -> Color {
    Color {
        r: c[0] as f32 / 255.0,
        g: c[1] as f32 / 255.0,
        b: c[2] as f32 / 255.0,
        a,
    }
}

fn view(s: &PetAppState, _cx: &mut Cx<PetAppState>) -> El {
    let mut root = el()
        .key("pet-surface")
        .w_full()
        .h_full()
        .on_drag(|s: &mut PetAppState, d, tx| {
            let [wx, wy] = d.window_point();
            match d.phase {
                Phase::Down => {
                    let Some((sx, sy)) = d.screen_point() else {
                        return;
                    };
                    s.press_origin_px = Some((sx as f32, sy as f32));
                    s.press_window_pt = Some((wx, wy));
                    s.drag_active = false;
                }
                Phase::Move => {
                    let Some((sx, sy)) = d.screen_point() else {
                        return;
                    };
                    let px = sx as f32;
                    let py = sy as f32;
                    if !s.drag_active
                        && let Some((ox, oy)) = s.press_origin_px
                    {
                        let thresh = scaled_px(DRAG_THRESHOLD_DIP, s.scale).max(4) as f32;
                        if (px - ox).hypot(py - oy) >= thresh {
                            s.drag_active = true;
                            s.pet.start_drag(ox, oy);
                        }
                    }
                    if s.drag_active {
                        let (wa, scale) = win32::monitor_work_area_and_scale_at(
                            px.round() as i32,
                            py.round() as i32,
                        );
                        if wa != s.pet.work_area() || (scale - s.scale).abs() > 1e-4 {
                            s.scale = scale;
                            s.pet.set_environment(wa, scale);
                            s.last_window_pos = None;
                        }
                        s.pet.move_drag(px, py);
                    }
                }
                Phase::Up => {
                    let was_drag = s.drag_active;
                    let press_pt = s.press_window_pt.take();
                    s.press_origin_px = None;
                    s.drag_active = false;
                    if was_drag {
                        if let Some((sx, sy)) = d.screen_point() {
                            let (wa, scale) = win32::monitor_work_area_and_scale_at(sx, sy);
                            s.scale = scale;
                            s.pet.set_environment(wa, scale);
                        }
                        let cands = win32::find_sit_candidates(
                            s.my_pid,
                            s.pet.work_area(),
                            s.pet.scale(),
                        );
                        s.cached_candidates = cands.clone();
                        s.pet.sync_window_candidates(&cands);
                        s.pet.end_drag_with_candidates(&cands);
                        tx.wake_after(Duration::from_millis(16));
                    } else if !d.cancelled() && d.clicks == 1 {
                        let (lx, ly) = press_pt.unwrap_or((wx, wy));
                        let nx = (lx / PET_WINDOW_WIDTH).clamp(0.0, 1.0);
                        let ny = (ly / PET_WINDOW_HEIGHT).clamp(0.0, 1.0);
                        let zone = HIT_MASK_SLOT
                            .get()
                            .and_then(|m| m.lock().ok())
                            .and_then(|mask| mask.hit_zone(nx, ny))
                            .unwrap_or(ClickZone::Body);
                        s.pet.on_click(zone);
                        s.delight.trigger_click_sparkles(lx, (ly - 12.0).max(16.0));
                        tx.wake_after(Duration::from_millis(16));
                    }
                }
            }
            let snap = s.pet.snapshot();
            s.sync_window_position(snap.x, snap.y);
        })
        .on_double_press_tx(|s: &mut PetAppState, tx| {
            s.pet.on_double_click();
            s.delight.trigger_pat_burst(PET_WINDOW_WIDTH * 0.5, 62.0);
            tx.wake_after(Duration::from_millis(16));
        });

    // Harness mood speech bubble rendered natively by Rune's SDF + Parley engine above the head.
    if let Some((msg, accent)) = s.bubble_text() {
        let dot = el()
            .key("bubble-dot")
            .size(7.0, 7.0)
            .radius(3.5)
            .fill(rgb_u8(accent, 1.0));
        let label = text(msg)
            .key("bubble-text")
            .font(type_::LABEL)
            .color(Color {
                r: 0.09,
                g: 0.11,
                b: 0.15,
                a: 1.0,
            });
        let pill = row()
            .key("speech-bubble")
            .center()
            .gap(6.0)
            .px(10.0)
            .py(5.0)
            .radius(12.0)
            .fill(Color {
                r: 0.99,
                g: 0.99,
                b: 1.0,
                a: 0.95,
            })
            .border(1.2, rgb_u8(accent, 0.75))
            .child(dot)
            .child(label);
        let tail = el()
            .key("bubble-tail")
            .size(8.0, 5.0)
            .radius(2.5)
            .fill(Color {
                r: 0.99,
                g: 0.99,
                b: 1.0,
                a: 0.95,
            })
            .border(1.0, rgb_u8(accent, 0.6));
        let bubble_box = col()
            .key("bubble-wrap")
            .w_full()
            .items_center()
            .pt(4.0)
            .gap(2.0)
            .child(pill)
            .child(tail);
        root = root.child(bubble_box);
    }

    // Delightful touch: floating pastel sparkles on head-pat / joyful double-click.
    for (i, sp) in s.delight.sparkles.iter().enumerate() {
        let progress = (sp.age / sp.lifetime).clamp(0.0, 1.0);
        let alpha = (1.0 - progress).powf(1.4);
        let color = if sp.warm {
            Color {
                r: 1.0,
                g: 0.45,
                b: 0.68,
                a: alpha,
            }
        } else {
            Color {
                r: 1.0,
                g: 0.82,
                b: 0.32,
                a: alpha,
            }
        };
        let sz = sp.size * (0.75 + 0.35 * (1.0 - progress));
        root = root.child(
            el().key(format!("sparkle-{i}"))
                .absolute(sp.x - sz * 0.5, sp.y - sz * 0.5)
                .size(sz, sz)
                .radius(sz * 0.5)
                .fill(color),
        );
    }

    root
}

fn scene_idle(s: &mut PetAppState) {
    s.apply_scene_preview("idle", "neutral", false, 0, 0.0, None);
}

fn scene_walk(s: &mut PetAppState) {
    s.apply_scene_preview("idle", "neutral", true, 1, 0.25, None);
}

fn scene_arms_up(s: &mut PetAppState) {
    s.apply_scene_preview("cheer", "happy", false, 0, 0.0, None);
}

fn scene_mood_worried(s: &mut PetAppState) {
    let m = mood_preset(HarnessMood::Worried, 0);
    s.apply_scene_preview(m.pose, m.expression, false, 0, 0.0, Some(m));
}

fn scene_mood_celebrate(s: &mut PetAppState) {
    let m = mood_preset(HarnessMood::Celebrate, 0);
    s.apply_scene_preview(m.pose, m.expression, false, 0, 0.0, Some(m));
}

fn scene_mood_working(s: &mut PetAppState) {
    let m = mood_preset(HarnessMood::Working, 2);
    s.apply_scene_preview(m.pose, m.expression, false, 0, 0.0, Some(m));
}

fn scene_mood_wave(s: &mut PetAppState) {
    let m = mood_preset(HarnessMood::Wave, 0);
    s.apply_scene_preview(m.pose, m.expression, false, 0, 0.0, Some(m));
}

fn scene_mood_sleep(s: &mut PetAppState) {
    let m = mood_preset(HarnessMood::Sleep, 0);
    s.apply_scene_preview(m.pose, m.expression, false, 0, 0.0, Some(m));
}

fn scene_delight_pat(s: &mut PetAppState) {
    s.apply_scene_preview("cheer", "happy", false, 0, 0.0, None);
    s.delight.trigger_pat_burst(PET_WINDOW_WIDTH * 0.5, 62.0);
    for sp in &mut s.delight.sparkles {
        sp.age = 0.25;
        sp.x += sp.vx * 0.25;
        sp.y += sp.vy * 0.25;
    }
}

fn init_state() -> PetAppState {
    let avatar_root = find_avatar_root();
    let poses = load_poses(&avatar_root.join("poses.json"))
        .expect("failed to load labs/avatar/poses.json");
    let (work_area, scale) = win32::primary_work_area();
    let cli = CLI.get();
    let forced = cli.map(|c| c.forced_behavior).unwrap_or(ForcedBehavior::Auto);
    let sit_first = cli.is_some_and(|c| c.sit_first);
    let state_file = cli.and_then(|c| c.state_file.clone());
    PetAppState::new(poses, work_area, scale, forced, sit_first, state_file)
}

fn main() {
    win32::ensure_dpi_awareness();

    let cli = parse_cli();
    let rune_args = cli.rune_args.clone();
    let is_headless_cli = rune_args
        .iter()
        .any(|a| matches!(a.as_str(), "--snap" | "--tree" | "--trace" | "--list-scenes"));
    let _timer_guard = (!is_headless_cli).then(win32::TimerResolutionGuard::acquire_1ms);
    let _ = CLI.set(cli);

    let win_slot = WINDOW_SLOT
        .get_or_init(|| Arc::new(Mutex::new(None)))
        .clone();
    let _ = TRAY_QUEUE.get_or_init(|| Arc::new(Mutex::new(Vec::new())));
    let metrics = RENDER_METRICS
        .get_or_init(|| Arc::new(Mutex::new(RenderMetrics::new(MEASURE_WARMUP_SEC))))
        .clone();
    let hit_mask = HIT_MASK_SLOT
        .get_or_init(|| Arc::new(Mutex::new(AvatarHitMask::default())))
        .clone();

    let avatar_root = find_avatar_root();
    let vrm_path = avatar_root.join("models/avatar.vrm");
    let vrm_model = VrmModel::load(&vrm_path).expect("failed to load models/avatar.vrm");

    let (work_area, scale) = win32::primary_work_area();
    let (avatar_w, _) = scaled_avatar_size(scale);
    let right_pad = scaled_px(80.0, scale);
    let init_x = (work_area.right() - avatar_w - right_pad).max(work_area.x);
    let init_y = floor_window_y(work_area, scale);

    let spec = AppSpec::<PetAppState> {
        min_size: Some((0.0, 0.0)),
        scenes: &[
            ("idle", scene_idle),
            ("walk", scene_walk),
            ("arms-up", scene_arms_up),
            ("mood-worried", scene_mood_worried),
            ("mood-celebrate", scene_mood_celebrate),
            ("mood-working", scene_mood_working),
            ("mood-wave", scene_mood_wave),
            ("mood-sleep", scene_mood_sleep),
            ("delight-pat", scene_delight_pat),
        ],
        setup: Some(setup),
        poll: Some(poll),
        ..AppSpec::new(
            APP_NAME,
            (PET_WINDOW_WIDTH, PET_WINDOW_HEIGHT),
            init_state,
            view,
        )
    };

    let options = WindowOptions {
        transparent: true,
        borderless: true,
        click_through: false,
        always_on_top: true,
        // Set the tool-window style before Explorer ever sees a visible pet.
        visible: !cfg!(windows),
        position: Some((init_x, init_y)),
        physical_size: None,
        min_size: Some((0.0, 0.0)),
        max_size: Some((PET_WINDOW_WIDTH, PET_WINDOW_HEIGHT)),
        resizable: false,
    };

    let win_slot_hook = win_slot.clone();
    App::new(spec)
        .args(rune_args)
        .window_options(options)
        .open_inactive()
        .srgb_blend(false)
        .compositor(VrmCompositor::new(vrm_path, vrm_model, metrics, hit_mask))
        .on_window(move |win| {
            win32::apply_tool_window_style(win);
            if let Ok(mut guard) = win_slot_hook.lock() {
                *guard = Some(win.clone());
            }
            #[cfg(windows)]
            win.show();
        })
        .run();

    emit_measurement_report(false);
}

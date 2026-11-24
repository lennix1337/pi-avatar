//! Native Rust desktop pet v2 built with the Rune GPU UI framework and a VRM 1.0 `Compositor`.
//!
//! Live desktop run:
//!   `rune-pet.exe`
//!   `rune-pet.exe --state-file path/to/harness-state.json`
//!   `rune-pet.exe --measure-secs 25` (or `--force-idle` / `--force-walk`)
//! Headless PNG snapshots (via Rune's built-in tooling):
//!   `rune-pet.exe --snap docs/rune-shot-idle.png --scene idle --scale 2`

mod delight;
mod harness;
mod hit_mask;
mod pacing;
mod pet_state;
mod poses;
mod renderer;
mod vrm;
mod win32;

use std::collections::{BTreeSet, HashMap};
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
use crate::hit_mask::{AvatarHitMask, ClickThroughController};
use crate::pacing::{FramePacer, PaceDecision, TARGET_FPS};
use crate::pet_state::{
    ForcedBehavior, PET_WINDOW_HEIGHT, PET_WINDOW_WIDTH, PetStateMachine, Rect, WindowCandidate,
    compute_walk_phase_delta, facing_to_yaw_rad, floor_window_y, scaled_avatar_size, scaled_px,
};
use crate::poses::{
    Pose, PoseMap, compute_auto_blink, compute_idle_breathing, compute_pose_secondary_motion,
    compute_walk_offsets, ease_in_out_cubic, interpolate_pose, load_poses, scale_pose,
};
use crate::renderer::{RenderMetrics, VrmCompositor};
use crate::vrm::VrmModel;

const EXPRESSION_NAMES: [&str; 5] = ["happy", "angry", "sad", "relaxed", "surprised"];
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

pub struct EvaluatedPose {
    pub pose: Pose,
    pub root_yaw: f32,
    pub root_bob_y: f32,
    pub expressions: HashMap<&'static str, f32>,
}

pub struct PetAppState {
    poses: PoseMap,
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
    cached_candidate: Option<WindowCandidate>,

    pose_name: &'static str,
    pose_from: Pose,
    pose_to: Pose,
    pose_current: Pose,
    pose_elapsed_ms: f32,
    pose_duration_ms: f32,

    expr_name: &'static str,
    expr_from: HashMap<&'static str, f32>,
    expr_to: HashMap<&'static str, f32>,
    expr_weights: HashMap<&'static str, f32>,
    expr_elapsed_ms: f32,
    expr_duration_ms: f32,

    walk_phase: f32,
    walk_blend: f32,
    facing_current: f32,
    total_time_sec: f32,
    forced_preview: bool,
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
        let initial_pose = poses.get(snap.pose).cloned().unwrap_or_default();

        let zero_expr: HashMap<&'static str, f32> =
            EXPRESSION_NAMES.iter().map(|&n| (n, 0.0)).collect();
        let mut init_expr = zero_expr.clone();
        if let Some(slot) = init_expr.get_mut(snap.expression) {
            *slot = 1.0;
        }

        let mut harness = HarnessWatcher::new(state_file);
        let _ = harness.poll();

        Self {
            poses,
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
            cached_candidate: None,
            pose_name: snap.pose,
            pose_from: initial_pose.clone(),
            pose_to: initial_pose.clone(),
            pose_current: initial_pose,
            pose_elapsed_ms: 0.0,
            pose_duration_ms: 0.0,
            expr_name: snap.expression,
            expr_from: init_expr.clone(),
            expr_to: init_expr.clone(),
            expr_weights: init_expr,
            expr_elapsed_ms: 0.0,
            expr_duration_ms: 0.0,
            walk_phase: std::f32::consts::FRAC_PI_2,
            walk_blend: 0.0,
            facing_current: facing_to_yaw_rad(snap.facing),
            total_time_sec: 0.0,
            forced_preview: false,
            drag_active: false,
            tray: None,
        }
    }

    pub fn active_mood(&self) -> Option<&MoodPresentation> {
        self.preview_mood.as_ref().or_else(|| self.harness.current())
    }

    pub fn bubble_text(&self) -> Option<(&str, [u8; 3])> {
        self.active_mood()
            .map(|m| (m.bubble_text.as_str(), m.accent_rgb))
    }

    pub fn should_render(&self) -> bool {
        if self.forced_preview {
            return false;
        }
        self.pet.snapshot().should_render
    }

    fn set_pose_target(&mut self, name: &'static str, duration_ms: f32) {
        if let Some(target) = self.poses.get(name).cloned() {
            self.pose_name = name;
            self.pose_from = self.pose_current.clone();
            self.pose_to = target;
            self.pose_elapsed_ms = 0.0;
            self.pose_duration_ms = duration_ms.max(0.0);
            if duration_ms <= 0.0 {
                self.pose_current = self.pose_to.clone();
            }
        }
    }

    fn set_expr_target(&mut self, name: &'static str, duration_ms: f32) {
        self.expr_name = name;
        self.expr_from = self.expr_weights.clone();
        self.expr_to = EXPRESSION_NAMES.iter().map(|&n| (n, 0.0)).collect();
        if let Some(slot) = self.expr_to.get_mut(name) {
            *slot = 1.0;
        }
        self.expr_elapsed_ms = 0.0;
        self.expr_duration_ms = duration_ms.max(0.0);
        if duration_ms <= 0.0 {
            self.expr_weights = self.expr_to.clone();
        }
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
        self.preview_mood = mood;
        self.pet.set_forced_behavior(if walking {
            ForcedBehavior::ForceWalk
        } else if pose_name == "cheer" {
            ForcedBehavior::ForceArmsUp
        } else {
            ForcedBehavior::ForceIdle
        });
        self.set_pose_target(pose_name, 0.0);
        self.set_expr_target(expr_name, 0.0);
        self.walk_phase = walk_phase;
        self.walk_blend = if walking { 1.0 } else { 0.0 };
        self.facing_current = facing_to_yaw_rad(facing);
        self.total_time_sec = 1.2;
    }

    fn refresh_live_desktop_environment(&mut self, now_sec: f64) {
        if rune::is_headless() {
            return;
        }
        // Periodically refresh per-monitor work area / DPI and best sit candidate (4 Hz).
        if now_sec - self.last_env_poll_t >= 0.25 {
            self.last_env_poll_t = now_sec;
            let win_opt = WINDOW_SLOT
                .get()
                .and_then(|s| s.lock().ok())
                .and_then(|g| g.clone());
            let (wa, scale) = win32::window_work_area_and_scale(win_opt.as_ref());
            self.scale = scale;
            self.pet.set_environment(wa, scale);
            if self.pet.tracked_window_id().is_none() {
                self.cached_candidate = win32::find_sit_candidate(self.my_pid, wa, scale);
            }
        }

        // Verify the active sit/climb target window on EVERY frame (exists, visible, not minimized,
        // not cloaked, not occluded, current rect) so closed/minimized windows never hold the pet.
        let tracked = self.pet.tracked_window_id().and_then(|id| {
            win32::query_window_candidate(id, self.my_pid, Some(self.pet.sit_anchor_center_x()))
        });
        self.pet.sync_window_targets(tracked, self.cached_candidate);
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

        let mood_override = self.harness.current().cloned();
        let snap = if mood_override.is_some() && !self.drag_active {
            // While reacting to a harness mood, stay attentive in place.
            let mut s = self.pet.snapshot();
            s.walking = false;
            s.walk_speed_factor = 0.0;
            s.facing = 0;
            s
        } else {
            self.pet.step(dt_sec)
        };

        let target_pose = mood_override
            .as_ref()
            .map(|m| m.pose)
            .unwrap_or(snap.pose);
        let target_expr = if self.delight.happy_boost_sec > 0.0 {
            "happy"
        } else {
            mood_override
                .as_ref()
                .map(|m| m.expression)
                .unwrap_or(snap.expression)
        };

        if target_pose != self.pose_name {
            let dur = if target_pose == "wave" { 260.0 } else { 360.0 };
            self.set_pose_target(target_pose, dur);
        }
        if target_expr != self.expr_name {
            self.set_expr_target(target_expr, 220.0);
        }

        let dt_ms = dt_sec * 1000.0;
        self.total_time_sec += dt_sec;

        if self.pose_duration_ms > 0.0 && self.pose_elapsed_ms < self.pose_duration_ms {
            self.pose_elapsed_ms = (self.pose_elapsed_ms + dt_ms).min(self.pose_duration_ms);
            self.pose_current = interpolate_pose(
                &self.pose_from,
                &self.pose_to,
                self.pose_elapsed_ms / self.pose_duration_ms,
            );
        } else {
            self.pose_current = self.pose_to.clone();
        }

        if self.expr_duration_ms > 0.0 && self.expr_elapsed_ms < self.expr_duration_ms {
            self.expr_elapsed_ms = (self.expr_elapsed_ms + dt_ms).min(self.expr_duration_ms);
            let alpha = ease_in_out_cubic(self.expr_elapsed_ms / self.expr_duration_ms);
            for &n in &EXPRESSION_NAMES {
                let a = self.expr_from.get(n).copied().unwrap_or(0.0);
                let b = self.expr_to.get(n).copied().unwrap_or(0.0);
                self.expr_weights.insert(n, a + (b - a) * alpha);
            }
        } else {
            self.expr_weights = self.expr_to.clone();
        }

        // Tie walk animation strictly to actual physical horizontal pixels moved (DPI-scaled)
        // and smoothly blend into/out of idle so starts and stops never snap.
        if snap.walking && snap.dx_px.abs() > 1e-4 {
            self.walk_phase += compute_walk_phase_delta(snap.dx_px, self.scale);
        }
        let blend_target = if snap.walking {
            (0.25 + 0.75 * snap.walk_speed_factor).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let blend_rate = if blend_target > self.walk_blend { 4.5 } else { 3.8 };
        let delta = blend_target - self.walk_blend;
        let max_step = blend_rate * dt_sec;
        if delta.abs() <= max_step {
            self.walk_blend = blend_target;
        } else {
            self.walk_blend += delta.signum() * max_step;
        }
        if !snap.walking && self.walk_blend <= 1e-3 {
            self.walk_blend = 0.0;
            // Anchor next walk cleanly from neutral passing stance.
            self.walk_phase = std::f32::consts::FRAC_PI_2;
        }

        let target_yaw = facing_to_yaw_rad(snap.facing);
        let lerp = (dt_sec * 8.0).min(1.0);
        self.facing_current += (target_yaw - self.facing_current) * lerp;

        self.sync_window_position(snap.x, snap.y);
        self.update_cursor_interactions(dt_sec, snap.walking);

        snap.should_render
    }

    fn update_cursor_interactions(&mut self, dt_sec: f32, walking: bool) {
        if rune::is_headless() {
            return;
        }
        let win_opt = WINDOW_SLOT
            .get()
            .and_then(|s| s.lock().ok())
            .and_then(|g| g.clone());
        let Some(win) = win_opt else { return };

        let cursor_px = win32::cursor_position_px();
        let rect_px = win32::window_rect_px(&win);

        let mut cursor_norm = None;
        let mut cursor_rel_logical = None;
        if let (Some((cx, cy)), Some((wx, wy, ww, wh))) = (cursor_px, rect_px) {
            let scale = self.scale.max(0.5) as f32;
            cursor_rel_logical = Some(((cx - wx) as f32 / scale, (cy - wy) as f32 / scale));
            let nx = (cx - wx) as f32 / ww.max(1) as f32;
            let ny = (cy - wy) as f32 / wh.max(1) as f32;
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
        let _ = self.click_through.apply(cursor_norm, self.drag_active, &mask, |desired| {
            win.set_click_through(desired)
        });

        let head_norm = mask
            .head_center()
            .unwrap_or(glam::Vec2::new(0.5, 0.30));
        let head_center_px = (
            head_norm.x * PET_WINDOW_WIDTH,
            head_norm.y * PET_WINDOW_HEIGHT,
        );
        let sleeping = self
            .active_mood()
            .is_some_and(|m| m.mood == HarnessMood::Sleep);
        let allow_gaze = !walking && !sleeping && !self.drag_active;
        self.delight
            .update(dt_sec, cursor_rel_logical, head_center_px, allow_gaze);
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
        let breathing = compute_idle_breathing(self.total_time_sec);
        let secondary = if self.forced_preview {
            Pose::new()
        } else {
            compute_pose_secondary_motion(self.pose_name, self.total_time_sec)
        };
        let walk_weight = if self.forced_preview {
            if self.walk_blend > 0.0 { 1.0 } else { 0.0 }
        } else if self.active_mood().is_some() {
            0.0
        } else {
            ease_in_out_cubic(self.walk_blend)
        };
        let (walk_offsets, bob_y) = if walk_weight > 1e-4 {
            let (raw_off, raw_bob) = compute_walk_offsets(self.walk_phase);
            (scale_pose(&raw_off, walk_weight), raw_bob * walk_weight)
        } else {
            (Pose::new(), 0.0)
        };
        let breath_weight = 1.0 - walk_weight * 0.65;

        let mut bone_names: BTreeSet<&str> = BTreeSet::new();
        for k in self.pose_current.keys() {
            bone_names.insert(k.as_str());
        }
        for k in breathing.keys() {
            bone_names.insert(k.as_str());
        }
        for k in secondary.keys() {
            bone_names.insert(k.as_str());
        }
        for k in walk_offsets.keys() {
            bone_names.insert(k.as_str());
        }
        for b in [
            "head",
            "leftUpperLeg",
            "rightUpperLeg",
            "leftLowerLeg",
            "rightLowerLeg",
            "leftFoot",
            "rightFoot",
        ] {
            bone_names.insert(b);
        }

        let mut combined = Pose::new();
        for bone in bone_names {
            let base = self.pose_current.get(bone).copied().unwrap_or([0.0; 3]);
            let add = breathing.get(bone).copied().unwrap_or([0.0; 3]);
            let sec = secondary.get(bone).copied().unwrap_or([0.0; 3]);
            let walk = walk_offsets.get(bone).copied().unwrap_or([0.0; 3]);
            let gaze = if bone == "head" && !self.forced_preview {
                [self.delight.head_pitch, self.delight.head_yaw, 0.0]
            } else {
                [0.0; 3]
            };
            combined.insert(
                bone.to_string(),
                [
                    base[0] + add[0] * breath_weight + sec[0] + walk[0] + gaze[0],
                    base[1] + add[1] * breath_weight + sec[1] + walk[1] + gaze[1],
                    base[2] + add[2] * breath_weight + sec[2] + walk[2] + gaze[2],
                ],
            );
        }

        let mut expressions = self.expr_weights.clone();
        let happy = expressions.get("happy").copied().unwrap_or(0.0);
        let eyes_closed = self.active_mood().is_some_and(|m| m.eyes_closed);
        let blink = if eyes_closed {
            1.0
        } else if happy > 0.5 || self.forced_preview {
            0.0
        } else {
            compute_auto_blink(self.total_time_sec, 3.6, 0.16)
        };
        expressions.insert("blink", blink);

        EvaluatedPose {
            pose: combined,
            root_yaw: self.facing_current,
            root_bob_y: bob_y,
            expressions,
        }
    }
}

fn make_tray_icon() -> TrayIcon {
    let size: u32 = 16;
    let mut rgba = vec![0u8; (size * size * 4) as usize];
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - 7.5;
            let dy = y as f32 - 7.5;
            let idx = ((y * size + x) * 4) as usize;
            if (dx * dx + dy * dy).sqrt() <= 6.5 {
                rgba[idx] = 110;
                rgba[idx + 1] = 231;
                rgba[idx + 2] = 183;
                rgba[idx + 3] = 255;
            }
        }
    }
    TrayIcon::rgba(size, size, rgba)
}

fn tray_menu(paused: bool) -> Vec<TrayItem> {
    vec![
        TrayItem::action("pause", if paused { "Retomar (Resume)" } else { "Pausar (Pause)" }),
        TrayItem::Separator,
        TrayItem::action("quit", "Sair (Quit)"),
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
    let spec = TraySpec::new("vitrine-rune-pet", "Vitrine Avatar Pet (Rune)", make_tray_icon())
        .tooltip("Vitrine Avatar Desktop Pet (Rune)")
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
        s.update_cursor_interactions(0.1, false);
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
        .on_drag(|s: &mut PetAppState, d, _tx| {
            let Some((sx, sy)) = d.screen_point() else {
                return;
            };
            let px = sx as f32;
            let py = sy as f32;
            match d.phase {
                Phase::Down => {
                    s.drag_active = true;
                    s.pet.start_drag(px, py);
                }
                Phase::Move => {
                    if s.drag_active {
                        s.pet.move_drag(px, py);
                    }
                }
                Phase::Up => {
                    if s.drag_active {
                        s.drag_active = false;
                        let drop_cand = win32::find_sit_candidate(
                            s.my_pid,
                            s.pet.work_area(),
                            s.pet.scale(),
                        );
                        s.pet.end_drag(drop_cand);
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
            "Vitrine Avatar Pet (Rune)",
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
        visible: true,
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
        })
        .run();

    emit_measurement_report(false);
}

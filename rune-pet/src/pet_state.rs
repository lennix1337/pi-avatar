//! Pure desktop pet state machine, DPI-aware floor height math, and verified window-sit logic.
//!
//! Desktop coordinates (`Rect`, `PetSnapshot::x/y`, drag positions) are in **physical desktop
//! pixels** matching Win32 `MONITORINFO::rcWork`, `DWMWA_EXTENDED_FRAME_BOUNDS`, and Rune
//! `WindowHandle::set_position`. Logical DIP constants are scaled by `scale` (`dpi / 96.0`).

pub const PET_WINDOW_WIDTH: f32 = 220.0;
pub const PET_WINDOW_HEIGHT: f32 = 280.0;
/// Logical DIP distance from the bottom of the 220x280 window up to the standing shoe soles
/// under the `CAMERA_Y = 1.02, CAMERA_Z = 3.85, FOV = 32 deg` projection.
pub const FOOT_BOTTOM_MARGIN_DIP: f32 = 5.8;
/// Logical DIP distance from the bottom of the 220x280 window up to the seated hips/thighs
/// in `sit` pose so the avatar rests flush on a window's top edge with calves dangling naturally.
pub const SEAT_BOTTOM_MARGIN_DIP: f32 = 107.0;
pub const WALK_SPEED_DIP: f32 = 68.0;
pub const CLIMB_SPEED_DIP: f32 = 280.0;
/// Horizontal distance (in logical DIP) travelled per full 2-step walk cycle (matches VRM leg
/// geometry at `HIP_SWING_RAD = 0.30` so planted feet do not slide).
pub const STRIDE_CYCLE_DIP: f32 = 120.0;
pub const WALK_RAMP_SEC: f32 = 0.28;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub const fn right(self) -> i32 {
        self.x.saturating_add(self.width)
    }

    pub const fn bottom(self) -> i32 {
        self.y.saturating_add(self.height)
    }
}

/// Verified snapshot of a real OS window candidate in physical desktop pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCandidate {
    pub id: u64,
    pub rect: Rect,
    pub exists: bool,
    pub visible: bool,
    pub minimized: bool,
    pub cloaked: bool,
    pub occluded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PetMode {
    Idle,
    Walk,
    Sit,
    Wave,
    Climb,
    Cheer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Taskbar,
    Window,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForcedBehavior {
    Auto,
    ForceIdle,
    ForceWalk,
    ForceArmsUp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PetSnapshot {
    pub mode: PetMode,
    pub surface: Surface,
    pub x: i32,
    pub y: i32,
    pub dx_px: f32,
    pub walk_speed_factor: f32,
    pub facing: i8,
    pub pose: &'static str,
    pub expression: &'static str,
    pub walking: bool,
    pub dragging: bool,
    pub paused: bool,
    pub should_render: bool,
}

pub fn sanitized_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.25 {
        scale
    } else {
        1.0
    }
}

pub fn scaled_px(dip: f32, scale: f64) -> i32 {
    (dip as f64 * sanitized_scale(scale)).round() as i32
}

pub fn scaled_avatar_size(scale: f64) -> (i32, i32) {
    (
        scaled_px(PET_WINDOW_WIDTH, scale),
        scaled_px(PET_WINDOW_HEIGHT, scale),
    )
}

/// Real floor Y coordinate (top of the taskbar = bottom of the monitor work area) in physical px.
pub const fn work_area_floor_y(work_area: Rect) -> i32 {
    work_area.bottom()
}

/// Top-left window Y (in physical pixels) that places the standing avatar's shoe soles flush on
/// the real taskbar top (`work_area.bottom()`) at the given DPI `scale`.
pub fn floor_window_y(work_area: Rect, scale: f64) -> i32 {
    let (_, avatar_h) = scaled_avatar_size(scale);
    let foot_margin = scaled_px(FOOT_BOTTOM_MARGIN_DIP, scale);
    (work_area_floor_y(work_area) - avatar_h + foot_margin).max(work_area.y)
}

/// Top-left window Y (in physical pixels) that seats the avatar's hips/thighs flush on a real
/// window's top edge (`target_rect.y`) with calves dangling over the title bar.
pub fn sit_window_y(target_rect: Rect, scale: f64) -> i32 {
    let (_, avatar_h) = scaled_avatar_size(scale);
    let seat_margin = scaled_px(SEAT_BOTTOM_MARGIN_DIP, scale);
    target_rect.y - avatar_h + seat_margin
}

/// Pure validation of a window sit target: must exist, be visible, not minimized, not cloaked,
/// not occluded on its top edge, wide/tall enough, horizontally overlapping the work area, and
/// positioned vertically with headroom above it and clear separation above the taskbar.
pub fn is_valid_sit_target(candidate: Option<WindowCandidate>, work_area: Rect, scale: f64) -> bool {
    let Some(c) = candidate else { return false };
    if !c.exists || !c.visible || c.minimized || c.cloaked || c.occluded {
        return false;
    }
    let (avatar_w, _) = scaled_avatar_size(scale);
    let min_pad = scaled_px(16.0, scale);
    let min_win_h = scaled_px(60.0, scale);
    if c.rect.width < avatar_w + min_pad || c.rect.height < min_win_h {
        return false;
    }
    let overlap_left = c.rect.x.max(work_area.x);
    let overlap_right = c.rect.right().min(work_area.right());
    if overlap_right - overlap_left < avatar_w {
        return false;
    }
    let target_y = sit_window_y(c.rect, scale);
    if target_y < work_area.y + scaled_px(4.0, scale) {
        return false;
    }
    let min_above_floor = scaled_px(60.0, scale);
    if c.rect.y > work_area.bottom() - min_above_floor {
        return false;
    }
    if c.rect.y >= work_area.bottom() || c.rect.bottom() <= work_area.y {
        return false;
    }
    true
}

pub fn facing_to_yaw_rad(facing: i8) -> f32 {
    if facing == 0 {
        0.0
    } else if facing > 0 {
        0.42 * std::f32::consts::PI
    } else {
        -0.42 * std::f32::consts::PI
    }
}

pub fn compute_walk_phase_delta(dx_px: f32, scale: f64) -> f32 {
    let stride_px = (STRIDE_CYCLE_DIP as f64 * sanitized_scale(scale)) as f32;
    (dx_px.abs() / stride_px.max(1.0)) * std::f32::consts::TAU
}

/// Smooth ease-in / ease-out envelope (`0..=1`) for walk velocity so starts and stops never snap.
pub fn walk_ease_factor(elapsed: f32, duration: f32, ramp_sec: f32) -> f32 {
    let ramp = ramp_sec.max(1e-4);
    let smoothstep = |t: f32| {
        let x = t.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    let in_f = smoothstep(elapsed / ramp);
    if duration > 1000.0 {
        return in_f;
    }
    let rem = (duration - elapsed).max(0.0);
    let out_f = smoothstep(rem / ramp);
    in_f.min(out_f)
}

#[derive(Clone, Debug)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
        (v as f32) / ((1u32 << 24) as f32)
    }
}

pub struct PetStateMachine {
    work_area: Rect,
    scale: f64,
    rng: SimpleRng,
    behavior: ForcedBehavior,
    prefer_sit_first: bool,

    mode: PetMode,
    surface: Surface,
    x: f32,
    y: f32,
    last_dx_px: f32,
    walk_speed_factor: f32,
    facing: i8,
    pose: &'static str,
    expression: &'static str,
    state_elapsed: f32,
    state_duration: f32,
    paused: bool,
    dragging: bool,
    drag_offset: (f32, f32),
    climb_target: Option<(f32, f32)>,
    sit_window: Option<WindowCandidate>,
    candidate_window: Option<WindowCandidate>,
}

impl PetStateMachine {
    pub fn new(work_area: Rect, scale: f64, seed: u64) -> Self {
        let scale = sanitized_scale(scale);
        let (avatar_w, _) = scaled_avatar_size(scale);
        let min_x = work_area.x as f32;
        let max_x = (work_area.right() - avatar_w).max(work_area.x) as f32;
        let right_pad = scaled_px(80.0, scale) as f32;
        let init_x = ((work_area.right() - avatar_w) as f32 - right_pad).clamp(min_x, max_x);
        let init_y = floor_window_y(work_area, scale) as f32;

        Self {
            work_area,
            scale,
            rng: SimpleRng::new(seed),
            behavior: ForcedBehavior::Auto,
            prefer_sit_first: false,
            mode: PetMode::Idle,
            surface: Surface::Taskbar,
            x: init_x,
            y: init_y,
            last_dx_px: 0.0,
            walk_speed_factor: 0.0,
            facing: 0,
            pose: "idle",
            expression: "neutral",
            state_elapsed: 0.0,
            state_duration: 1.8,
            paused: false,
            dragging: false,
            drag_offset: (0.0, 0.0),
            climb_target: None,
            sit_window: None,
            candidate_window: None,
        }
    }

    pub fn set_prefer_sit_first(&mut self, prefer: bool) {
        self.prefer_sit_first = prefer;
        if prefer && self.mode == PetMode::Idle {
            self.state_duration = self.state_duration.min(0.4);
        }
    }

    pub fn set_forced_behavior(&mut self, behavior: ForcedBehavior) {
        self.behavior = behavior;
        match behavior {
            ForcedBehavior::Auto => {}
            ForcedBehavior::ForceIdle => {
                self.surface = Surface::Taskbar;
                self.sit_window = None;
                self.y = self.taskbar_y();
                self.climb_target = None;
                self.enter_mode(PetMode::Idle, 99_999.0);
            }
            ForcedBehavior::ForceWalk => {
                self.surface = Surface::Taskbar;
                self.sit_window = None;
                self.y = self.taskbar_y();
                self.climb_target = None;
                self.facing = -1;
                self.enter_mode(PetMode::Walk, 99_999.0);
            }
            ForcedBehavior::ForceArmsUp => {
                self.surface = Surface::Taskbar;
                self.sit_window = None;
                self.y = self.taskbar_y();
                self.climb_target = None;
                self.enter_mode(PetMode::Cheer, 99_999.0);
            }
        }
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn work_area(&self) -> Rect {
        self.work_area
    }

    pub fn tracked_window_id(&self) -> Option<u64> {
        if self.surface == Surface::Window {
            self.sit_window.map(|w| w.id)
        } else {
            None
        }
    }

    pub fn sit_anchor_center_x(&self) -> i32 {
        let (avatar_w, _) = scaled_avatar_size(self.scale);
        let anchor_x = if self.mode == PetMode::Climb {
            self.climb_target.map(|(tx, _)| tx).unwrap_or(self.x)
        } else {
            self.x
        };
        anchor_x.round() as i32 + avatar_w / 2
    }

    fn avatar_w(&self) -> f32 {
        scaled_avatar_size(self.scale).0 as f32
    }

    fn walk_speed(&self) -> f32 {
        (WALK_SPEED_DIP as f64 * self.scale) as f32
    }

    fn climb_speed(&self) -> f32 {
        (CLIMB_SPEED_DIP as f64 * self.scale) as f32
    }

    fn taskbar_y(&self) -> f32 {
        floor_window_y(self.work_area, self.scale) as f32
    }

    fn min_x(&self) -> f32 {
        self.work_area.x as f32
    }

    fn max_x(&self) -> f32 {
        (self.work_area.right() as f32 - self.avatar_w()).max(self.min_x())
    }

    fn window_x_bounds(&self, rect: Rect) -> (f32, f32) {
        let pad = scaled_px(10.0, self.scale) as f32;
        let left = (rect.x as f32 + pad).clamp(self.min_x(), self.max_x());
        let right = ((rect.right() as f32) - self.avatar_w() - pad)
            .clamp(self.min_x(), self.max_x())
            .max(left);
        (left, right)
    }

    fn enter_mode(&mut self, next: PetMode, duration: f32) {
        self.mode = next;
        self.state_elapsed = 0.0;
        self.state_duration = duration;
        if next != PetMode::Walk {
            self.walk_speed_factor = 0.0;
        }

        match next {
            PetMode::Idle => {
                self.pose = "idle";
                self.expression = "neutral";
                self.facing = 0;
            }
            PetMode::Sit => {
                self.pose = "sit";
                self.expression = "relaxed";
                self.facing = 0;
            }
            PetMode::Wave => {
                self.pose = "wave";
                self.expression = "happy";
                self.facing = 0;
            }
            PetMode::Cheer => {
                self.pose = "cheer";
                self.expression = "happy";
                self.facing = 0;
            }
            PetMode::Walk => {
                self.pose = "idle";
                self.expression = "neutral";
                if self.facing == 0 {
                    self.facing = if self.rng.next_f32() < 0.5 { -1 } else { 1 };
                }
            }
            PetMode::Climb => {
                self.pose = "idle";
                self.expression = "happy";
            }
        }
    }

    fn begin_return_to_taskbar(&mut self) {
        self.sit_window = None;
        self.surface = Surface::Taskbar;
        let tx = self.x.clamp(self.min_x(), self.max_x());
        let ty = self.taskbar_y();
        self.facing = 0;
        if (self.y - ty).abs() < 2.0 && (self.x - tx).abs() < 2.0 {
            self.x = tx;
            self.y = ty;
            self.climb_target = None;
            self.enter_mode(PetMode::Idle, 2.5);
        } else {
            self.climb_target = Some((tx, ty));
            self.enter_mode(PetMode::Climb, 4.0);
        }
    }

    fn begin_climb_to_window(&mut self, win: WindowCandidate) {
        let (win_min_x, win_max_x) = self.window_x_bounds(win.rect);
        let target_x = win_min_x + (win_max_x - win_min_x) * self.rng.next_f32();
        let target_y = sit_window_y(win.rect, self.scale) as f32;
        self.surface = Surface::Window;
        self.sit_window = Some(win);
        self.climb_target = Some((target_x, target_y));
        self.facing = if target_x >= self.x { 1 } else { -1 };
        self.enter_mode(PetMode::Climb, 5.0);
    }

    fn pick_next_mode(&mut self) {
        if self.behavior != ForcedBehavior::Auto {
            return;
        }
        if self.surface == Surface::Window {
            // While perched on a verified real window, stay seated or hop back down to the taskbar.
            if !is_valid_sit_target(self.sit_window, self.work_area, self.scale) {
                self.begin_return_to_taskbar();
                return;
            }
            let r = self.rng.next_f32();
            if r < 0.65 {
                let dur = 4.0 + self.rng.next_f32() * 3.0;
                self.enter_mode(PetMode::Sit, dur);
            } else {
                self.begin_return_to_taskbar();
            }
            return;
        }

        // On the taskbar: never enter mid-air `Sit`; prefer calm idle, natural strolls,
        // brief gestures, and climbing only onto verified real visible windows.
        if self.prefer_sit_first
            && let Some(win) = self.candidate_window
            && is_valid_sit_target(Some(win), self.work_area, self.scale)
        {
            self.prefer_sit_first = false;
            self.begin_climb_to_window(win);
            return;
        }

        let r = self.rng.next_f32();
        if r < 0.36 {
            let dur = 3.0 + self.rng.next_f32() * 3.0;
            self.enter_mode(PetMode::Idle, dur);
        } else if r < 0.68 {
            self.facing = if self.rng.next_f32() < 0.5 { -1 } else { 1 };
            let dur = 2.6 + self.rng.next_f32() * 2.6;
            self.enter_mode(PetMode::Walk, dur);
        } else if r < 0.76 {
            let dur = 2.0 + self.rng.next_f32() * 1.4;
            self.enter_mode(PetMode::Wave, dur);
        } else if r < 0.82 {
            let dur = 2.0 + self.rng.next_f32() * 1.4;
            self.enter_mode(PetMode::Cheer, dur);
        } else if let Some(win) = self.candidate_window
            && is_valid_sit_target(Some(win), self.work_area, self.scale)
        {
            self.begin_climb_to_window(win);
        } else {
            let dur = 2.5 + self.rng.next_f32() * 2.5;
            self.enter_mode(PetMode::Idle, dur);
        }
    }

    pub fn set_environment(&mut self, next_work_area: Rect, next_scale: f64) {
        self.work_area = next_work_area;
        self.scale = sanitized_scale(next_scale);
        self.x = self.x.clamp(self.min_x(), self.max_x());
        if self.surface == Surface::Taskbar && !self.dragging && self.mode != PetMode::Climb {
            self.y = self.taskbar_y();
        } else if self.surface == Surface::Taskbar && self.mode == PetMode::Climb {
            if let Some((tx, _)) = self.climb_target {
                self.climb_target = Some((tx.clamp(self.min_x(), self.max_x()), self.taskbar_y()));
            }
        } else {
            self.y = self.y.clamp(self.work_area.y as f32, self.taskbar_y());
        }
    }

    /// Reconciles live per-frame OS window state for both the currently tracked sit/climb target
    /// and the best available climb candidate. Any closed, hidden, minimized, cloaked, occluded,
    /// or off-screen tracked window immediately aborts sitting or climbing back to the taskbar.
    pub fn sync_window_targets(
        &mut self,
        tracked: Option<WindowCandidate>,
        best_candidate: Option<WindowCandidate>,
    ) {
        self.candidate_window =
            best_candidate.filter(|c| is_valid_sit_target(Some(*c), self.work_area, self.scale));

        if self.surface != Surface::Window || self.dragging {
            return;
        }

        let expected_id = self.sit_window.map(|w| w.id);
        let valid_tracked = tracked.filter(|live| {
            expected_id == Some(live.id)
                && is_valid_sit_target(Some(*live), self.work_area, self.scale)
        });

        let Some(live) = valid_tracked else {
            self.begin_return_to_taskbar();
            return;
        };

        let (win_min_x, win_max_x) = self.window_x_bounds(live.rect);
        let target_y = sit_window_y(live.rect, self.scale) as f32;
        if self.mode == PetMode::Climb {
            let next_tx = self
                .climb_target
                .map(|(tx, _)| tx)
                .unwrap_or(self.x)
                .clamp(win_min_x, win_max_x);
            self.climb_target = Some((next_tx, target_y));
        } else {
            if let Some(prev) = self.sit_window {
                let delta_x = (live.rect.x - prev.rect.x) as f32;
                self.x += delta_x;
            }
            self.x = self.x.clamp(win_min_x, win_max_x);
            self.y = target_y;
        }
        self.sit_window = Some(live);
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn on_double_click(&mut self) {
        if self.paused || self.dragging || self.behavior != ForcedBehavior::Auto {
            return;
        }
        if self.surface == Surface::Window {
            return;
        }
        if self.mode == PetMode::Cheer {
            self.enter_mode(PetMode::Wave, 2.2);
        } else {
            self.enter_mode(PetMode::Cheer, 2.6);
        }
    }

    pub fn start_drag(&mut self, screen_x: f32, screen_y: f32) {
        self.dragging = true;
        self.drag_offset = (screen_x - self.x, screen_y - self.y);
        self.surface = Surface::Taskbar;
        self.sit_window = None;
        self.climb_target = None;
        self.enter_mode(PetMode::Cheer, 9999.0);
    }

    pub fn move_drag(&mut self, screen_x: f32, screen_y: f32) {
        if !self.dragging {
            return;
        }
        self.x = (screen_x - self.drag_offset.0).clamp(self.min_x(), self.max_x());
        self.y = (screen_y - self.drag_offset.1).clamp(self.work_area.y as f32, self.taskbar_y());
    }

    pub fn end_drag(&mut self, drop_candidate: Option<WindowCandidate>) {
        if !self.dragging {
            return;
        }
        self.dragging = false;
        let snap_pad_x = scaled_px(20.0, self.scale) as f32;
        let snap_tol_y = scaled_px(44.0, self.scale) as f32;
        if let Some(win) = drop_candidate
            && is_valid_sit_target(Some(win), self.work_area, self.scale)
            && self.x >= win.rect.x as f32 - snap_pad_x
            && self.x <= win.rect.right() as f32 - self.avatar_w() + snap_pad_x
            && (self.y - sit_window_y(win.rect, self.scale) as f32).abs() <= snap_tol_y
        {
            let (win_min_x, win_max_x) = self.window_x_bounds(win.rect);
            self.surface = Surface::Window;
            self.sit_window = Some(win);
            self.x = self.x.clamp(win_min_x, win_max_x);
            self.y = sit_window_y(win.rect, self.scale) as f32;
            self.enter_mode(PetMode::Sit, 4.5);
        } else {
            self.begin_return_to_taskbar();
        }
    }

    pub fn step(&mut self, dt_sec: f32) -> PetSnapshot {
        self.last_dx_px = 0.0;
        if self.paused {
            return self.snapshot();
        }
        let dt = dt_sec.max(0.0);
        if self.dragging {
            return self.snapshot();
        }

        self.state_elapsed += dt;
        let prev_x = self.x;

        if self.mode == PetMode::Walk {
            let left_bound = self.min_x();
            let right_bound = self.max_x();
            let env_ease = walk_ease_factor(self.state_elapsed, self.state_duration, WALK_RAMP_SEC);
            let turn_zone = scaled_px(32.0, self.scale) as f32;
            let dist_to_edge = if self.facing < 0 {
                (self.x - left_bound).max(0.0)
            } else {
                (right_bound - self.x).max(0.0)
            };
            let edge_ease = if dist_to_edge < turn_zone && turn_zone > 1.0 {
                0.35 + 0.65 * (dist_to_edge / turn_zone)
            } else {
                1.0
            };
            let speed_factor = (env_ease * edge_ease).clamp(0.0, 1.0);
            self.walk_speed_factor = speed_factor;
            self.x += (self.facing as f32) * self.walk_speed() * speed_factor * dt;
            if self.x <= left_bound {
                self.x = left_bound;
                self.facing = 1;
            } else if self.x >= right_bound {
                self.x = right_bound;
                self.facing = -1;
            }
            self.last_dx_px = self.x - prev_x;
        } else if self.mode == PetMode::Climb
            && let Some((tx, ty)) = self.climb_target
        {
            let dx = tx - self.x;
            let dy = ty - self.y;
            let dist = dx.hypot(dy);
            let step_dist = self.climb_speed() * dt;
            if dist <= step_dist || dist < 1.0 {
                self.x = tx;
                self.y = ty;
                self.climb_target = None;
                if self.surface == Surface::Window {
                    if is_valid_sit_target(self.sit_window, self.work_area, self.scale) {
                        self.enter_mode(PetMode::Sit, 4.5);
                    } else {
                        self.begin_return_to_taskbar();
                    }
                } else {
                    self.enter_mode(PetMode::Idle, 2.8);
                }
                return self.snapshot();
            }
            self.x += (dx / dist) * step_dist;
            self.y += (dy / dist) * step_dist;
            self.last_dx_px = self.x - prev_x;
        }

        if self.behavior == ForcedBehavior::Auto && self.state_elapsed >= self.state_duration {
            self.pick_next_mode();
        }

        self.snapshot()
    }

    pub fn snapshot(&self) -> PetSnapshot {
        let should_render = !self.paused;
        PetSnapshot {
            mode: self.mode,
            surface: self.surface,
            x: self.x.round() as i32,
            y: self.y.round() as i32,
            dx_px: self.last_dx_px,
            walk_speed_factor: self.walk_speed_factor,
            facing: self.facing,
            pose: self.pose,
            expression: self.expression,
            walking: !self.paused && self.mode == PetMode::Walk,
            dragging: self.dragging,
            paused: self.paused,
            should_render,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORK_100: Rect = Rect {
        x: 0,
        y: 0,
        width: 1920,
        height: 1040,
    };

    fn make_window(id: u64, rect: Rect) -> WindowCandidate {
        WindowCandidate {
            id,
            rect,
            exists: true,
            visible: true,
            minimized: false,
            cloaked: false,
            occluded: false,
        }
    }

    #[test]
    fn floor_window_y_places_shoe_soles_flush_on_taskbar_top_across_dpis_and_monitors() {
        for (work, scale) in [
            (WORK_100, 1.0),
            (Rect { x: 0, y: 0, width: 3440, height: 1392 }, 1.0),
            (Rect { x: 0, y: 0, width: 2560, height: 1380 }, 1.25),
            (Rect { x: 0, y: 0, width: 3840, height: 2088 }, 1.5),
            (Rect { x: 0, y: 0, width: 3840, height: 2064 }, 2.0),
            (Rect { x: 3440, y: -240, width: 1920, height: 1032 }, 1.0),
        ] {
            let win_y = floor_window_y(work, scale);
            let (_, avatar_h) = scaled_avatar_size(scale);
            let foot_margin = scaled_px(FOOT_BOTTOM_MARGIN_DIP, scale);
            let sole_y = win_y + avatar_h - foot_margin;
            assert_eq!(
                sole_y,
                work_area_floor_y(work),
                "sole_y must equal exact taskbar top (work_area.bottom()) for {work:?} @ {scale}x"
            );
            assert!(
                foot_margin > 0,
                "foot_margin must offset transparent sole padding so feet do not float"
            );
        }
    }

    #[test]
    fn sit_window_y_places_seated_hips_flush_on_window_top_edge() {
        let win = Rect { x: 400, y: 500, width: 900, height: 500 };
        for scale in [1.0, 1.25, 1.5, 2.0] {
            let y = sit_window_y(win, scale);
            let (_, avatar_h) = scaled_avatar_size(scale);
            let seat_margin = scaled_px(SEAT_BOTTOM_MARGIN_DIP, scale);
            assert_eq!(y + avatar_h - seat_margin, win.y);
        }
    }

    #[test]
    fn sit_target_validity_accepts_real_visible_window_and_rejects_invalid_states() {
        let valid_rect = Rect { x: 300, y: 420, width: 800, height: 500 };
        let good = make_window(10, valid_rect);
        assert!(is_valid_sit_target(Some(good), WORK_100, 1.0));
        assert!(!is_valid_sit_target(None, WORK_100, 1.0));

        // Closed / non-existent
        assert!(!is_valid_sit_target(Some(WindowCandidate { exists: false, ..good }), WORK_100, 1.0));
        // Invisible
        assert!(!is_valid_sit_target(Some(WindowCandidate { visible: false, ..good }), WORK_100, 1.0));
        // Minimized
        assert!(!is_valid_sit_target(Some(WindowCandidate { minimized: true, ..good }), WORK_100, 1.0));
        // Cloaked (hidden UWP / other virtual desktop)
        assert!(!is_valid_sit_target(Some(WindowCandidate { cloaked: true, ..good }), WORK_100, 1.0));
        // Occluded by another window in Z-order
        assert!(!is_valid_sit_target(Some(WindowCandidate { occluded: true, ..good }), WORK_100, 1.0));
        // Maximized / no headroom above top edge
        let maximized = make_window(11, Rect { x: 0, y: 0, width: 1920, height: 1040 });
        assert!(!is_valid_sit_target(Some(maximized), WORK_100, 1.0));
        // Too narrow for avatar
        let narrow = make_window(12, Rect { x: 400, y: 420, width: 180, height: 400 });
        assert!(!is_valid_sit_target(Some(narrow), WORK_100, 1.0));
        // Too close to taskbar floor
        let too_low = make_window(13, Rect { x: 400, y: 1000, width: 800, height: 200 });
        assert!(!is_valid_sit_target(Some(too_low), WORK_100, 1.0));
        // Off-screen horizontally (e.g., on another monitor)
        let off_screen = make_window(14, Rect { x: 2500, y: 420, width: 800, height: 500 });
        assert!(!is_valid_sit_target(Some(off_screen), WORK_100, 1.0));
    }

    #[test]
    fn taskbar_never_enters_midair_sit_mode() {
        let mut sm = PetStateMachine::new(WORK_100, 1.0, 0x2026_1123);
        for _ in 0..4000 {
            let snap = sm.step(0.1);
            assert!(
                !(snap.surface == Surface::Taskbar && snap.mode == PetMode::Sit),
                "pet must never sit in mid-air on the taskbar"
            );
        }
    }

    #[test]
    fn closing_or_minimizing_window_aborts_both_mid_climb_and_seated_immediately() {
        let win_rect = Rect { x: 300, y: 420, width: 800, height: 500 };
        let live = make_window(77, win_rect);

        // Case 1: window minimizes mid-climb
        let mut sm = PetStateMachine::new(WORK_100, 1.0, 12345);
        sm.set_prefer_sit_first(true);
        sm.sync_window_targets(None, Some(live));
        let mut climbed = false;
        for _ in 0..100 {
            let s = sm.step(0.05);
            if s.surface == Surface::Window && s.mode == PetMode::Climb {
                climbed = true;
                break;
            }
        }
        assert!(climbed, "pet should start climbing toward valid window");
        assert_eq!(sm.tracked_window_id(), Some(77));

        let minimized = WindowCandidate { minimized: true, ..live };
        sm.sync_window_targets(Some(minimized), None);
        let after = sm.step(0.016);
        assert_eq!(after.surface, Surface::Taskbar, "minimizing mid-climb aborts immediately");
        assert_eq!(sm.tracked_window_id(), None);

        // Case 2: reaches sit, then window closes -> returns to taskbar on next frame
        let mut sm2 = PetStateMachine::new(WORK_100, 1.0, 12345);
        sm2.set_prefer_sit_first(true);
        sm2.sync_window_targets(None, Some(live));
        for _ in 0..300 {
            sm2.sync_window_targets(Some(live), Some(live));
            let s = sm2.step(0.05);
            if s.surface == Surface::Window && s.mode == PetMode::Sit {
                break;
            }
        }
        assert_eq!(sm2.snapshot().mode, PetMode::Sit);
        assert_eq!(sm2.snapshot().y, sit_window_y(win_rect, 1.0));

        // Window moves: pet tracks exact new rect on the same frame
        let moved_rect = Rect { x: 420, y: 360, width: 800, height: 500 };
        let moved = make_window(77, moved_rect);
        let prev_x = sm2.snapshot().x;
        sm2.sync_window_targets(Some(moved), Some(moved));
        assert_eq!(sm2.snapshot().y, sit_window_y(moved_rect, 1.0));
        assert_eq!(sm2.snapshot().x, prev_x + 120);

        // Window closes: pet immediately leaves Surface::Window
        let closed = WindowCandidate { exists: false, visible: false, ..moved };
        sm2.sync_window_targets(Some(closed), None);
        assert_eq!(sm2.snapshot().surface, Surface::Taskbar);
        assert_eq!(sm2.tracked_window_id(), None);
        for _ in 0..200 {
            sm2.step(0.05);
        }
        assert_eq!(sm2.snapshot().y, floor_window_y(WORK_100, 1.0));
    }

    #[test]
    fn idle_stays_alive_for_60fps_breathing_and_pause_stops_rendering() {
        let mut sm = PetStateMachine::new(WORK_100, 1.0, 42);
        sm.set_forced_behavior(ForcedBehavior::ForceIdle);
        let snap = sm.step(1.0);
        assert!(snap.should_render, "unpaused idle renders smooth 60fps breathing/gaze");
        assert_eq!(snap.pose, "idle");
        assert_eq!(snap.y, floor_window_y(WORK_100, 1.0));

        sm.set_paused(true);
        assert!(!sm.snapshot().should_render, "paused pet stops rendering");
    }

    #[test]
    fn walk_eases_in_moves_along_taskbar_and_bounces_at_edges() {
        let mut sm = PetStateMachine::new(WORK_100, 1.0, 7);
        sm.set_forced_behavior(ForcedBehavior::ForceWalk);
        let start_x = sm.snapshot().x;
        let s0 = sm.step(0.05);
        assert!(s0.walk_speed_factor < 0.5, "walk must ease in smoothly from rest");
        let s1 = sm.step(0.4);
        assert!(s1.walking);
        assert!(s1.walk_speed_factor > 0.9, "walk reaches full cruise speed after ramp");
        assert!(s1.dx_px.abs() > 10.0, "actual dx_px reported for stride sync");
        assert!(s1.x < start_x, "starts walking left away from right edge");
        assert_eq!(s1.y, floor_window_y(WORK_100, 1.0), "stays on exact taskbar floor");

        for _ in 0..220 {
            sm.step(0.2);
        }
        assert_eq!(sm.snapshot().facing, 1);
        assert!(sm.snapshot().x >= WORK_100.x);
    }

    #[test]
    fn double_click_triggers_cheer_arms_up_pose() {
        let mut sm = PetStateMachine::new(WORK_100, 1.0, 1);
        sm.on_double_click();
        let snap = sm.snapshot();
        assert_eq!(snap.pose, "cheer");
        assert_eq!(snap.expression, "happy");
    }

    #[test]
    fn drag_and_drop_returns_to_taskbar_or_snaps_only_to_valid_window() {
        let mut sm = PetStateMachine::new(WORK_100, 1.0, 1);
        let init = sm.snapshot();
        sm.start_drag(init.x as f32 + 50.0, init.y as f32 + 50.0);
        sm.move_drag(400.0, 200.0);
        assert!(sm.snapshot().dragging);
        // Dropping on a minimized window refuses to sit and returns to taskbar
        let minimized = WindowCandidate {
            minimized: true,
            ..make_window(99, Rect { x: 200, y: 380, width: 800, height: 500 })
        };
        sm.end_drag(Some(minimized));
        assert_eq!(sm.snapshot().surface, Surface::Taskbar);
        assert_eq!(sm.snapshot().mode, PetMode::Climb);
        for _ in 0..80 {
            sm.step(0.1);
        }
        assert_eq!(sm.snapshot().y, floor_window_y(WORK_100, 1.0));
    }
}

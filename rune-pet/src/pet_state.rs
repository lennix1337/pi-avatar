//! Pure desktop pet state machine, window bounds, and taskbar/window climbing logic.

pub const PET_WINDOW_WIDTH: f32 = 220.0;
pub const PET_WINDOW_HEIGHT: f32 = 280.0;
pub const WALK_SPEED_PX: f32 = 76.0;
pub const CLIMB_SPEED_PX: f32 = 260.0;
pub const STRIDE_CYCLE_PX: f32 = 104.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
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
    pub facing: i8,
    pub pose: &'static str,
    pub expression: &'static str,
    pub walking: bool,
    pub dragging: bool,
    pub paused: bool,
    pub should_render: bool,
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

pub fn compute_walk_phase_delta(dx_px: f32) -> f32 {
    (dx_px.abs() / STRIDE_CYCLE_PX) * std::f32::consts::TAU
}

pub fn is_valid_climb_target(fg: Option<Rect>, work_area: Rect, avatar_w: f32, avatar_h: f32) -> bool {
    let Some(fg) = fg else { return false };
    if fg.width < (avatar_w as i32) + 16 || fg.height < 40 {
        return false;
    }
    let target_y = fg.y as f32 - avatar_h;
    if target_y < work_area.y as f32 + 4.0 {
        return false;
    }
    let taskbar_y = (work_area.y + work_area.height) as f32 - avatar_h;
    if target_y >= taskbar_y - 24.0 {
        return false;
    }
    true
}

#[derive(Clone, Debug)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed },
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
    avatar_w: f32,
    avatar_h: f32,
    walk_speed: f32,
    climb_speed: f32,
    rng: SimpleRng,
    behavior: ForcedBehavior,

    mode: PetMode,
    surface: Surface,
    x: f32,
    y: f32,
    last_dx_px: f32,
    facing: i8,
    pose: &'static str,
    expression: &'static str,
    state_elapsed: f32,
    state_duration: f32,
    paused: bool,
    dragging: bool,
    drag_offset: (f32, f32),
    climb_target: Option<(f32, f32)>,
    foreground_rect: Option<Rect>,
}

impl PetStateMachine {
    pub fn new(work_area: Rect, seed: u64) -> Self {
        let avatar_w = PET_WINDOW_WIDTH;
        let avatar_h = PET_WINDOW_HEIGHT;
        let min_x = work_area.x as f32;
        let max_x = ((work_area.x + work_area.width) as f32 - avatar_w).max(min_x);
        let init_x = ((work_area.x + work_area.width) as f32 - avatar_w - 80.0).clamp(min_x, max_x);
        let init_y = (work_area.y + work_area.height) as f32 - avatar_h;

        Self {
            work_area,
            avatar_w,
            avatar_h,
            walk_speed: WALK_SPEED_PX,
            climb_speed: CLIMB_SPEED_PX,
            rng: SimpleRng::new(seed),
            behavior: ForcedBehavior::Auto,
            mode: PetMode::Idle,
            surface: Surface::Taskbar,
            x: init_x,
            y: init_y,
            last_dx_px: 0.0,
            facing: 0,
            pose: "idle",
            expression: "neutral",
            state_elapsed: 0.0,
            state_duration: 2.0,
            paused: false,
            dragging: false,
            drag_offset: (0.0, 0.0),
            climb_target: None,
            foreground_rect: None,
        }
    }

    pub fn set_forced_behavior(&mut self, behavior: ForcedBehavior) {
        self.behavior = behavior;
        match behavior {
            ForcedBehavior::Auto => {}
            ForcedBehavior::ForceIdle => {
                self.surface = Surface::Taskbar;
                self.y = self.taskbar_y();
                self.climb_target = None;
                self.enter_mode(PetMode::Idle, 99_999.0);
            }
            ForcedBehavior::ForceWalk => {
                self.surface = Surface::Taskbar;
                self.y = self.taskbar_y();
                self.climb_target = None;
                self.facing = -1;
                self.enter_mode(PetMode::Walk, 99_999.0);
            }
            ForcedBehavior::ForceArmsUp => {
                self.surface = Surface::Taskbar;
                self.y = self.taskbar_y();
                self.climb_target = None;
                self.enter_mode(PetMode::Cheer, 99_999.0);
            }
        }
    }

    fn taskbar_y(&self) -> f32 {
        (self.work_area.y + self.work_area.height) as f32 - self.avatar_h
    }

    fn min_x(&self) -> f32 {
        self.work_area.x as f32
    }

    fn max_x(&self) -> f32 {
        ((self.work_area.x + self.work_area.width) as f32 - self.avatar_w).max(self.min_x())
    }

    fn enter_mode(&mut self, next: PetMode, duration: f32) {
        self.mode = next;
        self.state_elapsed = 0.0;
        self.state_duration = duration;

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

    fn pick_next_mode(&mut self) {
        if self.behavior != ForcedBehavior::Auto {
            return;
        }
        if self.surface == Surface::Window {
            let r = self.rng.next_f32();
            if r < 0.45 {
                let dur = 3.5 + self.rng.next_f32() * 2.5;
                self.enter_mode(PetMode::Sit, dur);
            } else if r < 0.70 {
                let dur = 2.0 + self.rng.next_f32() * 1.5;
                self.enter_mode(PetMode::Idle, dur);
            } else {
                self.surface = Surface::Taskbar;
                let tx = self.x.clamp(self.min_x(), self.max_x());
                self.climb_target = Some((tx, self.taskbar_y()));
                self.facing = 0;
                self.enter_mode(PetMode::Climb, 4.0);
            }
            return;
        }

        let r = self.rng.next_f32();
        if r < 0.45 {
            self.facing = if self.rng.next_f32() < 0.5 { -1 } else { 1 };
            let dur = 3.0 + self.rng.next_f32() * 3.5;
            self.enter_mode(PetMode::Walk, dur);
        } else if r < 0.65 {
            let dur = 2.2 + self.rng.next_f32() * 2.5;
            self.enter_mode(PetMode::Idle, dur);
        } else if r < 0.78 {
            let dur = 2.0 + self.rng.next_f32() * 1.8;
            self.enter_mode(PetMode::Cheer, dur);
        } else if r < 0.90 {
            let dur = 3.0 + self.rng.next_f32() * 2.5;
            self.enter_mode(PetMode::Sit, dur);
        } else if is_valid_climb_target(
            self.foreground_rect,
            self.work_area,
            self.avatar_w,
            self.avatar_h,
        ) {
            let fg = self.foreground_rect.unwrap();
            let fg_min_x = (fg.x as f32 + 12.0).clamp(self.min_x(), self.max_x());
            let fg_max_x = ((fg.x + fg.width) as f32 - self.avatar_w - 12.0)
                .clamp(self.min_x(), self.max_x());
            let target_x = fg_min_x + (fg_max_x - fg_min_x).max(0.0) * self.rng.next_f32();
            let target_y = fg.y as f32 - self.avatar_h;
            self.surface = Surface::Window;
            self.climb_target = Some((target_x, target_y));
            self.facing = if target_x >= self.x { 1 } else { -1 };
            self.enter_mode(PetMode::Climb, 5.0);
        } else {
            self.facing = if self.rng.next_f32() < 0.5 { -1 } else { 1 };
            let dur = 2.5 + self.rng.next_f32() * 2.5;
            self.enter_mode(PetMode::Walk, dur);
        }
    }

    pub fn set_work_area(&mut self, next: Rect) {
        self.work_area = next;
        self.x = self.x.clamp(self.min_x(), self.max_x());
        if self.surface == Surface::Taskbar && !self.dragging && self.mode != PetMode::Climb {
            self.y = self.taskbar_y();
        } else {
            self.y = self.y.clamp(self.work_area.y as f32, self.taskbar_y());
        }
    }

    pub fn set_foreground_rect(&mut self, rect: Option<Rect>) {
        self.foreground_rect = rect;
        if self.surface == Surface::Window && !self.dragging && self.mode != PetMode::Climb {
            if !is_valid_climb_target(
                self.foreground_rect,
                self.work_area,
                self.avatar_w,
                self.avatar_h,
            ) {
                self.surface = Surface::Taskbar;
                let tx = self.x.clamp(self.min_x(), self.max_x());
                self.climb_target = Some((tx, self.taskbar_y()));
                self.enter_mode(PetMode::Climb, 4.0);
            } else if let Some(fg) = self.foreground_rect {
                self.y = fg.y as f32 - self.avatar_h;
                self.x = self.x.clamp(
                    fg.x as f32,
                    ((fg.x + fg.width) as f32 - self.avatar_w).max(fg.x as f32),
                );
            }
        }
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

    pub fn end_drag(&mut self) {
        if !self.dragging {
            return;
        }
        self.dragging = false;
        if let Some(fg) = self.foreground_rect
            && is_valid_climb_target(Some(fg), self.work_area, self.avatar_w, self.avatar_h)
            && self.x >= fg.x as f32 - 20.0
            && self.x <= (fg.x + fg.width) as f32 - self.avatar_w + 20.0
            && (self.y - (fg.y as f32 - self.avatar_h)).abs() <= 36.0
        {
            self.surface = Surface::Window;
            self.y = fg.y as f32 - self.avatar_h;
            self.enter_mode(PetMode::Sit, 3.5);
        } else {
            self.surface = Surface::Taskbar;
            let tx = self.x.clamp(self.min_x(), self.max_x());
            self.climb_target = Some((tx, self.taskbar_y()));
            self.enter_mode(PetMode::Climb, 4.0);
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
            let mut left_bound = self.min_x();
            let mut right_bound = self.max_x();
            if self.surface == Surface::Window
                && is_valid_climb_target(
                    self.foreground_rect,
                    self.work_area,
                    self.avatar_w,
                    self.avatar_h,
                )
            {
                let fg = self.foreground_rect.unwrap();
                left_bound = (fg.x as f32).clamp(self.min_x(), self.max_x());
                right_bound = ((fg.x + fg.width) as f32 - self.avatar_w)
                    .clamp(self.min_x(), self.max_x());
            }
            self.x += (self.facing as f32) * self.walk_speed * dt;
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
            let step_dist = self.climb_speed * dt;
            if dist <= step_dist || dist < 1.0 {
                self.x = tx;
                self.y = ty;
                self.climb_target = None;
                let next = if self.surface == Surface::Window {
                    PetMode::Sit
                } else {
                    PetMode::Idle
                };
                self.enter_mode(next, 2.8);
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
        // Keep continuous 60 FPS animation (breathing, auto-blink, gaze, sparkles) whenever unpaused.
        let should_render = !self.paused;
        PetSnapshot {
            mode: self.mode,
            surface: self.surface,
            x: self.x.round() as i32,
            y: self.y.round() as i32,
            dx_px: self.last_dx_px,
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

    const WORK: Rect = Rect {
        x: 0,
        y: 0,
        width: 1920,
        height: 1040,
    };

    #[test]
    fn idle_stays_alive_for_60fps_breathing_and_pause_stops_rendering() {
        let mut sm = PetStateMachine::new(WORK, 42);
        sm.set_forced_behavior(ForcedBehavior::ForceIdle);
        let snap = sm.step(1.0);
        assert!(snap.should_render, "unpaused idle renders smooth 60fps breathing/gaze");
        assert_eq!(snap.pose, "idle");
        assert_eq!(snap.y, 1040 - PET_WINDOW_HEIGHT as i32);

        sm.set_paused(true);
        assert!(!sm.snapshot().should_render, "paused pet stops rendering");
    }

    #[test]
    fn walk_moves_along_taskbar_and_bounces_at_edges() {
        let mut sm = PetStateMachine::new(WORK, 7);
        sm.set_forced_behavior(ForcedBehavior::ForceWalk);
        let start_x = sm.snapshot().x;
        let s1 = sm.step(0.5);
        assert!(s1.walking);
        assert!(s1.should_render);
        assert!(s1.dx_px.abs() > 10.0, "actual dx_px reported for stride sync");
        assert!(s1.x < start_x, "starts walking left away from right edge");
        assert_eq!(s1.y, 1040 - PET_WINDOW_HEIGHT as i32, "stays on taskbar line");
        // Walk all the way to left edge and verify bounce back right
        for _ in 0..160 {
            sm.step(0.2);
        }
        assert_eq!(sm.snapshot().facing, 1);
        assert!(sm.snapshot().x >= WORK.x);
    }

    #[test]
    fn double_click_triggers_cheer_arms_up_pose() {
        let mut sm = PetStateMachine::new(WORK, 1);
        sm.on_double_click();
        let snap = sm.snapshot();
        assert_eq!(snap.pose, "cheer");
        assert_eq!(snap.expression, "happy");
    }

    #[test]
    fn drag_and_drop_returns_to_taskbar() {
        let mut sm = PetStateMachine::new(WORK, 1);
        let init = sm.snapshot();
        sm.start_drag(init.x as f32 + 50.0, init.y as f32 + 50.0);
        sm.move_drag(400.0, 200.0);
        let mid = sm.snapshot();
        assert!(mid.dragging);
        assert!(mid.y < init.y);
        sm.end_drag();
        assert_eq!(sm.snapshot().mode, PetMode::Climb);
        for _ in 0..60 {
            sm.step(0.1);
        }
        assert_eq!(sm.snapshot().y, 1040 - PET_WINDOW_HEIGHT as i32);
    }
}

//! Delightful touches: subtle cursor-tracking head gaze and interactive head-pat sparkle burst.

#[derive(Clone, Copy, Debug)]
pub struct Sparkle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub age: f32,
    pub lifetime: f32,
    pub size: f32,
    pub warm: bool,
}

#[derive(Default)]
pub struct DelightController {
    pub head_yaw: f32,
    pub head_pitch: f32,
    pub neck_yaw: f32,
    pub neck_pitch: f32,
    pub chest_yaw: f32,
    pub cursor_proximity: f32,
    pub petting_intensity: f32,
    pub happy_boost_sec: f32,
    pub sparkles: Vec<Sparkle>,
    last_pat_x: Option<f32>,
    last_pet_pos: Option<(f32, f32)>,
    last_pat_dir: i8,
    pat_flips: u8,
    pat_window_sec: f32,
    pat_cooldown_sec: f32,
    pet_travel_dip: f32,
    pet_idle_sec: f32,
}

impl DelightController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates cursor-tracking gaze (head + neck + subtle chest), slow-stroke petting ("carinho")
    /// and head-pat oscillation detection, and floating sparkles.
    /// `cursor_rel_px` is cursor position relative to the window's top-left in logical points.
    pub fn update(
        &mut self,
        dt_sec: f32,
        cursor_rel_px: Option<(f32, f32)>,
        head_center_px: (f32, f32),
        allow_gaze: bool,
    ) -> bool {
        let dt = dt_sec.clamp(0.0, 0.2);

        if self.happy_boost_sec > 0.0 {
            self.happy_boost_sec = (self.happy_boost_sec - dt).max(0.0);
        }
        if self.pat_cooldown_sec > 0.0 {
            self.pat_cooldown_sec = (self.pat_cooldown_sec - dt).max(0.0);
        }

        // Advance floating sparkle particles
        for sp in &mut self.sparkles {
            sp.age += dt;
            sp.x += sp.vx * dt + (sp.age * 5.0).sin() * 8.0 * dt;
            sp.y += sp.vy * dt;
        }
        self.sparkles.retain(|sp| sp.age < sp.lifetime);

        // Compute damped gaze target toward nearby cursor (split across head, neck, subtle chest)
        let (target_yaw, target_pitch, target_prox) = if allow_gaze
            && let Some((cx, cy)) = cursor_rel_px
        {
            let dx = cx - head_center_px.0;
            let dy = cy - head_center_px.1;
            let dist = dx.hypot(dy);
            if dist <= 420.0 {
                let falloff = (1.0 - dist / 420.0).clamp(0.0, 1.0);
                let prox = (1.0 - dist / 240.0).clamp(0.0, 1.0);
                let yaw = ((dx / 240.0).clamp(-1.0, 1.0) * 0.42) * (0.45 + 0.55 * falloff);
                let pitch = ((dy / 240.0).clamp(-1.0, 1.0) * 0.24) * (0.45 + 0.55 * falloff);
                (yaw, pitch, prox)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0)
        };

        let lerp = (dt * 7.5).min(1.0);
        let target_head_yaw = target_yaw * 0.68;
        let target_head_pitch = target_pitch * 0.68 - self.petting_intensity * 0.07;
        let target_neck_yaw = target_yaw * 0.32;
        let target_neck_pitch = target_pitch * 0.32;
        let target_chest_yaw = target_yaw * 0.16 * target_prox;

        self.head_yaw += (target_head_yaw - self.head_yaw) * lerp;
        self.head_pitch += (target_head_pitch - self.head_pitch) * lerp;
        self.neck_yaw += (target_neck_yaw - self.neck_yaw) * (dt * 6.0).min(1.0);
        self.neck_pitch += (target_neck_pitch - self.neck_pitch) * (dt * 6.0).min(1.0);
        self.chest_yaw += (target_chest_yaw - self.chest_yaw) * (dt * 4.5).min(1.0);
        self.cursor_proximity += (target_prox - self.cursor_proximity) * (dt * 5.0).min(1.0);

        // Detect gentle slow-stroke petting ("carinho") and head-pat oscillations over head region.
        // Fast accidental cursor swipes (> 480 DIP/s) are ignored.
        let mut pat_triggered = false;
        let mut inside_head = false;
        if allow_gaze && let Some((cx, cy)) = cursor_rel_px {
            let dx = cx - head_center_px.0;
            let dy = cy - head_center_px.1;
            if dx.abs() <= 48.0 && dy.abs() <= 42.0 {
                inside_head = true;
                self.pat_window_sec += dt;
                if self.pat_window_sec > 2.0 {
                    self.pat_window_sec = 0.0;
                    self.pat_flips = 0;
                    self.last_pat_dir = 0;
                }

                let mut gentle_motion = false;
                let mut fast_swipe = false;
                if let Some((px, py)) = self.last_pet_pos {
                    let step_dist = (cx - px).hypot(cy - py);
                    let speed = if dt > 1e-4 { step_dist / dt } else { 0.0 };
                    if speed > 480.0 {
                        fast_swipe = true;
                        self.pet_travel_dip = 0.0;
                        self.pat_flips = 0;
                        self.last_pat_dir = 0;
                        self.petting_intensity = (self.petting_intensity - dt * 4.0).max(0.0);
                    } else if step_dist >= 0.8 && (10.0..=360.0).contains(&speed) {
                        gentle_motion = true;
                        self.pet_idle_sec = 0.0;
                        self.pet_travel_dip += step_dist;
                        self.petting_intensity = (self.petting_intensity + dt * 2.2).min(1.0);
                    }
                }
                self.last_pet_pos = Some((cx, cy));

                if !fast_swipe {
                    if let Some(prev_x) = self.last_pat_x {
                        let delta = cx - prev_x;
                        if delta.abs() >= 6.0 {
                            let dir = if delta > 0.0 { 1 } else { -1 };
                            if self.last_pat_dir != 0 && dir != self.last_pat_dir {
                                self.pat_flips = self.pat_flips.saturating_add(1);
                            }
                            self.last_pat_dir = dir;
                            self.last_pat_x = Some(cx);
                        }
                    } else {
                        self.last_pat_x = Some(cx);
                    }
                }

                if !gentle_motion {
                    self.pet_idle_sec += dt;
                    if self.pet_idle_sec > 0.35 {
                        self.petting_intensity = (self.petting_intensity - dt * 2.2).max(0.0);
                        self.pet_travel_dip = (self.pet_travel_dip - dt * 40.0).max(0.0);
                    }
                }

                if self.pat_cooldown_sec <= 0.0
                    && !fast_swipe
                    && (self.pet_travel_dip >= 80.0 || self.pat_flips >= 3)
                {
                    self.pat_flips = 0;
                    self.pet_travel_dip = 0.0;
                    self.pat_window_sec = 0.0;
                    self.last_pat_dir = 0;
                    self.pat_cooldown_sec = 2.0;
                    self.trigger_pat_burst(head_center_px.0, head_center_px.1 - 24.0);
                    pat_triggered = true;
                }
            }
        }

        if !inside_head {
            self.last_pat_x = None;
            self.last_pet_pos = None;
            self.last_pat_dir = 0;
            self.pat_flips = 0;
            self.pat_window_sec = 0.0;
            self.pet_idle_sec = 0.0;
            self.pet_travel_dip = (self.pet_travel_dip - dt * 50.0).max(0.0);
            self.petting_intensity = (self.petting_intensity - dt * 2.8).max(0.0);
        }

        pat_triggered
    }

    pub fn trigger_click_sparkles(&mut self, origin_x: f32, origin_y: f32) {
        self.happy_boost_sec = self.happy_boost_sec.max(1.0);
        const OFFSETS: &[(f32, f32, f32, f32, f32, bool)] = &[
            (-14.0, -6.0, -12.0, -32.0, 6.5, true),
            (0.0, -14.0, 0.0, -38.0, 7.5, false),
            (14.0, -6.0, 12.0, -32.0, 6.5, true),
        ];
        for &(ox, oy, vx, vy, size, warm) in OFFSETS {
            self.sparkles.push(Sparkle {
                x: origin_x + ox,
                y: origin_y + oy,
                vx,
                vy,
                age: 0.0,
                lifetime: 1.05,
                size,
                warm,
            });
        }
    }

    pub fn trigger_pat_burst(&mut self, origin_x: f32, origin_y: f32) {
        self.happy_boost_sec = 1.9;
        const OFFSETS: &[(f32, f32, f32, f32, f32, bool)] = &[
            (-24.0, -4.0, -10.0, -34.0, 7.0, true),
            (-10.0, -12.0, -4.0, -42.0, 8.5, false),
            (6.0, -14.0, 5.0, -40.0, 7.5, true),
            (22.0, -6.0, 12.0, -32.0, 6.5, false),
            (-16.0, 4.0, -14.0, -28.0, 6.0, false),
            (15.0, 2.0, 14.0, -30.0, 7.0, true),
        ];
        for &(ox, oy, vx, vy, size, warm) in OFFSETS {
            self.sparkles.push(Sparkle {
                x: origin_x + ox,
                y: origin_y + oy,
                vx,
                vy,
                age: 0.0,
                lifetime: 1.45,
                size,
                warm,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaze_tracks_nearby_cursor_and_clamps_to_natural_angles() {
        let mut d = DelightController::new();
        let head = (110.0, 90.0);
        for _ in 0..40 {
            d.update(0.016, Some((260.0, 40.0)), head, true);
        }
        assert!(d.head_yaw > 0.05 && d.head_yaw <= 0.36, "yaw={}", d.head_yaw);
        assert!(d.head_pitch.abs() <= 0.24, "pitch={}", d.head_pitch);

        for _ in 0..80 {
            d.update(0.016, Some((900.0, 900.0)), head, true);
        }
        assert!(d.head_yaw.abs() < 0.05, "gaze should relax: yaw={}", d.head_yaw);
    }

    #[test]
    fn head_pat_oscillation_triggers_happy_burst_and_sparkles_expire() {
        let mut d = DelightController::new();
        let head = (110.0, 90.0);
        let xs = [95.0, 125.0, 95.0, 125.0, 95.0];
        let mut triggered = false;
        for &x in &xs {
            if d.update(0.12, Some((x, 85.0)), head, true) {
                triggered = true;
            }
        }
        assert!(triggered, "petting head left-right should trigger happy burst");
        assert!(d.happy_boost_sec > 0.5);
        assert!(!d.sparkles.is_empty());

        for _ in 0..140 {
            d.update(0.016, None, head, true);
        }
        assert!(d.sparkles.is_empty(), "sparkles must expire cleanly");
        assert_eq!(d.happy_boost_sec, 0.0);
    }

    #[test]
    fn slow_cursor_stroke_over_head_builds_petting_and_triggers_carinho_while_fast_swipe_does_not() {
        let mut d = DelightController::new();
        let head = (110.0, 90.0);

        // Fast swipe across head (~1200 DIP/s) must NOT trigger carinho
        let fast_xs = [70.0, 90.0, 110.0, 130.0, 150.0];
        let mut fast_triggered = false;
        for &x in &fast_xs {
            if d.update(0.016, Some((x, 86.0)), head, true) {
                fast_triggered = true;
            }
        }
        assert!(!fast_triggered, "fast accidental cursor swipe must not trigger petting");
        assert!(d.petting_intensity < 0.2, "fast swipe must not build petting intensity");

        // Slow gentle stroke across head (~90 DIP/s) builds petting intensity and triggers burst
        let mut d2 = DelightController::new();
        let mut slow_triggered = false;
        let mut max_intensity = 0.0f32;
        for i in 0..45 {
            // Smooth gentle sweep back and forth at ~85 DIP/s inside head region
            let x = 86.0 + ((i as f32 * 0.22).sin() * 0.5 + 0.5) * 48.0;
            if d2.update(0.05, Some((x, 84.0)), head, true) {
                slow_triggered = true;
            }
            max_intensity = max_intensity.max(d2.petting_intensity);
        }
        assert!(max_intensity > 0.35, "slow strokes should raise petting intensity, got {max_intensity}");
        assert!(slow_triggered, "slow strokes over head should trigger carinho burst");
        assert!(d2.happy_boost_sec > 0.5);
    }

    #[test]
    fn approaching_cursor_drives_neck_chest_gaze_and_proximity() {
        let mut d = DelightController::new();
        let head = (110.0, 90.0);
        for _ in 0..45 {
            d.update(0.016, Some((180.0, 75.0)), head, true);
        }
        assert!(d.cursor_proximity > 0.5, "close cursor should raise proximity: {}", d.cursor_proximity);
        assert!(d.neck_yaw > 0.01 && d.neck_yaw < d.head_yaw, "neck should share gaze naturally");
        assert!(d.chest_yaw > 0.005, "chest should subtly lean toward close cursor");
    }
}

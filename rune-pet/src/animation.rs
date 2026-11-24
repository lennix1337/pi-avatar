//! Upper-body accents and eyelids. No root motion or leg offsets live in this layer.

use crate::harness::HarnessMood;

pub(crate) fn smooth_ramp(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn gesture(age: f32, start: f32, attack: f32, release: f32) -> f32 {
    let t = age - start;
    smooth_ramp(t / attack) * smooth_ramp((attack + release - t) / release)
}

#[derive(Clone, Copy, Default)]
struct Accent {
    head: [f32; 3],
    chest_pitch: f32,
    shoulder_lift: f32,
}

impl Accent {
    fn follow(&mut self, target: Self, amount: f32) {
        for (angle, goal) in self.head.iter_mut().zip(target.head) {
            *angle += (goal - *angle) * amount;
        }
        self.chest_pitch += (target.chest_pitch - self.chest_pitch) * amount;
        self.shoulder_lift += (target.shoulder_lift - self.shoulder_lift) * amount;
    }
}

fn idle_accent(age: f32, side: f32) -> Accent {
    let glance = gesture(age, 0.9, 0.44, 0.88) - gesture(age, 7.0, 0.50, 1.05);
    let sigh = gesture(age, 3.7, 0.80, 1.48);
    Accent {
        head: [sigh * 0.058, glance * side * 0.13, glance * side * 0.080],
        chest_pitch: -sigh * 0.045,
        shoulder_lift: sigh * 0.042,
    }
}

fn mood_accent(mood: Option<HarnessMood>, age: f32) -> Accent {
    let beat = gesture(age, 0.12, 0.28, 0.47);
    match mood {
        Some(HarnessMood::Worried) => Accent {
            head: [0.115 * beat, 0.0, -0.100 * beat],
            chest_pitch: 0.035 * beat,
            shoulder_lift: 0.045 * beat,
        },
        Some(HarnessMood::Celebrate) => Accent {
            head: [-0.14 * beat, 0.0, 0.085 * beat],
            chest_pitch: -0.060 * beat,
            shoulder_lift: 0.075 * beat,
        },
        Some(HarnessMood::Working | HarnessMood::Wave) => {
            let nod = beat + 0.65 * gesture(age, 0.72, 0.21, 0.31);
            Accent { head: [0.13 * nod, 0.0, 0.035 * beat], ..Accent::default() }
        }
        _ => Accent::default(),
    }
}

// Asymmetric closure: quick downstroke, a very short hold, and slower reopening. A few
// irregular intervals and an occasional double blink avoid a fixed metronome rhythm.
const BLINK_GAPS: [f32; 9] = [3.3, 4.7, 2.6, 0.36, 5.2, 3.9, 2.85, 4.4, 5.55];
const BLINK_CLOSE: f32 = 0.075;
const BLINK_HOLD: f32 = 0.090;
const BLINK_END: f32 = 0.205;

pub(crate) struct AnimationLife {
    idle_age: f32,
    idle_mirrored: bool,
    mood: Option<HarnessMood>,
    mood_age: f32,
    blink_age: f32,
    blink_index: usize,
    accent: Accent,
}

impl Default for AnimationLife {
    fn default() -> Self {
        Self {
            idle_age: 0.0,
            idle_mirrored: false,
            mood: None,
            mood_age: 0.0,
            blink_age: BLINK_END,
            blink_index: 0,
            accent: Accent::default(),
        }
    }
}

impl AnimationLife {
    pub(crate) fn step(&mut self, dt: f32, idle: bool, mood: Option<HarnessMood>, eyes_closed: bool) {
        if mood != self.mood {
            self.mood = mood;
            self.mood_age = 0.0;
        }
        self.mood_age = (self.mood_age + dt).min(2.0);
        if eyes_closed {
            // Finish a blink already in flight while the sleep lid blend takes over.
            self.blink_age = (self.blink_age + dt).min(BLINK_END);
        } else {
            self.blink_age += dt;
            let gap = BLINK_GAPS[self.blink_index];
            if self.blink_age >= gap {
                self.blink_age -= gap;
                self.blink_index = (self.blink_index + 1) % BLINK_GAPS.len();
            }
        }
        let target = if idle && !eyes_closed {
            self.idle_age += dt;
            if self.idle_age >= 12.0 {
                self.idle_age -= 12.0;
                self.idle_mirrored = !self.idle_mirrored;
            }
            idle_accent(self.idle_age, if self.idle_mirrored { -1.0 } else { 1.0 })
        } else {
            mood_accent(mood, self.mood_age)
        };
        self.accent.follow(target, 1.0 - (-dt * 13.0).exp());
    }

    pub(crate) fn blink(&self) -> f32 {
        if self.blink_age >= BLINK_END {
            return 0.0;
        }
        let close = smooth_ramp(self.blink_age / BLINK_CLOSE);
        let open = 1.0 - smooth_ramp((self.blink_age - BLINK_HOLD) / (BLINK_END - BLINK_HOLD));
        close * open
    }

    pub(crate) fn offset(&self, bone: &str) -> [f32; 3] {
        match bone {
            "head" => self.accent.head,
            "chest" => [self.accent.chest_pitch, 0.0, 0.0],
            "leftShoulder" => [0.0, 0.0, self.accent.shoulder_lift],
            "rightShoulder" => [0.0, 0.0, -self.accent.shoulder_lift],
            _ => [0.0; 3],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blink_has_irregular_intervals_a_double_and_quieter_reopening() {
        let mut life = AnimationLife::default();
        let mut starts = Vec::new();
        let mut last = 0.0;
        let mut speeds: [f32; 2] = [0.0; 2];
        for frame in 1..1800 {
            life.step(0.01, true, None, false);
            let blink = life.blink();
            assert!((0.0..=1.0).contains(&blink));
            if last == 0.0 && blink > 0.0 {
                starts.push(frame);
            }
            speeds[0] = speeds[0].max(blink - last);
            speeds[1] = speeds[1].max(last - blink);
            last = blink;
        }
        let gaps: Vec<_> = starts.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().any(|&gap| gap < 50));
        assert!(gaps.iter().any(|&gap| gap > 400));
        assert!(speeds[0] > speeds[1]);
        life.step(0.016, false, Some(HarnessMood::Sleep), true);
        assert_eq!(life.blink(), 0.0);
    }

    #[test]
    fn mood_beats_are_distinct_and_settle_without_repeating() {
        let mut life = AnimationLife::default();
        for _ in 0..30 {
            life.step(0.016, false, Some(HarnessMood::Worried), false);
        }
        let worried = life.offset("head")[0];
        assert!(worried > 0.08);
        life.step(0.016, false, Some(HarnessMood::Celebrate), false);
        assert!(life.offset("head")[0] > worried * 0.7, "no snap on interruption");
        for _ in 0..30 {
            life.step(0.016, false, Some(HarnessMood::Celebrate), false);
        }
        assert!(life.offset("head")[0] < -0.08);
        for _ in 0..250 {
            life.step(0.016, false, Some(HarnessMood::Celebrate), false);
        }
        assert!(life.offset("head")[0].abs() < 0.001);
        for _ in 0..100 {
            life.step(0.016, false, Some(HarnessMood::Celebrate), false);
            assert!(life.offset("head")[0].abs() < 0.001);
        }
    }

    #[test]
    fn idle_glances_mirror_and_never_move_hips_or_feet() {
        let mut life = AnimationLife::default();
        for _ in 0..90 {
            life.step(0.016, true, None, false);
        }
        let first = life.offset("head")[1];
        assert!(first > 0.08);
        for _ in 0..750 {
            life.step(0.016, true, None, false);
        }
        assert!((life.offset("head")[1] + first).abs() < 0.001);
        assert_eq!(life.offset("hips"), [0.0; 3]);
        assert_eq!(life.offset("leftFoot"), [0.0; 3]);
        for _ in 0..40 {
            life.step(0.016, false, None, false);
        }
        assert!(life.offset("head")[1].abs() < 0.001);
    }
}

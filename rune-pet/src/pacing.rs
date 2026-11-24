//! Deterministic 60 FPS frame pacer and FPS statistics calculator (`avg`, `min`, `p5`).

use std::time::Duration;

pub const TARGET_FPS: f64 = 60.0;
pub const TARGET_FRAME_SEC: f64 = 1.0 / TARGET_FPS;
/// Tolerance so Windows timers firing ~1.8 ms early (e.g. 15.625 ms tick) render immediately
/// instead of sleeping another timer quantum and halving FPS.
pub const JITTER_TOLERANCE_SEC: f64 = 0.0018;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PaceDecision {
    Render { dt_sec: f32, next_wake: Duration },
    Wait { next_wake: Duration },
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FpsSummary {
    pub frames: u64,
    pub duration_sec: f64,
    pub avg_fps: f64,
    pub min_fps: f64,
    pub p5_fps: f64,
}

pub struct FramePacer {
    target_interval: f64,
    last_render_sec: Option<f64>,
    next_deadline_sec: Option<f64>,
}

impl FramePacer {
    pub fn new(target_fps: f64) -> Self {
        Self {
            target_interval: 1.0 / target_fps.max(1.0),
            last_render_sec: None,
            next_deadline_sec: None,
        }
    }

    pub fn reset(&mut self) {
        self.last_render_sec = None;
        self.next_deadline_sec = None;
    }

    pub fn tick(&mut self, now_sec: f64) -> PaceDecision {
        let Some(last) = self.last_render_sec else {
            self.last_render_sec = Some(now_sec);
            self.next_deadline_sec = Some(now_sec + self.target_interval);
            return PaceDecision::Render {
                dt_sec: self.target_interval as f32,
                next_wake: Duration::from_secs_f64(self.target_interval),
            };
        };

        let deadline = self.next_deadline_sec.unwrap_or(last + self.target_interval);
        let remaining = deadline - now_sec;
        if remaining > JITTER_TOLERANCE_SEC {
            return PaceDecision::Wait {
                next_wake: Duration::from_secs_f64(remaining.max(0.001)),
            };
        }

        let dt = (now_sec - last).clamp(0.0005, 0.1);
        self.last_render_sec = Some(now_sec);

        // Advance deadline smoothly; if we fell behind by more than 1 frame, re-anchor to `now_sec`
        // so lag spikes never cause a catch-up burst.
        // A frame rendered inside the early-jitter window anchors to `now_sec`; otherwise the
        // deadline would drift ahead of coarse (15.6 ms) timer ticks and drop one frame in ~16.
        let mut next = if now_sec < deadline {
            now_sec + self.target_interval
        } else {
            deadline + self.target_interval
        };
        if next <= now_sec + JITTER_TOLERANCE_SEC {
            next = now_sec + self.target_interval;
        }
        self.next_deadline_sec = Some(next);
        let wake_sec = (next - now_sec).clamp(0.001, self.target_interval);
        PaceDecision::Render {
            dt_sec: dt as f32,
            next_wake: Duration::from_secs_f64(wake_sec),
        }
    }
}

pub struct FpsTracker {
    warmup_sec: f64,
    start_sec: Option<f64>,
    measure_start_sec: Option<f64>,
    last_frame_sec: Option<f64>,
    dt_samples_sec: Vec<f64>,
    frames: u64,
}

impl FpsTracker {
    pub fn new(warmup_sec: f64) -> Self {
        Self {
            warmup_sec: warmup_sec.max(0.0),
            start_sec: None,
            measure_start_sec: None,
            last_frame_sec: None,
            dt_samples_sec: Vec::with_capacity(2048),
            frames: 0,
        }
    }

    pub fn record_frame(&mut self, now_sec: f64) {
        let start = *self.start_sec.get_or_insert(now_sec);
        if now_sec - start < self.warmup_sec {
            return;
        }
        if self.measure_start_sec.is_none() {
            self.measure_start_sec = Some(now_sec);
            self.last_frame_sec = Some(now_sec);
            self.frames = 1;
            return;
        }
        if let Some(prev) = self.last_frame_sec {
            let dt = now_sec - prev;
            if dt > 1e-6 {
                self.dt_samples_sec.push(dt);
            }
        }
        self.last_frame_sec = Some(now_sec);
        self.frames += 1;
    }

    pub fn summary(&self) -> FpsSummary {
        let (Some(a), Some(b)) = (self.measure_start_sec, self.last_frame_sec) else {
            return FpsSummary::default();
        };
        let duration_sec = (b - a).max(0.0);
        if duration_sec <= 1e-6 || self.dt_samples_sec.is_empty() {
            return FpsSummary {
                frames: self.frames,
                duration_sec,
                ..Default::default()
            };
        }
        let avg_fps = (self.frames.saturating_sub(1)) as f64 / duration_sec;
        let mut sorted = self.dt_samples_sec.clone();
        sorted.sort_by(|x, y| x.total_cmp(y));
        let max_dt = *sorted.last().unwrap_or(&TARGET_FRAME_SEC);
        let p95_idx = (((sorted.len() - 1) as f64) * 0.95).round() as usize;
        let p95_dt = sorted[p95_idx.min(sorted.len() - 1)];
        let min_fps = if max_dt > 1e-6 { 1.0 / max_dt } else { 0.0 };
        let p5_fps = if p95_dt > 1e-6 { 1.0 / p95_dt } else { 0.0 };

        FpsSummary {
            frames: self.frames,
            duration_sec,
            avg_fps,
            min_fps,
            p5_fps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jittered_15_6ms_windows_ticks_stay_above_55_fps() {
        let mut pacer = FramePacer::new(60.0);
        let mut tracker = FpsTracker::new(0.5);
        let mut t = 0.0;
        let mut rendered = 0u64;
        while t <= 25.5 {
            if let PaceDecision::Render { dt_sec, .. } = pacer.tick(t) {
                assert!(dt_sec > 0.0 && dt_sec <= 0.1, "dt_sec out of bounds: {dt_sec}");
                tracker.record_frame(t);
                rendered += 1;
            }
            t += 0.015625;
        }
        let s = tracker.summary();
        assert!(rendered > 1400, "expected >1400 frames in 25.5s, got {rendered}");
        assert!(s.avg_fps >= 55.0, "expected avg_fps >= 55, got {}", s.avg_fps);
        assert!(s.p5_fps >= 50.0, "expected p5_fps >= 50, got {}", s.p5_fps);
        assert!(s.min_fps >= 50.0, "expected min_fps >= 50, got {}", s.min_fps);
    }

    #[test]
    fn lag_spike_recovers_without_frame_burst() {
        let mut pacer = FramePacer::new(60.0);
        assert!(matches!(pacer.tick(0.0), PaceDecision::Render { .. }));
        assert!(matches!(pacer.tick(0.5), PaceDecision::Render { .. }));
        assert!(matches!(pacer.tick(0.502), PaceDecision::Wait { .. }));
    }

    #[test]
    fn tracker_skips_warmup_and_computes_avg_min_p5() {
        let mut tracker = FpsTracker::new(1.0);
        tracker.record_frame(0.0);
        tracker.record_frame(0.45);
        tracker.record_frame(0.95);
        let mut t = 1.0;
        for i in 0..300 {
            tracker.record_frame(t);
            t += if i == 150 { 1.0 / 30.0 } else { 1.0 / 60.0 };
        }
        let s = tracker.summary();
        assert!(s.avg_fps >= 58.0 && s.avg_fps <= 61.0, "avg_fps={}", s.avg_fps);
        assert!(s.min_fps >= 29.0 && s.min_fps <= 31.0, "min_fps={}", s.min_fps);
        assert!(s.p5_fps >= 55.0, "p5_fps={}", s.p5_fps);
    }
}

//! Conversation layer: turns companion messages into the pet's expressive state.
//!
//! Everything here is plain data in, plain data out (no GPU, no window, no clock): the app feeds
//! [`Incoming`] messages and frame deltas, and reads a [`ConvFrame`] (additive bone offsets,
//! expression weights, mouth) plus the subtitle to draw. That keeps every behaviour testable.
//!
//! Angle conventions follow the rest of the rig: positive head/chest pitch looks/leans down,
//! negative looks/leans up or back; eyes are `[pitch, yaw]`.

use crate::animation::smooth_ramp;
use crate::poses::PoseMap;
use crate::protocol::{EmotionKind, GestureKind, Incoming};

// ---------------------------------------------------------------------------------- bones

/// Bones the conversation layer may move; the blender adds these offsets on top of its pose.
pub const CONV_BONES: [&str; 15] = [
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "leftShoulder",
    "rightShoulder",
    "leftUpperArm",
    "leftLowerArm",
    "leftHand",
    "rightUpperArm",
    "rightLowerArm",
    "rightHand",
    "leftEye",
    "rightEye",
];
const N: usize = CONV_BONES.len();
const HIPS: usize = 0;
const SPINE: usize = 1;
const CHEST: usize = 2;
const NECK: usize = 3;
const HEAD: usize = 4;
const L_SHOULDER: usize = 5;
const R_SHOULDER: usize = 6;
const L_UPPER: usize = 7;
const L_LOWER: usize = 8;
const L_HAND: usize = 9;
const R_UPPER: usize = 10;
const R_LOWER: usize = 11;
const R_HAND: usize = 12;
const L_EYE: usize = 13;
const R_EYE: usize = 14;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Offsets(pub [[f32; 3]; N]);

impl Default for Offsets {
    fn default() -> Self {
        Self([[0.0; 3]; N])
    }
}

impl Offsets {
    fn add(&mut self, bone: usize, delta: [f32; 3]) {
        for (a, d) in self.0[bone].iter_mut().zip(delta) {
            *a += d;
        }
    }

    fn add_scaled(&mut self, other: &Offsets, w: f32) {
        for bone in 0..N {
            self.add(bone, other.0[bone].map(|x| x * w));
        }
    }

    fn eyes(&mut self, pitch: f32, yaw: f32) {
        self.add(L_EYE, [pitch, yaw, 0.0]);
        self.add(R_EYE, [pitch, yaw, 0.0]);
    }

    fn shoulders(&mut self, lift: f32) {
        self.add(L_SHOULDER, [0.0, 0.0, lift]);
        self.add(R_SHOULDER, [0.0, 0.0, -lift]);
    }

    pub fn get(&self, bone: &str) -> [f32; 3] {
        CONV_BONES.iter().position(|b| *b == bone).map_or([0.0; 3], |i| self.0[i])
    }

    fn from_pose_delta(poses: &PoseMap, name: &str) -> Self {
        let mut out = Self::default();
        let (Some(pose), Some(idle)) = (poses.get(name), poses.get("idle")) else {
            return out;
        };
        for (i, bone) in CONV_BONES.iter().enumerate() {
            let a = pose.get(*bone).copied().unwrap_or([0.0; 3]);
            let b = idle.get(*bone).copied().unwrap_or([0.0; 3]);
            out.0[i] = std::array::from_fn(|k| a[k] - b[k]);
        }
        out
    }
}

/// What the conversation asks of the rig this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConvFrame {
    pub active: bool,
    pub offsets: Offsets,
    /// Additive weights for `happy, angry, sad, relaxed, surprised`.
    pub expr: [f32; 5],
    /// Eyelid floor in 0..1 (half-lidded looks); blends with the automatic blink by `max`.
    pub lids: f32,
    /// Smoothed `aa, ih, ou, ee, oh`.
    pub mouth: [f32; 5],
}

// ------------------------------------------------------------------------------- visemes

/// After this long without a `mouth` message the target is forced to closed.
pub const MOUTH_STALE_SEC: f32 = 0.150;
const ATTACK_TAU: f32 = 0.030;
const RELEASE_TAU: f32 = 0.040;

#[derive(Clone, Copy, Debug, Default)]
pub struct VisemeSmoother {
    target: [f32; 5],
    current: [f32; 5],
    since_update: f32,
}

impl VisemeSmoother {
    pub fn set_target(&mut self, target: [f32; 5]) {
        self.target = target.map(|x| x.clamp(0.0, 1.0));
        self.since_update = 0.0;
    }

    pub fn step(&mut self, dt: f32) {
        self.since_update += dt;
        let stale = self.since_update > MOUTH_STALE_SEC;
        for (cur, goal) in self.current.iter_mut().zip(self.target) {
            let goal = if stale { 0.0 } else { goal };
            let tau = if goal > *cur { ATTACK_TAU } else { RELEASE_TAU };
            *cur += (goal - *cur) * (1.0 - (-dt / tau).exp());
            if *cur < 0.001 {
                *cur = 0.0;
            }
        }
    }

    pub fn current(&self) -> [f32; 5] {
        self.current
    }

    pub fn energy(&self) -> f32 {
        self.current.iter().sum::<f32>().min(1.2)
    }
}

/// How much of the mouth shape survives under the other expressions. VRM marks `happy`, `sad`
/// and `surprised` as "blend" mouth overrides, which would shut the mouth at full weight; a
/// talking face keeps at least 60% of the shape so lip sync stays readable.
pub fn mouth_attenuation(happy: f32, sad: f32, surprised: f32) -> f32 {
    1.0 - 0.38 * happy.max(sad).max(surprised).clamp(0.0, 1.0)
}

// ------------------------------------------------------------------------------ emotions

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EmotionProfile {
    /// `happy, angry, sad, relaxed, surprised`.
    pub expr: [f32; 5],
    pub lids: f32,
    /// Head `[pitch, yaw, roll]`.
    pub head: [f32; 3],
    pub chest_pitch: f32,
    pub shoulder_lift: f32,
}

impl EmotionProfile {
    fn scaled(self, k: f32) -> Self {
        Self {
            expr: self.expr.map(|x| x * k),
            lids: self.lids * k,
            head: self.head.map(|x| x * k),
            chest_pitch: self.chest_pitch * k,
            shoulder_lift: self.shoulder_lift * k,
        }
    }

    fn approach(&mut self, goal: &Self, alpha: f32) {
        let lerp = |a: &mut f32, b: f32| *a += (b - *a) * alpha;
        for (a, b) in self.expr.iter_mut().zip(goal.expr) {
            lerp(a, b);
        }
        for (a, b) in self.head.iter_mut().zip(goal.head) {
            lerp(a, b);
        }
        lerp(&mut self.lids, goal.lids);
        lerp(&mut self.chest_pitch, goal.chest_pitch);
        lerp(&mut self.shoulder_lift, goal.shoulder_lift);
    }
}

/// Emotion name → blend of VRM expressions plus subtle posture.
pub fn emotion_profile(kind: EmotionKind) -> EmotionProfile {
    use EmotionKind::*;
    // expr: happy, angry, sad, relaxed, surprised. head: pitch (+ down), yaw, roll.
    let p = |expr: [f32; 5], lids: f32, head: [f32; 3], chest_pitch: f32, shoulder_lift: f32| EmotionProfile {
        expr,
        lids,
        head,
        chest_pitch,
        shoulder_lift,
    };
    match kind {
        Neutral => EmotionProfile::default(),
        Happy => p([0.85, 0.0, 0.0, 0.0, 0.0], 0.0, [-0.03, 0.0, 0.04], -0.01, 0.02),
        Angry => p([0.0, 0.80, 0.0, 0.0, 0.0], 0.05, [0.05, 0.0, 0.0], 0.02, 0.03),
        Sad => p([0.0, 0.0, 0.85, 0.0, 0.0], 0.10, [0.10, 0.0, 0.03], 0.04, -0.02),
        Relaxed => p([0.0, 0.0, 0.0, 0.85, 0.0], 0.12, [0.0, 0.0, 0.04], -0.01, -0.01),
        Surprised => p([0.0, 0.0, 0.0, 0.0, 0.85], 0.0, [-0.06, 0.0, 0.0], -0.03, 0.04),
        Smug => p([0.35, 0.0, 0.0, 0.45, 0.0], 0.22, [-0.08, 0.10, 0.05], -0.02, 0.0),
        Teasing => p([0.60, 0.0, 0.0, 0.10, 0.0], 0.08, [-0.02, 0.06, 0.15], 0.0, 0.02),
        Embarrassed => p([0.30, 0.0, 0.30, 0.0, 0.10], 0.05, [0.12, -0.14, 0.08], 0.03, 0.06),
        Bored => p([0.0, 0.0, 0.10, 0.55, 0.0], 0.30, [0.06, 0.0, 0.06], 0.03, -0.02),
        Excited => p([0.85, 0.0, 0.0, 0.0, 0.25], 0.0, [-0.05, 0.0, 0.0], -0.02, 0.06),
        Confused => p([0.0, 0.22, 0.10, 0.0, 0.22], 0.0, [0.0, 0.06, 0.16], 0.0, 0.01),
    }
}

// --------------------------------------------------------------------------- subtitles

pub const SUBTITLE_FONT_PT: f32 = 11.5;
/// Inner text width of the bubble in points.
pub const SUBTITLE_WIDTH_PT: f32 = 168.0;
pub const SUBTITLE_MAX_LINES: usize = 3;
/// The subtitle stays fully visible this long after `end`, then fades.
pub const SUBTITLE_HOLD_SEC: f32 = 1.5;
pub const SUBTITLE_FADE_SEC: f32 = 0.35;
const SUBTITLE_FADE_IN_SEC: f32 = 0.12;
/// Reading pace used to scroll long text while the voice is still going.
const SUBTITLE_CHARS_PER_SEC: f32 = 16.0;

/// Approximate advance of `c` in ems for the UI font; deliberately a little wide so a line that
/// fits here never wraps again inside the renderer.
fn glyph_em(c: char) -> f32 {
    match c {
        'i' | 'l' | 'j' | '.' | ',' | ':' | ';' | '!' | '\'' | '|' | 'I' => 0.32,
        ' ' | 't' | 'f' | 'r' | '(' | ')' | '-' | '"' => 0.40,
        'm' | 'w' | 'M' | 'W' | '@' => 0.92,
        c if c.is_ascii_uppercase() => 0.68,
        c if c.is_ascii() => 0.58,
        c if c.is_alphabetic() && (c as u32) < 0x2E80 => 0.60,
        _ => 1.10,
    }
}

fn text_width_pt(s: &str) -> f32 {
    s.chars().map(glyph_em).sum::<f32>() * SUBTITLE_FONT_PT
}

/// Word-wraps `text` to `max_width_pt`. Existing newlines are kept; a word wider than a whole
/// line is broken between characters. Never loses or reorders a character other than the
/// whitespace used as break points.
pub fn wrap_text(text: &str, max_width_pt: f32) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if text_width_pt(word) > max_width_pt {
                if !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                }
                for c in word.chars() {
                    if !line.is_empty() && text_width_pt(&line) + text_width_pt(c.encode_utf8(&mut [0; 4])) > max_width_pt {
                        lines.push(std::mem::take(&mut line));
                    }
                    line.push(c);
                }
                continue;
            }
            if line.is_empty() {
                line.push_str(word);
            } else if text_width_pt(&line) + text_width_pt(" ") + text_width_pt(word) <= max_width_pt {
                line.push(' ');
                line.push_str(word);
            } else {
                lines.push(std::mem::replace(&mut line, word.to_string()));
            }
        }
        if !line.is_empty() {
            lines.push(line);
        }
    }
    lines
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubtitleView {
    pub lines: Vec<String>,
    /// More text above the visible lines.
    pub more_above: bool,
    pub opacity: f32,
}

#[derive(Clone, Debug, Default)]
struct Subtitle {
    lines: Vec<String>,
    age: f32,
    ended_age: Option<f32>,
}

/// Opacity of a subtitle `age` seconds after it appeared, `ended_age` seconds after its `end`.
pub fn subtitle_opacity(age: f32, ended_age: Option<f32>) -> f32 {
    let fade_in = smooth_ramp(age / SUBTITLE_FADE_IN_SEC);
    let fade_out = ended_age.map_or(1.0, |e| 1.0 - smooth_ramp((e - SUBTITLE_HOLD_SEC) / SUBTITLE_FADE_SEC));
    fade_in * fade_out
}

/// The slice of `lines` to show: all of them when they fit, otherwise the window ending at the
/// line being spoken (estimated from `age`), and the last lines once the speech has ended.
pub fn visible_window(lines: &[String], age: f32, ended: bool) -> (usize, usize) {
    if lines.len() <= SUBTITLE_MAX_LINES {
        return (0, lines.len());
    }
    let end = if ended {
        lines.len()
    } else {
        let spoken = (age * SUBTITLE_CHARS_PER_SEC) as usize;
        let mut acc = 0;
        let mut upto = lines.len();
        for (i, l) in lines.iter().enumerate() {
            acc += l.chars().count() + 1;
            if acc > spoken {
                upto = i + 1;
                break;
            }
        }
        upto.max(SUBTITLE_MAX_LINES)
    };
    (end - SUBTITLE_MAX_LINES, end)
}

// -------------------------------------------------------------------------- conversation

#[derive(Clone, Copy, Debug, PartialEq)]
struct ActiveGesture {
    kind: GestureKind,
    age: f32,
}

fn gesture_duration(kind: GestureKind) -> f32 {
    match kind {
        GestureKind::Wave => 2.4,
        GestureKind::Cheer => 2.0,
        GestureKind::Nod => 0.9,
        GestureKind::Shake => 1.1,
        GestureKind::Tilt => 1.8,
        GestureKind::Laugh => 2.0,
        GestureKind::Shrug => 1.5,
        GestureKind::LeanIn => 2.2,
        GestureKind::LookAway => 2.0,
    }
}

const MAX_GESTURES: usize = 4;
/// Without any message for this long, a "speaking" conversation is considered over (a companion
/// that died mid-sentence must not leave a subtitle on screen).
const SILENCE_TIMEOUT_SEC: f32 = 20.0;
/// How long a bare `emotion` message (no speech) is held before returning to neutral.
const EMOTION_HOLD_SEC: f32 = 6.0;

fn hold_env(age: f32, dur: f32, fade_in: f32, fade_out: f32) -> f32 {
    smooth_ramp(age / fade_in) * smooth_ramp((dur - age) / fade_out)
}

pub struct Conversation {
    listening: bool,
    thinking: bool,
    speaking: bool,
    w_listen: f32,
    w_think: f32,
    w_speak: f32,
    emotion: EmotionKind,
    intensity: f32,
    emotion_ttl: f32,
    mix: EmotionProfile,
    visemes: VisemeSmoother,
    energy_slow: f32,
    energy_fast: f32,
    accent: f32,
    brow: f32,
    gestures: Vec<ActiveGesture>,
    subtitle: Option<Subtitle>,
    silence: f32,
    clock: f32,
    saccade: [f32; 2],
    saccade_goal: [f32; 2],
    next_saccade: f32,
    rng: u32,
    wave_delta: Offsets,
    cheer_delta: Offsets,
}

impl Conversation {
    pub fn new(poses: &PoseMap) -> Self {
        Self {
            listening: false,
            thinking: false,
            speaking: false,
            w_listen: 0.0,
            w_think: 0.0,
            w_speak: 0.0,
            emotion: EmotionKind::Neutral,
            intensity: 1.0,
            emotion_ttl: 0.0,
            mix: EmotionProfile::default(),
            visemes: VisemeSmoother::default(),
            energy_slow: 0.0,
            energy_fast: 0.0,
            accent: 0.0,
            brow: 0.0,
            gestures: Vec::new(),
            subtitle: None,
            silence: 0.0,
            clock: 0.0,
            saccade: [0.0; 2],
            saccade_goal: [0.0; 2],
            next_saccade: 0.8,
            rng: 0x9E37_79B9,
            wave_delta: Offsets::from_pose_delta(poses, "wave"),
            cheer_delta: Offsets::from_pose_delta(poses, "cheer"),
        }
    }

    fn rand01(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.rng >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn handle(&mut self, msg: Incoming) {
        self.silence = 0.0;
        match msg {
            Incoming::State { listening, thinking, speaking } => {
                self.listening = listening;
                self.thinking = thinking;
                self.speaking = speaking;
            }
            Incoming::Say { text, emotion, .. } => {
                self.speaking = true;
                self.thinking = false;
                self.listening = false;
                if let Some(e) = emotion {
                    self.set_emotion(e, 1.0);
                }
                self.emotion_ttl = f32::INFINITY;
                let lines = wrap_text(&text, SUBTITLE_WIDTH_PT);
                self.subtitle = (!lines.is_empty()).then_some(Subtitle { lines, age: 0.0, ended_age: None });
            }
            Incoming::Mouth(v) => self.visemes.set_target(v.to_array()),
            Incoming::Emotion { kind, intensity } => {
                self.set_emotion(kind, intensity);
                if !self.speaking {
                    self.emotion_ttl = EMOTION_HOLD_SEC;
                }
            }
            Incoming::Gesture(kind) => {
                if self.gestures.len() >= MAX_GESTURES {
                    self.gestures.remove(0);
                }
                self.gestures.push(ActiveGesture { kind, age: 0.0 });
            }
            Incoming::End { .. } => {
                self.speaking = false;
                self.visemes.set_target([0.0; 5]);
                if let Some(s) = &mut self.subtitle {
                    s.ended_age.get_or_insert(0.0);
                }
                self.emotion_ttl = SUBTITLE_HOLD_SEC + SUBTITLE_FADE_SEC + 1.0;
            }
        }
    }

    fn set_emotion(&mut self, kind: EmotionKind, intensity: f32) {
        self.emotion = kind;
        self.intensity = intensity.clamp(0.0, 1.0);
    }

    /// The conversation owns the avatar (harness moods yield, the pet stays in place).
    pub fn is_active(&self) -> bool {
        self.listening
            || self.thinking
            || self.speaking
            || self.subtitle.is_some()
            || !self.gestures.is_empty()
            || self.emotion_ttl > 0.0
            || self.w_listen + self.w_think + self.w_speak > 0.01
            || self.mix.expr.iter().any(|x| *x > 0.01)
    }

    pub fn subtitle_view(&self) -> Option<SubtitleView> {
        let s = self.subtitle.as_ref()?;
        let (a, b) = visible_window(&s.lines, s.age, s.ended_age.is_some());
        Some(SubtitleView {
            lines: s.lines[a..b].to_vec(),
            more_above: a > 0,
            opacity: subtitle_opacity(s.age, s.ended_age),
        })
    }

    pub fn step(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.1);
        self.clock += dt;
        self.silence += dt;
        if self.speaking && self.silence > SILENCE_TIMEOUT_SEC {
            self.handle(Incoming::End { id: String::new() });
            self.silence = 0.0;
        }

        let ease = |dt: f32, rate: f32| 1.0 - (-dt * rate).exp();
        let a = ease(dt, 6.0);
        self.w_listen += (f32::from(self.listening) - self.w_listen) * a;
        self.w_think += (f32::from(self.thinking) - self.w_think) * a;
        self.w_speak += (f32::from(self.speaking) - self.w_speak) * a;

        self.emotion_ttl -= dt;
        let goal = if self.emotion_ttl > 0.0 {
            emotion_profile(self.emotion).scaled(self.intensity)
        } else {
            EmotionProfile::default()
        };
        self.mix.approach(&goal, ease(dt, 7.0));

        self.visemes.step(dt);
        let energy = self.visemes.energy();
        self.energy_fast += (energy - self.energy_fast) * ease(dt, 22.0);
        self.energy_slow += (energy - self.energy_slow) * ease(dt, 3.0);
        self.accent = (self.energy_fast - self.energy_slow).clamp(-0.6, 0.6);
        self.brow += (self.accent.max(0.0) - self.brow) * ease(dt, 12.0);

        self.next_saccade -= dt;
        if self.next_saccade <= 0.0 {
            self.next_saccade = 0.9 + self.rand01() * 2.0;
            self.saccade_goal = [(self.rand01() - 0.5) * 0.05, (self.rand01() - 0.5) * 0.10];
        }
        let follow = ease(dt, 14.0);
        for (cur, goal) in self.saccade.iter_mut().zip(self.saccade_goal) {
            *cur += (goal - *cur) * follow;
        }

        for g in &mut self.gestures {
            g.age += dt;
        }
        self.gestures.retain(|g| g.age < gesture_duration(g.kind));

        if let Some(s) = &mut self.subtitle {
            s.age += dt;
            if let Some(e) = &mut s.ended_age {
                *e += dt;
            }
            if subtitle_opacity(s.age, s.ended_age) <= 0.0 && s.ended_age.is_some() {
                self.subtitle = None;
            }
        }
    }

    pub fn frame(&self) -> ConvFrame {
        let mut o = Offsets::default();
        let t = self.clock;
        let m = &self.mix;
        let (ws, wl, wt) = (self.w_speak, self.w_listen, self.w_think);

        // Held emotional posture.
        o.add(HEAD, m.head);
        o.add(CHEST, [m.chest_pitch, 0.0, 0.0]);
        o.shoulders(m.shoulder_lift);

        // Listening: attentive, leaning in a little, head slightly tilted and lifted.
        o.add(HEAD, [-0.03 * wl, 0.0, 0.05 * wl + 0.012 * wl * (t * 1.1).sin()]);
        o.add(CHEST, [0.07 * wl, 0.0, 0.0]);
        o.add(SPINE, [0.02 * wl, 0.0, 0.0]);

        // Thinking: eyes up and to one side, head following a little; the side drifts slowly.
        let side = 0.55 + 0.45 * (t * 0.8).sin();
        o.eyes(-0.09 * wt, 0.13 * wt * side);
        o.add(HEAD, [-0.07 * wt, 0.15 * wt * side, -0.06 * wt]);
        o.add(NECK, [0.0, 0.05 * wt * side, 0.0]);

        // Speaking: head bobs on syllable energy, gentle sway, eyes wander, small hand talk.
        let bob = self.accent * 0.14 * ws;
        o.add(HEAD, [bob, 0.04 * ws * (t * 0.9).sin() + 0.015 * ws * (t * 2.3).sin(), 0.03 * ws * (t * 1.7).sin() + 0.02 * ws * (t * 0.63).sin()]);
        o.add(NECK, [bob * 0.5, 0.0, 0.0]);
        o.add(CHEST, [self.accent * 0.025 * ws, 0.0, 0.0]);
        o.shoulders(self.energy_fast * 0.02 * ws);
        o.eyes(self.saccade[0] * ws, self.saccade[1] * ws);
        let talk = (t * 1.9).sin();
        o.add(L_HAND, [0.0, 0.0, 0.05 * talk * ws * self.energy_fast]);
        o.add(R_HAND, [0.0, 0.0, -0.06 * talk * ws * self.energy_fast]);
        o.add(L_UPPER, [0.04 * self.accent * ws, 0.0, 0.0]);
        o.add(R_UPPER, [0.04 * self.accent * ws, 0.0, 0.0]);

        let mut expr = m.expr;
        expr[4] += 0.10 * wl + 0.12 * self.brow * ws; // brows rise: attentive, emphasis
        expr[3] += 0.12 * wt;

        let mut mouth = self.visemes.current();
        for g in &self.gestures {
            self.gesture_into(*g, &mut o, &mut expr, &mut mouth);
        }
        for x in expr.iter_mut().chain(mouth.iter_mut()) {
            *x = x.clamp(0.0, 1.0);
        }

        ConvFrame { active: self.is_active(), offsets: o, expr, lids: m.lids.clamp(0.0, 1.0), mouth }
    }

    fn gesture_into(&self, g: ActiveGesture, o: &mut Offsets, expr: &mut [f32; 5], mouth: &mut [f32; 5]) {
        let dur = gesture_duration(g.kind);
        let age = g.age;
        match g.kind {
            GestureKind::Wave => {
                let e = hold_env(age, dur, 0.35, 0.45);
                o.add_scaled(&self.wave_delta, e);
                let w = (age * 6.8).sin() * e;
                o.add(R_UPPER, [0.0, 0.0, w * 0.08]);
                o.add(R_LOWER, [0.0, w * 0.16, w * 0.24]);
                o.add(R_HAND, [0.0, 0.0, w * 0.18]);
                expr[0] += 0.3 * e;
            }
            GestureKind::Cheer => {
                let e = hold_env(age, dur, 0.3, 0.45);
                o.add_scaled(&self.cheer_delta, e);
                let bounce = (age * 9.0).sin();
                o.add(L_UPPER, [0.0, 0.0, bounce * 0.08 * e]);
                o.add(R_UPPER, [0.0, 0.0, -bounce * 0.08 * e]);
                o.add(CHEST, [-bounce.abs() * 0.035 * e, 0.0, 0.0]);
                expr[0] += 0.55 * e;
            }
            GestureKind::Nod => {
                let e = hold_env(age, dur, 0.10, 0.25);
                o.add(HEAD, [0.19 * (age * 13.5).sin() * e, 0.0, 0.0]);
                o.add(NECK, [0.06 * (age * 13.5).sin() * e, 0.0, 0.0]);
            }
            GestureKind::Shake => {
                let e = hold_env(age, dur, 0.10, 0.3);
                o.add(HEAD, [0.0, 0.27 * (age * 15.0).sin() * e, 0.02 * e]);
                o.add(NECK, [0.0, 0.08 * (age * 15.0).sin() * e, 0.0]);
            }
            GestureKind::Tilt => {
                let e = hold_env(age, dur, 0.28, 0.5);
                o.add(HEAD, [0.0, 0.05 * e, 0.27 * e]);
                o.add(NECK, [0.0, 0.0, 0.07 * e]);
                expr[4] += 0.08 * e;
            }
            GestureKind::Laugh => {
                let e = hold_env(age, dur, 0.2, 0.5);
                let b = (age * 17.0).sin();
                o.add(HEAD, [-0.12 * e + 0.04 * b * e, 0.0, 0.03 * b * e]);
                o.add(CHEST, [-0.04 * e - 0.03 * b.abs() * e, 0.0, 0.0]);
                o.shoulders(0.09 * b.abs() * e);
                expr[0] += 0.6 * e;
                mouth[0] += 0.35 * (0.5 + 0.5 * b) * e;
            }
            GestureKind::Shrug => {
                let e = hold_env(age, dur, 0.28, 0.5);
                o.shoulders(0.15 * e);
                o.add(HEAD, [0.0, 0.0, 0.12 * e]);
                o.add(L_UPPER, [0.0, 0.0, 0.18 * e]);
                o.add(R_UPPER, [0.0, 0.0, -0.18 * e]);
                o.add(L_LOWER, [0.0, -0.25 * e, 0.0]);
                o.add(R_LOWER, [0.0, 0.25 * e, 0.0]);
                expr[2] += 0.10 * e;
            }
            GestureKind::LeanIn => {
                let e = hold_env(age, dur, 0.4, 0.6);
                o.add(CHEST, [0.13 * e, 0.0, 0.0]);
                o.add(SPINE, [0.06 * e, 0.0, 0.0]);
                o.add(HIPS, [0.03 * e, 0.0, 0.0]);
                o.add(HEAD, [-0.05 * e, 0.0, 0.0]);
                expr[4] += 0.12 * e;
            }
            GestureKind::LookAway => {
                let e = hold_env(age, dur, 0.25, 0.5);
                o.add(HEAD, [0.03 * e, 0.30 * e, -0.04 * e]);
                o.add(NECK, [0.0, 0.10 * e, 0.0]);
                o.eyes(0.02 * e, 0.12 * e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poses::Pose;
    use crate::protocol::Visemes;

    fn poses() -> PoseMap {
        let mut m = PoseMap::new();
        let mut idle = Pose::new();
        idle.insert("rightUpperArm".into(), [0.08, 0.0, 1.22]);
        let mut wave = Pose::new();
        wave.insert("rightUpperArm".into(), [-0.35, -0.2, -1.35]);
        let mut cheer = Pose::new();
        cheer.insert("leftUpperArm".into(), [-0.3, 0.25, 1.25]);
        m.insert("idle".into(), idle);
        m.insert("wave".into(), wave);
        m.insert("cheer".into(), cheer);
        m
    }

    fn conv() -> Conversation {
        Conversation::new(&poses())
    }

    fn run(c: &mut Conversation, secs: f32) {
        for _ in 0..(secs * 60.0) as usize {
            c.step(1.0 / 60.0);
        }
    }

    fn mouth(aa: f32) -> Incoming {
        Incoming::Mouth(Visemes { aa, ..Visemes::default() })
    }

    // ---- visemes

    #[test]
    fn visemes_attack_fast_and_do_not_overshoot() {
        let mut s = VisemeSmoother::default();
        s.set_target([0.8, 0.0, 0.0, 0.0, 0.0]);
        for _ in 0..6 {
            s.step(1.0 / 60.0); // 100 ms
        }
        assert!(s.current()[0] > 0.6, "attack too slow: {}", s.current()[0]);
        for _ in 0..30 {
            s.set_target([0.8, 0.0, 0.0, 0.0, 0.0]);
            s.step(1.0 / 60.0);
            assert!(s.current()[0] <= 0.8 + 1e-4, "overshoot");
        }
    }

    #[test]
    fn visemes_release_smoothly_instead_of_snapping_shut() {
        let mut s = VisemeSmoother::default();
        for _ in 0..20 {
            s.set_target([0.9, 0.0, 0.0, 0.0, 0.0]);
            s.step(1.0 / 60.0);
        }
        let open = s.current()[0];
        s.set_target([0.0; 5]);
        s.step(1.0 / 60.0);
        let next = s.current()[0];
        assert!(next < open && next > open * 0.4, "release should ease: {open} -> {next}");
    }

    #[test]
    fn visemes_hold_for_150ms_then_decay_to_closed() {
        let mut s = VisemeSmoother::default();
        for _ in 0..20 {
            s.set_target([0.7, 0.0, 0.2, 0.0, 0.0]);
            s.step(1.0 / 60.0);
        }
        // Silence: 120 ms in, the last frame still holds.
        for _ in 0..7 {
            s.step(1.0 / 60.0);
        }
        assert!(s.current()[0] > 0.55, "stale window must still hold: {}", s.current()[0]);
        // 150 ms of silence plus 200 ms of release.
        for _ in 0..21 {
            s.step(1.0 / 60.0);
        }
        assert!(s.current().iter().all(|x| *x < 0.02), "mouth must be closed: {:?}", s.current());
    }

    #[test]
    fn mouth_attenuation_never_closes_the_mouth_fully() {
        assert_eq!(mouth_attenuation(0.0, 0.0, 0.0), 1.0);
        let full = mouth_attenuation(1.0, 0.0, 0.0);
        assert!(full >= 0.6 && full < 1.0, "{full}");
        assert!(mouth_attenuation(1.0, 1.0, 1.0) >= 0.6);
        assert!(mouth_attenuation(0.3, 0.0, 0.0) > full);
    }

    // ---- emotions

    #[test]
    fn every_emotion_has_a_sane_profile() {
        use EmotionKind::*;
        assert_eq!(emotion_profile(Neutral), EmotionProfile::default());
        for k in [Happy, Angry, Sad, Relaxed, Surprised, Smug, Teasing, Embarrassed, Bored, Excited, Confused] {
            let p = emotion_profile(k);
            assert!(p.expr.iter().all(|x| (0.0..=1.0).contains(x)), "{k:?} {p:?}");
            assert!((0.0..=0.5).contains(&p.lids), "{k:?}");
            assert!(p.expr.iter().sum::<f32>() > 0.3, "{k:?} needs a visible face");
            assert_ne!(p, EmotionProfile::default(), "{k:?}");
        }
    }

    #[test]
    fn base_emotions_drive_their_own_vrm_expression() {
        use EmotionKind::*;
        for (k, idx) in [(Happy, 0), (Angry, 1), (Sad, 2), (Relaxed, 3), (Surprised, 4)] {
            let p = emotion_profile(k);
            let max = p.expr.iter().cloned().fold(0.0, f32::max);
            assert_eq!(p.expr[idx], max, "{k:?}");
            assert!(p.expr[idx] >= 0.6, "{k:?}");
        }
    }

    #[test]
    fn extra_emotions_are_blends_with_their_own_signature() {
        use EmotionKind::*;
        let smug = emotion_profile(Smug);
        assert!(smug.lids > 0.1 && smug.expr[0] > 0.1 && smug.expr[3] > 0.2, "smug: half-lidded smirk");
        assert!(smug.head[0] < 0.0, "smug lifts the chin");
        let emb = emotion_profile(Embarrassed);
        assert!(emb.expr[0] > 0.1 && emb.expr[2] > 0.1, "embarrassed: smile mixed with sad");
        assert!(emb.head[0] > 0.05 && emb.head[1].abs() > 0.05, "embarrassed looks down and away");
        let bored = emotion_profile(Bored);
        assert!(bored.lids >= 0.25 && bored.expr[3] > 0.4);
        let excited = emotion_profile(Excited);
        assert!(excited.expr[0] > 0.6 && excited.expr[4] > 0.15);
        let confused = emotion_profile(Confused);
        assert!(confused.head[2].abs() > 0.1 && confused.expr[1] > 0.1 && confused.expr[4] > 0.1);
        let teasing = emotion_profile(Teasing);
        assert!(teasing.expr[0] > 0.4 && teasing.head[2].abs() > 0.1);
        let all = [smug, emb, bored, excited, confused, teasing];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn emotion_changes_ease_in_over_a_few_frames() {
        let mut c = conv();
        c.handle(Incoming::Emotion { kind: EmotionKind::Happy, intensity: 1.0 });
        c.step(1.0 / 60.0);
        let first = c.frame().expr[0];
        assert!(first > 0.0 && first < 0.2, "first frame must not jump: {first}");
        run(&mut c, 1.0);
        assert!(c.frame().expr[0] > 0.8);
    }

    #[test]
    fn intensity_scales_the_blend() {
        let mut c = conv();
        c.handle(Incoming::Emotion { kind: EmotionKind::Happy, intensity: 0.4 });
        run(&mut c, 1.5);
        let w = c.frame().expr[0];
        assert!((w - 0.4 * emotion_profile(EmotionKind::Happy).expr[0]).abs() < 0.03, "{w}");
    }

    #[test]
    fn a_bare_emotion_returns_to_neutral_by_itself() {
        let mut c = conv();
        c.handle(Incoming::Emotion { kind: EmotionKind::Sad, intensity: 1.0 });
        run(&mut c, 2.0);
        assert!(c.is_active());
        run(&mut c, 12.0);
        assert!(!c.is_active(), "conversation must release the avatar");
        assert!(c.frame().expr.iter().all(|x| *x < 0.02));
    }

    // ---- states and speech

    #[test]
    fn listening_leans_in_and_thinking_glances_up_and_aside() {
        let mut c = conv();
        c.handle(Incoming::State { listening: true, thinking: false, speaking: false });
        run(&mut c, 1.0);
        let f = c.frame();
        assert!(f.active);
        assert!(f.offsets.get("chest")[0] > 0.04, "leans forward");
        assert!(f.expr[4] > 0.05, "eyes a little wider");

        c.handle(Incoming::State { listening: false, thinking: true, speaking: false });
        run(&mut c, 1.5);
        let f = c.frame();
        assert!(f.offsets.get("leftEye")[0] < -0.03, "eyes up");
        assert!(f.offsets.get("leftEye")[1].abs() > 0.02, "eyes aside");
        assert!(f.offsets.get("chest")[0] < 0.02, "lean released");
    }

    #[test]
    fn head_bobs_with_syllable_energy_only_while_speaking() {
        let drive = |speaking: bool| {
            let mut c = conv();
            if speaking {
                c.handle(Incoming::State { listening: false, thinking: false, speaking: true });
            }
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for i in 0..180 {
                if i % 2 == 0 {
                    let on = (i / 8) % 2 == 0;
                    c.handle(mouth(if on { 0.9 } else { 0.05 }));
                }
                c.step(1.0 / 60.0);
                if i > 90 {
                    let p = c.frame().offsets.get("head")[0];
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            hi - lo
        };
        let talking = drive(true);
        assert!(talking > 0.04, "bob amplitude {talking}");
        assert!(drive(false) < talking * 0.4, "no bob without the speaking state");
    }

    #[test]
    fn speaking_eyes_wander_within_small_bounds() {
        let mut c = conv();
        c.handle(Incoming::State { listening: false, thinking: false, speaking: true });
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..600 {
            c.step(1.0 / 60.0);
            let e = c.frame().offsets.get("leftEye");
            assert!(e[0].abs() <= 0.05 && e[1].abs() <= 0.10);
            seen.insert((e[1] * 1000.0) as i32);
        }
        assert!(seen.len() > 5);
    }

    #[test]
    fn say_sets_emotion_subtitle_and_end_fades_after_one_and_a_half_seconds() {
        let mut c = conv();
        c.handle(Incoming::Say { id: "u1".into(), text: "Olá, tudo bem?".into(), emotion: Some(EmotionKind::Smug) });
        run(&mut c, 1.0);
        let v = c.subtitle_view().expect("subtitle shown");
        assert_eq!(v.lines, vec!["Olá, tudo bem?"]);
        assert_eq!(v.opacity, 1.0);
        assert!(c.frame().lids > 0.1, "smug lids");
        c.handle(Incoming::End { id: "u1".into() });
        run(&mut c, 1.4);
        assert_eq!(c.subtitle_view().map(|v| v.opacity), Some(1.0), "still fully visible at 1.4 s");
        run(&mut c, 0.2);
        let mid = c.subtitle_view().map(|v| v.opacity).unwrap_or(0.0);
        assert!(mid < 1.0, "fading after 1.5 s: {mid}");
        run(&mut c, 0.5);
        assert!(c.subtitle_view().is_none());
        run(&mut c, 3.0);
        assert!(!c.is_active());
    }

    #[test]
    fn end_closes_the_mouth() {
        let mut c = conv();
        c.handle(Incoming::State { listening: false, thinking: false, speaking: true });
        for _ in 0..30 {
            c.handle(mouth(0.9));
            c.step(1.0 / 60.0);
        }
        assert!(c.frame().mouth[0] > 0.6);
        c.handle(Incoming::End { id: String::new() });
        run(&mut c, 0.5);
        assert!(c.frame().mouth.iter().all(|x| *x < 0.02));
    }

    #[test]
    fn a_companion_that_goes_silent_mid_sentence_releases_the_avatar() {
        let mut c = conv();
        c.handle(Incoming::Say { id: "x".into(), text: "Vou dizer algo".into(), emotion: None });
        run(&mut c, SILENCE_TIMEOUT_SEC + 8.0);
        assert!(c.subtitle_view().is_none());
        assert!(!c.is_active());
    }

    // ---- gestures

    #[test]
    fn every_gesture_moves_something_and_then_ends_cleanly() {
        use GestureKind::*;
        for kind in [Wave, Cheer, Nod, Shake, Tilt, Laugh, Shrug, LeanIn, LookAway] {
            let mut c = conv();
            c.handle(Incoming::Gesture(kind));
            let mut peak = 0.0f32;
            for _ in 0..((gesture_duration(kind) * 60.0) as usize / 2) {
                c.step(1.0 / 60.0);
                let f = c.frame();
                let moved: f32 = f.offsets.0.iter().flatten().map(|x| x.abs()).sum::<f32>()
                    + f.expr.iter().sum::<f32>()
                    + f.mouth.iter().sum::<f32>();
                peak = peak.max(moved);
            }
            assert!(peak > 0.15, "{kind:?} barely moves: {peak}");
            run(&mut c, gesture_duration(kind) + 4.0);
            let f = c.frame();
            let rest: f32 = f.offsets.0.iter().flatten().map(|x| x.abs()).sum::<f32>();
            assert!(rest < 0.01, "{kind:?} must leave no residue: {rest}");
            assert!(!c.is_active(), "{kind:?} left the conversation active");
        }
    }

    #[test]
    fn nod_and_shake_use_the_right_axes() {
        let swing = |kind, axis: usize| {
            let mut c = conv();
            c.handle(Incoming::Gesture(kind));
            let mut m = 0.0f32;
            for _ in 0..40 {
                c.step(1.0 / 60.0);
                m = m.max(c.frame().offsets.get("head")[axis].abs());
            }
            m
        };
        assert!(swing(GestureKind::Nod, 0) > 0.1 && swing(GestureKind::Nod, 1) < 0.02);
        assert!(swing(GestureKind::Shake, 1) > 0.1 && swing(GestureKind::Shake, 0) < 0.02);
        assert!(swing(GestureKind::Tilt, 2) > 0.15);
    }

    #[test]
    fn wave_reuses_the_wave_pose_as_an_additive_delta() {
        let mut c = conv();
        c.handle(Incoming::Gesture(GestureKind::Wave));
        run(&mut c, 0.9);
        let arm = c.frame().offsets.get("rightUpperArm");
        assert!(arm[2] < -1.5, "arm raised toward the wave pose: {arm:?}");
    }

    #[test]
    fn gesture_stack_is_bounded() {
        let mut c = conv();
        for _ in 0..20 {
            c.handle(Incoming::Gesture(GestureKind::Nod));
        }
        assert!(c.gestures.len() <= MAX_GESTURES);
    }

    // ---- subtitles

    #[test]
    fn wrap_respects_width_and_keeps_every_word() {
        let text = "Você de novo? Achei que tinha ido dormir faz tempo, sinceramente.";
        let lines = wrap_text(text, SUBTITLE_WIDTH_PT);
        assert!(lines.len() >= 2);
        for l in &lines {
            assert!(text_width_pt(l) <= SUBTITLE_WIDTH_PT + 0.01, "too wide: {l:?}");
            assert_eq!(l.trim(), l);
        }
        assert_eq!(lines.join(" "), text);
    }

    #[test]
    fn wrap_breaks_a_word_longer_than_the_line_and_honours_newlines() {
        let long = "a".repeat(80);
        let lines = wrap_text(&long, SUBTITLE_WIDTH_PT);
        assert!(lines.len() >= 3);
        assert!(lines.iter().all(|l| text_width_pt(l) <= SUBTITLE_WIDTH_PT + 0.01));
        assert_eq!(lines.concat(), long);
        assert_eq!(wrap_text("um\ndois", SUBTITLE_WIDTH_PT), vec!["um", "dois"]);
        assert!(wrap_text("", SUBTITLE_WIDTH_PT).is_empty());
        assert!(wrap_text("  \n ", SUBTITLE_WIDTH_PT).is_empty());
    }

    #[test]
    fn wrap_handles_wide_glyphs() {
        let text = "こんにちは世界、これは長い字幕のテストです。もっと長くします。";
        let lines = wrap_text(text, SUBTITLE_WIDTH_PT);
        assert!(lines.len() >= 2);
        assert!(lines.iter().all(|l| text_width_pt(l) <= SUBTITLE_WIDTH_PT + 0.01));
        assert_eq!(lines.concat(), text);
    }

    #[test]
    fn long_subtitles_scroll_with_the_speech_and_settle_on_the_last_lines() {
        let lines: Vec<String> = (0..8).map(|i| format!("linha numero {i} do texto")).collect();
        assert_eq!(visible_window(&lines[..3], 0.0, false), (0, 3));
        assert_eq!(visible_window(&lines, 0.0, false), (0, 3), "starts at the top");
        let mid = visible_window(&lines, 6.0, false);
        assert!(mid.0 > 0 && mid.1 < 8 && mid.1 - mid.0 == SUBTITLE_MAX_LINES, "{mid:?}");
        assert_eq!(visible_window(&lines, 600.0, false), (5, 8), "never past the end");
        assert_eq!(visible_window(&lines, 0.0, true), (5, 8), "end jumps to the tail");
    }

    #[test]
    fn subtitle_opacity_fades_in_holds_and_fades_out() {
        assert!(subtitle_opacity(0.0, None) < 0.05);
        assert!(subtitle_opacity(0.06, None) > 0.3 && subtitle_opacity(0.06, None) < 0.9);
        assert_eq!(subtitle_opacity(0.5, None), 1.0);
        assert_eq!(subtitle_opacity(9.0, Some(1.49)), 1.0);
        let mid = subtitle_opacity(9.0, Some(SUBTITLE_HOLD_SEC + SUBTITLE_FADE_SEC * 0.5));
        assert!(mid > 0.3 && mid < 0.7, "{mid}");
        assert_eq!(subtitle_opacity(9.0, Some(SUBTITLE_HOLD_SEC + SUBTITLE_FADE_SEC)), 0.0);
    }
}

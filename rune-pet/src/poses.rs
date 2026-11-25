//! Pure humanoid VRM pose interpolation, idle breathing, grounded biomechanical walk cycle,
//! live pose secondary motion, auto-blink, and jump-free crossfading `PoseBlender`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f32::consts::{PI, TAU};
use std::path::Path;

use crate::animation::AnimationLife;
use crate::conversation::{CONV_BONES, ConvFrame, mouth_attenuation};
use crate::harness::HarnessMood;
use crate::pet_state::{compute_walk_phase_delta, facing_to_yaw_rad};

pub type Pose = BTreeMap<String, [f32; 3]>;
pub type PoseMap = BTreeMap<String, Pose>;

pub const HIP_SWING_RAD: f32 = 0.30;
pub const EXPRESSION_NAMES: [&str; 5] = ["happy", "angry", "sad", "relaxed", "surprised"];
/// VRM mouth presets driven by lip sync, in the order of `ConvFrame::mouth`.
pub const VISEME_NAMES: [&str; 5] = ["aa", "ih", "ou", "ee", "oh"];

pub struct EvaluatedPose {
    pub pose: Pose,
    pub root_yaw: f32,
    pub root_bob_y: f32,
    pub expressions: HashMap<&'static str, f32>,
}

pub fn clamp01(t: f32) -> f32 {
    t.clamp(0.0, 1.0)
}

pub fn ease_in_out_cubic(t: f32) -> f32 {
    let x = clamp01(t);
    if x < 0.5 {
        4.0 * x * x * x
    } else {
        1.0 - (-2.0 * x + 2.0).powi(3) / 2.0
    }
}

pub fn load_poses(path: &Path) -> Result<PoseMap, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read poses file {}: {e}", path.display()))?;
    serde_json::from_str::<PoseMap>(&raw)
        .map_err(|e| format!("failed to parse poses JSON {}: {e}", path.display()))
}

pub fn interpolate_pose(from_pose: &Pose, to_pose: &Pose, t: f32) -> Pose {
    let alpha = ease_in_out_cubic(t);
    let mut bones = BTreeSet::new();
    for k in from_pose.keys() {
        bones.insert(k.as_str());
    }
    for k in to_pose.keys() {
        bones.insert(k.as_str());
    }
    let mut out = Pose::new();
    for bone in bones {
        let a = from_pose.get(bone).copied().unwrap_or([0.0, 0.0, 0.0]);
        let b = to_pose.get(bone).copied().unwrap_or([0.0, 0.0, 0.0]);
        out.insert(
            bone.to_string(),
            [
                a[0] + (b[0] - a[0]) * alpha,
                a[1] + (b[1] - a[1]) * alpha,
                a[2] + (b[2] - a[2]) * alpha,
            ],
        );
    }
    out
}

pub fn scale_pose(pose: &Pose, weight: f32) -> Pose {
    let w = clamp01(weight);
    if w <= 1e-5 {
        return Pose::new();
    }
    pose.iter()
        .map(|(k, v)| (k.clone(), [v[0] * w, v[1] * w, v[2] * w]))
        .collect()
}

pub fn compute_idle_breathing(time_sec: f32) -> Pose {
    let breath = (time_sec * 1.55).sin();
    let sway = (time_sec * 0.65).sin();
    let mut out = Pose::new();
    out.insert("hips".into(), [0.0, sway * 0.012, sway * 0.008]);
    out.insert("spine".into(), [breath * 0.015, -sway * 0.008, -sway * 0.005]);
    out.insert("chest".into(), [breath * 0.022, 0.0, sway * 0.006]);
    out.insert("head".into(), [-breath * 0.012, sway * 0.016, sway * 0.008]);
    out.insert("leftUpperArm".into(), [0.0, 0.0, -breath * 0.022]);
    out.insert("rightUpperArm".into(), [0.0, 0.0, breath * 0.022]);
    out
}

/// Live secondary motion `(bone_offsets, root_bob_y)` for every pose so nothing feels static or stiff.
pub fn compute_pose_secondary_motion_full(
    pose_name: &str,
    time_sec: f32,
    carry_sway: f32,
) -> (Pose, f32) {
    let mut out = Pose::new();
    let mut bob_y = 0.0;
    match pose_name {
        "idle" => {
            let shift = (time_sec * 0.85).sin();
            let relax = (time_sec * 1.35 + 0.7).sin();
            out.insert("leftLowerArm".into(), [0.0, -relax * 0.04, -shift * 0.02]);
            out.insert("rightLowerArm".into(), [0.0, relax * 0.04, shift * 0.02]);
            out.insert("leftHand".into(), [0.0, 0.0, relax * 0.04]);
            out.insert("rightHand".into(), [0.0, 0.0, -relax * 0.04]);
        }
        "wave" => {
            let w = (time_sec * 6.8).sin();
            out.insert("rightUpperArm".into(), [0.0, 0.0, w * 0.08]);
            out.insert("rightLowerArm".into(), [0.0, w * 0.16, w * 0.24]);
            out.insert("rightHand".into(), [0.0, 0.0, w * 0.18]);
            out.insert("head".into(), [0.0, 0.0, -w * 0.04]);
            bob_y = 0.008 * w.abs();
        }
        "cheer" => {
            let bounce = (time_sec * 5.2).sin();
            out.insert("leftUpperArm".into(), [0.0, 0.0, bounce * 0.08]);
            out.insert("rightUpperArm".into(), [0.0, 0.0, -bounce * 0.08]);
            out.insert("chest".into(), [-bounce.abs() * 0.035, 0.0, 0.0]);
            out.insert("head".into(), [-bounce.abs() * 0.03, 0.0, bounce * 0.035]);
            bob_y = 0.016 * bounce.abs();
        }
        "sit" => {
            // Gentle relaxed dangling leg swing while perched on a window top edge.
            let swing = (time_sec * 2.1).sin();
            let sway = (time_sec * 0.9).sin();
            out.insert("leftLowerLeg".into(), [swing * 0.14, 0.0, 0.0]);
            out.insert("rightLowerLeg".into(), [-swing * 0.14, 0.0, 0.0]);
            out.insert("leftFoot".into(), [0.12 + swing * 0.06, 0.0, 0.0]);
            out.insert("rightFoot".into(), [0.12 - swing * 0.06, 0.0, 0.0]);
            out.insert("spine".into(), [0.0, sway * 0.02, sway * 0.015]);
        }
        "think" => {
            let nod = (time_sec * 1.6).sin();
            out.insert("head".into(), [nod * 0.04, 0.0, -nod * 0.02]);
            out.insert("rightHand".into(), [nod * 0.05, 0.0, 0.0]);
        }
        "point" => {
            let pulse = (time_sec * 3.4).sin();
            out.insert("rightUpperArm".into(), [pulse * 0.04, 0.0, 0.0]);
            out.insert("head".into(), [0.0, pulse * 0.02, 0.0]);
        }
        "carry" => {
            let sway = carry_sway.clamp(-1.0, 1.0);
            let kick = (time_sec * 6.2).sin();
            out.insert("hips".into(), [0.0, 0.0, -sway * 0.16]);
            out.insert("spine".into(), [0.0, 0.0, sway * 0.08]);
            out.insert("head".into(), [0.0, -sway * 0.08, -sway * 0.08]);
            out.insert("leftUpperArm".into(), [0.0, 0.0, kick * 0.07 - sway * 0.10]);
            out.insert("rightUpperArm".into(), [0.0, 0.0, -kick * 0.07 - sway * 0.10]);
            out.insert("leftUpperLeg".into(), [kick * 0.14, 0.0, -sway * 0.24]);
            out.insert("rightUpperLeg".into(), [-kick * 0.14, 0.0, -sway * 0.24]);
            out.insert("leftLowerLeg".into(), [kick * 0.16, 0.0, 0.0]);
            out.insert("rightLowerLeg".into(), [-kick * 0.16, 0.0, 0.0]);
        }
        "fall" => {
            let flail = (time_sec * 11.0).sin();
            out.insert("leftUpperArm".into(), [flail * 0.12, 0.0, flail * 0.14]);
            out.insert("rightUpperArm".into(), [-flail * 0.12, 0.0, flail * 0.14]);
            out.insert("leftUpperLeg".into(), [flail * 0.16, 0.0, 0.0]);
            out.insert("rightUpperLeg".into(), [-flail * 0.16, 0.0, 0.0]);
            out.insert("leftLowerLeg".into(), [-flail * 0.12, 0.0, 0.0]);
            out.insert("rightLowerLeg".into(), [flail * 0.12, 0.0, 0.0]);
        }
        "land" => {
            // Soft crouch hip drop so bent knees keep shoes planted on the floor.
            let settle = (time_sec * 5.0).sin();
            out.insert("leftUpperArm".into(), [0.0, 0.0, -settle * 0.03]);
            out.insert("rightUpperArm".into(), [0.0, 0.0, settle * 0.03]);
            bob_y = -0.055;
        }
        "jump" => {
            let tuck = (time_sec * 7.5).sin();
            out.insert("leftUpperArm".into(), [tuck * 0.06, 0.0, 0.0]);
            out.insert("rightUpperArm".into(), [-tuck * 0.06, 0.0, 0.0]);
            bob_y = 0.015;
        }
        "stretch" => {
            let s = (time_sec * 1.45).sin();
            let reach = 0.5 + 0.5 * s;
            out.insert("spine".into(), [-reach * 0.06, 0.0, s * 0.03]);
            out.insert("chest".into(), [-reach * 0.08, 0.0, -s * 0.03]);
            out.insert("leftUpperArm".into(), [-reach * 0.08, 0.0, reach * 0.10]);
            out.insert("rightUpperArm".into(), [-reach * 0.08, 0.0, -reach * 0.10]);
            out.insert("head".into(), [-reach * 0.06, 0.0, s * 0.05]);
            bob_y = 0.008 * reach;
        }
        "sleep" | "sleep_sit" => {
            let slow_breath = (time_sec * 0.95).sin();
            let nod = (time_sec * 0.45).sin();
            out.insert("spine".into(), [slow_breath * 0.025, 0.0, 0.0]);
            out.insert("chest".into(), [slow_breath * 0.035, 0.0, 0.0]);
            out.insert("head".into(), [nod * 0.06 + slow_breath * 0.02, 0.0, nod * 0.03]);
            if pose_name == "sleep_sit" {
                out.insert("leftLowerLeg".into(), [slow_breath * 0.03, 0.0, 0.0]);
                out.insert("rightLowerLeg".into(), [-slow_breath * 0.03, 0.0, 0.0]);
            }
        }
        "react" => {
            let hop = (time_sec * 6.5).sin();
            out.insert("head".into(), [-hop.abs() * 0.04, 0.0, hop * 0.07]);
            out.insert("leftLowerArm".into(), [0.0, -hop * 0.10, 0.0]);
            out.insert("rightLowerArm".into(), [0.0, hop * 0.10, 0.0]);
            bob_y = 0.022 * hop.abs();
        }
        _ => {}
    }
    (out, bob_y)
}

#[cfg(test)]
pub fn compute_pose_secondary_motion(pose_name: &str, time_sec: f32) -> Pose {
    compute_pose_secondary_motion_full(pose_name, time_sec, 0.0).0
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BlenderInput {
    pub dt_sec: f32,
    pub target_pose: &'static str,
    pub target_expression: &'static str,
    pub walking: bool,
    pub walk_dx_px: f32,
    pub walk_speed_factor: f32,
    pub scale: f64,
    pub facing: i8,
    pub eyes_closed: bool,
    pub carry_sway: f32,
    pub head_yaw: f32,
    pub head_pitch: f32,
    pub neck_yaw: f32,
    pub neck_pitch: f32,
    pub chest_yaw: f32,
    pub eyes: [f32; 2],
    pub cursor_attention: bool,
    pub petting_intensity: f32,
    pub harness_mood: Option<HarnessMood>,
    pub ambient_drowsiness: f32,
}

fn max_pose_angular_delta(a: &Pose, b: &Pose) -> f32 {
    let mut max_d = 0.0f32;
    for k in a.keys().chain(b.keys()) {
        let va = a.get(k).copied().unwrap_or([0.0; 3]);
        let vb = b.get(k).copied().unwrap_or([0.0; 3]);
        for i in 0..3 {
            max_d = max_d.max((vb[i] - va[i]).abs());
        }
    }
    max_d
}

fn pose_transition_duration_ms(from: &str, to: &str, max_angle_rad: f32) -> f32 {
    if from == to {
        return 0.0;
    }
    let base: f32 = match to {
        "fall" | "land" => 220.0,
        "carry" | "react" => 220.0,
        "wave" | "jump" => 260.0,
        "sleep" | "sleep_sit" | "stretch" => 480.0,
        "sit" => 380.0,
        _ => match from {
            "fall" | "land" | "react" => 260.0,
            "sleep" | "sleep_sit" | "stretch" => 440.0,
            _ => 340.0,
        },
    };
    // Scale duration up when large limb swings occur so no single frame ever jumps
    // (ease_in_out_cubic peaks at 3x average velocity at t=0.5).
    base.max(max_angle_rad * 195.0).min(580.0)
}

/// Pure, jump-free pose/expression/walk/secondary-motion crossfader.
pub struct PoseBlender {
    poses: PoseMap,
    pose_from_name: &'static str,
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
    eyes_closed_blend: f32,
    carry_sway_current: f32,
    head_yaw: f32,
    head_pitch: f32,
    neck_yaw: f32,
    neck_pitch: f32,
    chest_yaw: f32,
    eyes: [f32; 2],
    petting_intensity: f32,
    life: AnimationLife,
    drowsiness: f32,
    total_time_sec: f32,
    conv: ConvFrame,
}

impl PoseBlender {
    pub fn new(
        poses: PoseMap,
        initial_pose: &'static str,
        initial_expr: &'static str,
        initial_facing: i8,
    ) -> Self {
        let base_pose = poses
            .get(initial_pose)
            .or_else(|| poses.get("idle"))
            .cloned()
            .unwrap_or_default();
        let mut init_expr: HashMap<&'static str, f32> =
            EXPRESSION_NAMES.iter().map(|&n| (n, 0.0)).collect();
        if let Some(slot) = init_expr.get_mut(initial_expr) {
            *slot = 1.0;
        }
        Self {
            poses,
            pose_from_name: initial_pose,
            pose_name: initial_pose,
            pose_from: base_pose.clone(),
            pose_to: base_pose.clone(),
            pose_current: base_pose,
            pose_elapsed_ms: 0.0,
            pose_duration_ms: 0.0,
            expr_name: initial_expr,
            expr_from: init_expr.clone(),
            expr_to: init_expr.clone(),
            expr_weights: init_expr,
            expr_elapsed_ms: 0.0,
            expr_duration_ms: 0.0,
            walk_phase: std::f32::consts::FRAC_PI_2,
            walk_blend: 0.0,
            facing_current: facing_to_yaw_rad(initial_facing),
            eyes_closed_blend: 0.0,
            carry_sway_current: 0.0,
            head_yaw: 0.0,
            head_pitch: 0.0,
            neck_yaw: 0.0,
            neck_pitch: 0.0,
            chest_yaw: 0.0,
            eyes: [0.0; 2],
            petting_intensity: 0.0,
            life: AnimationLife::default(),
            drowsiness: 0.0,
            total_time_sec: 0.0,
            conv: ConvFrame::default(),
        }
    }

    /// Conversation layer for the next evaluation: additive bone offsets, expression weights,
    /// an eyelid floor and lip-sync weights. The default frame changes nothing.
    pub fn set_conversation(&mut self, frame: ConvFrame) {
        self.conv = frame;
    }

    pub fn set_pose_target(&mut self, name: &'static str, duration_ms: f32) {
        let target = self
            .poses
            .get(name)
            .or_else(|| self.poses.get("idle"))
            .cloned()
            .unwrap_or_default();
        self.pose_from_name = self.pose_name;
        self.pose_name = name;
        self.pose_from = self.pose_current.clone();
        self.pose_to = target;
        self.pose_elapsed_ms = 0.0;
        self.pose_duration_ms = duration_ms.max(0.0);
        if duration_ms <= 0.0 {
            self.pose_from_name = name;
            self.pose_current = self.pose_to.clone();
        }
    }

    pub fn set_expr_target(&mut self, name: &'static str, duration_ms: f32) {
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

    pub fn apply_preview(
        &mut self,
        pose_name: &'static str,
        expr_name: &'static str,
        walking: bool,
        facing: i8,
        walk_phase: f32,
        eyes_closed: bool,
    ) {
        self.set_pose_target(pose_name, 0.0);
        self.set_expr_target(expr_name, 0.0);
        self.walk_phase = walk_phase;
        self.walk_blend = if walking { 1.0 } else { 0.0 };
        self.facing_current = facing_to_yaw_rad(facing);
        self.eyes_closed_blend = if eyes_closed { 1.0 } else { 0.0 };
        self.carry_sway_current = 0.0;
        self.head_yaw = 0.0;
        self.head_pitch = 0.0;
        self.neck_yaw = 0.0;
        self.neck_pitch = 0.0;
        self.chest_yaw = 0.0;
        self.eyes = [0.0; 2];
        self.petting_intensity = 0.0;
        self.total_time_sec = 1.2;
    }

    pub fn step(&mut self, input: BlenderInput) {
        let dt_sec = input.dt_sec.clamp(0.0, 0.1);
        let dt_ms = dt_sec * 1000.0;
        self.total_time_sec += dt_sec;

        if input.target_pose != self.pose_name {
            let target = self
                .poses
                .get(input.target_pose)
                .or_else(|| self.poses.get("idle"))
                .cloned()
                .unwrap_or_default();
            let max_angle = max_pose_angular_delta(&self.pose_current, &target);
            let dur = pose_transition_duration_ms(self.pose_name, input.target_pose, max_angle);
            self.set_pose_target(input.target_pose, dur);
        }
        if input.target_expression != self.expr_name {
            self.set_expr_target(input.target_expression, 220.0);
        }

        if self.pose_duration_ms > 0.0 && self.pose_elapsed_ms < self.pose_duration_ms {
            self.pose_elapsed_ms = (self.pose_elapsed_ms + dt_ms).min(self.pose_duration_ms);
            if self.pose_elapsed_ms >= self.pose_duration_ms {
                self.pose_from_name = self.pose_name;
                self.pose_current = self.pose_to.clone();
            } else {
                self.pose_current = interpolate_pose(
                    &self.pose_from,
                    &self.pose_to,
                    self.pose_elapsed_ms / self.pose_duration_ms,
                );
            }
        }

        if self.expr_duration_ms > 0.0 && self.expr_elapsed_ms < self.expr_duration_ms {
            self.expr_elapsed_ms = (self.expr_elapsed_ms + dt_ms).min(self.expr_duration_ms);
            let alpha = ease_in_out_cubic(self.expr_elapsed_ms / self.expr_duration_ms);
            for &n in &EXPRESSION_NAMES {
                let a = self.expr_from.get(n).copied().unwrap_or(0.0);
                let b = self.expr_to.get(n).copied().unwrap_or(0.0);
                self.expr_weights.insert(n, a + (b - a) * alpha);
            }
        }

        // Tie walk animation strictly to actual physical horizontal pixels moved (DPI-scaled)
        // and smoothly blend into/out of idle so starts and stops never snap.
        if input.walking && input.walk_dx_px.abs() > 1e-4 {
            self.walk_phase += compute_walk_phase_delta(input.walk_dx_px, input.scale);
        }
        let blend_target = if input.walking {
            (0.25 + 0.75 * input.walk_speed_factor).clamp(0.0, 1.0)
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
        if !input.walking && self.walk_blend <= 1e-3 {
            self.walk_blend = 0.0;
            self.walk_phase = std::f32::consts::FRAC_PI_2;
        }

        let target_yaw = facing_to_yaw_rad(input.facing);
        let yaw_lerp = (dt_sec * 8.0).min(1.0);
        self.facing_current += (target_yaw - self.facing_current) * yaw_lerp;

        let eyes_target = if input.eyes_closed { 1.0 } else { 0.0 };
        let eyes_step = 5.5 * dt_sec;
        let eyes_delta = eyes_target - self.eyes_closed_blend;
        if eyes_delta.abs() <= eyes_step {
            self.eyes_closed_blend = eyes_target;
        } else {
            self.eyes_closed_blend += eyes_delta.signum() * eyes_step;
        }

        let sway_lerp = (dt_sec * 10.0).min(1.0);
        self.carry_sway_current += (input.carry_sway.clamp(-1.0, 1.0) - self.carry_sway_current) * sway_lerp;

        // Smooth gating as well as tracking: starting a walk or closing the eyes must
        // not drop the last live head/neck offsets in one frame.
        let gaze_alpha = 1.0 - (-dt_sec * 18.0).exp();
        for (value, target) in [
            (&mut self.head_yaw, input.head_yaw),
            (&mut self.head_pitch, input.head_pitch),
            (&mut self.neck_yaw, input.neck_yaw),
            (&mut self.neck_pitch, input.neck_pitch),
            (&mut self.chest_yaw, input.chest_yaw),
            (&mut self.petting_intensity, input.petting_intensity.clamp(0.0, 1.0)),
        ] {
            *value += (target - *value) * gaze_alpha;
        }
        let eye_alpha = 1.0 - (-dt_sec * 32.0).exp();
        for (value, target) in self.eyes.iter_mut().zip(input.eyes) {
            *value += (target - *value) * eye_alpha;
        }
        self.life.step_landing(dt_sec, input.target_pose == "land");
        let idle = input.target_pose == "idle"
            && !input.walking
            && !input.cursor_attention
            && input.harness_mood.is_none();
        self.life.step(dt_sec, idle, input.harness_mood, input.eyes_closed);
        let drowsiness = if idle && !input.eyes_closed && input.petting_intensity < 0.1 {
            input.ambient_drowsiness.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let rate = if drowsiness > self.drowsiness { 3.0 } else { 7.0 };
        self.drowsiness += (drowsiness - self.drowsiness) * (1.0 - (-dt_sec * rate).exp());
    }

    pub fn evaluate(&self, forced_preview: bool) -> EvaluatedPose {
        let breathing = compute_idle_breathing(self.total_time_sec);
        let (secondary, sec_bob_y) = if forced_preview {
            (Pose::new(), 0.0)
        } else {
            let pose_alpha = if self.pose_duration_ms > 0.0 && self.pose_elapsed_ms < self.pose_duration_ms {
                ease_in_out_cubic(self.pose_elapsed_ms / self.pose_duration_ms)
            } else {
                1.0
            };
            let (to_sec, to_bob) = compute_pose_secondary_motion_full(
                self.pose_name,
                self.total_time_sec,
                self.carry_sway_current,
            );
            if pose_alpha < 0.999 && self.pose_from_name != self.pose_name {
                let (from_sec, from_bob) = compute_pose_secondary_motion_full(
                    self.pose_from_name,
                    self.total_time_sec,
                    self.carry_sway_current,
                );
                let mut blended = Pose::new();
                for k in from_sec.keys().chain(to_sec.keys()) {
                    let a = from_sec.get(k).copied().unwrap_or([0.0; 3]);
                    let b = to_sec.get(k).copied().unwrap_or([0.0; 3]);
                    blended.insert(
                        k.clone(),
                        [
                            a[0] + (b[0] - a[0]) * pose_alpha,
                            a[1] + (b[1] - a[1]) * pose_alpha,
                            a[2] + (b[2] - a[2]) * pose_alpha,
                        ],
                    );
                }
                (blended, from_bob + (to_bob - from_bob) * pose_alpha)
            } else {
                (to_sec, to_bob)
            }
        };

        let walk_weight = if forced_preview {
            if self.walk_blend > 0.0 { 1.0 } else { 0.0 }
        } else {
            ease_in_out_cubic(self.walk_blend)
        };
        let (walk_offsets, walk_bob_y) = if walk_weight > 1e-4 {
            let (raw_off, raw_bob) = compute_walk_offsets(self.walk_phase);
            (scale_pose(&raw_off, walk_weight), raw_bob * walk_weight)
        } else {
            (Pose::new(), 0.0)
        };
        let breath_weight = 1.0 - walk_weight * 0.65;
        let sec_weight = 1.0 - walk_weight;

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
            "neck",
            "chest",
            "leftUpperLeg",
            "rightUpperLeg",
            "leftLowerLeg",
            "rightLowerLeg",
            "leftFoot",
            "rightFoot",
        ] {
            bone_names.insert(b);
        }

        let conv = if forced_preview { ConvFrame::default() } else { self.conv };
        if conv.active {
            for (i, bone) in CONV_BONES.iter().enumerate() {
                if conv.offsets.0[i].iter().any(|x| x.abs() > 1e-5) {
                    bone_names.insert(bone);
                }
            }
        }

        if !forced_preview {
            if self.eyes.iter().any(|x| x.abs() > 1e-4) {
                bone_names.extend(["leftEye", "rightEye"]);
            }
            if self.life.offset("leftShoulder")[2].abs() > 1e-4 {
                bone_names.extend(["leftShoulder", "rightShoulder"]);
            }
        }

        let petting_roll = if self.petting_intensity > 1e-3 && !forced_preview {
            (self.total_time_sec * 4.2).sin() * 0.09 * self.petting_intensity
        } else {
            0.0
        };

        let mut combined = Pose::new();
        for bone in bone_names {
            let base = self.pose_current.get(bone).copied().unwrap_or([0.0; 3]);
            let add = breathing.get(bone).copied().unwrap_or([0.0; 3]);
            let sec = secondary.get(bone).copied().unwrap_or([0.0; 3]);
            let walk = walk_offsets.get(bone).copied().unwrap_or([0.0; 3]);
            let gaze = if forced_preview {
                [0.0; 3]
            } else {
                match bone {
                    "head" => [self.head_pitch + self.drowsiness * 0.11, self.head_yaw, petting_roll],
                    "neck" => [self.neck_pitch, self.neck_yaw, 0.0],
                    "chest" => [0.0, self.chest_yaw, 0.0],
                    "leftEye" | "rightEye" => [self.eyes[0], self.eyes[1], 0.0],
                    _ => [0.0; 3],
                }
            };
            let accent = if forced_preview { [0.0; 3] } else { self.life.offset(bone) };
            let talk = if conv.active { conv.offsets.get(bone) } else { [0.0; 3] };
            let rotation = std::array::from_fn(|axis| {
                base[axis] + add[axis] * breath_weight + sec[axis] * sec_weight
                    + walk[axis] + gaze[axis] + accent[axis] * sec_weight
                    + talk[axis] * sec_weight
            });
            combined.insert(bone.to_string(), rotation);
        }

        let mut expressions = self.expr_weights.clone();
        if !forced_preview {
            let [brows, mood_softness] = self.life.mood_face();
            let softness = mood_softness.max(self.drowsiness * 0.35);
            for (name, accent) in [("angry", brows), ("relaxed", softness)] {
                if let Some(weight) = expressions.get_mut(name) {
                    *weight = (*weight + accent).min(1.0);
                }
            }
        }
        if conv.active {
            for (name, add) in EXPRESSION_NAMES.iter().zip(conv.expr) {
                if let Some(weight) = expressions.get_mut(*name) {
                    *weight = (*weight + add).min(1.0);
                }
            }
            let w = |name: &str| expressions.get(name).copied().unwrap_or(0.0);
            let keep = mouth_attenuation(w("happy"), w("sad"), w("surprised"));
            for (name, v) in VISEME_NAMES.iter().zip(conv.mouth) {
                expressions.insert(*name, (v * keep).clamp(0.0, 1.0));
            }
        }
        let happy = expressions.get("happy").copied().unwrap_or(0.0);
        let happy_suppress = 1.0 - smoothstep(0.25, 0.65, happy);
        let auto_blink = if forced_preview {
            0.0
        } else {
            self.life.blink() * happy_suppress
        };
        let pet_squint = if forced_preview {
            0.0
        } else {
            (self.petting_intensity * 0.65).clamp(0.0, 0.72)
        };
        let blink = auto_blink
            .max(self.eyes_closed_blend)
            .max(pet_squint)
            .max(if forced_preview { 0.0 } else { self.drowsiness * 0.22 })
            .max(conv.lids)
            .clamp(0.0, 1.0);
        expressions.insert("blink", blink);

        EvaluatedPose {
            pose: combined,
            root_yaw: self.facing_current,
            root_bob_y: walk_bob_y + sec_bob_y * sec_weight,
            expressions,
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp01((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

/// Evaluates one leg's `(thigh_x, knee_x, foot_x)` at cycle phase `phase` (radians):
/// - `phase = 0` (`u = 0.0`): heel strike (forward contact)
/// - `phase = PI/2` (`u = 0.25`): mid-stance (planted foot directly under hip)
/// - `phase = PI` (`u = 0.50`): toe-off (push-off)
/// - `phase = 3*PI/2` (`u = 0.75`): mid-swing (bent knee clears floor)
fn leg_kinematics(phase: f32, hips_x: f32) -> (f32, f32, f32) {
    let u = phase.rem_euclid(TAU) / TAU;
    let thigh_x = -HIP_SWING_RAD * phase.cos() - 0.025 * (2.0 * phase).sin();

    // Knee flexion (>= 0): slight cushion at heel strike, near-straight at mid-stance,
    // high clearance bend during swing.
    let stance_cushion = if u < 0.28 {
        let s = (PI * (u / 0.28)).sin();
        0.07 * s * s
    } else {
        0.0
    };
    let swing_flex = if u > 0.36 {
        let s = ((u - 0.36) / 0.64).clamp(0.0, 1.0);
        0.66 * (PI * s).sin().powf(1.45)
    } else {
        0.0
    };
    let knee_x = 0.035 + stance_cushion + swing_flex;

    // Ankle pitch: flat sole during stance, dorsiflex at heel strike/swing, plantarflex at push-off.
    let flat_sole = -(hips_x + thigh_x + knee_x);
    let flat_weight = smoothstep(0.02, 0.12, u) * (1.0 - smoothstep(0.32, 0.44, u));
    let push_off = smoothstep(0.36, 0.48, u) * (1.0 - smoothstep(0.52, 0.66, u));
    let heel_strike = (1.0 - smoothstep(0.0, 0.10, u)) + smoothstep(0.90, 1.0, u);
    let foot_x = flat_weight * flat_sole + push_off * 0.22 - heel_strike * 0.10 - 0.05 * (1.0 - flat_weight) * (1.0 - push_off);

    (thigh_x, knee_x, foot_x)
}

/// Natural grounded bipedal walk cycle tied to horizontal stride phase:
/// - `phase = 0` / `PI`: double-support contact where hips drop smoothly (`bob_y < 0`) so angled
///   legs keep the planted shoe soles flush on the floor (`y = 0`).
/// - `phase = PI/2` / `3*PI/2`: mid-stance passing pose where the stance leg is vertical (`bob_y ~ 0`)
///   and the swing knee lifts cleanly off the floor.
/// - Opposite arms swing naturally with VRM Y-axis elbow flexion and pelvis/torso counter-twist.
pub fn compute_walk_offsets(walk_phase: f32) -> (Pose, f32) {
    let hips_x = 0.025;
    let (l_thigh, l_knee, l_foot) = leg_kinematics(walk_phase, hips_x);
    let (r_thigh, r_knee, r_foot) = leg_kinematics(walk_phase + PI, hips_x);

    let c = walk_phase.cos();
    let s = walk_phase.sin();
    // Neutralize idle's slight +0.08 backward arm bias so forward/back arm swing is symmetric.
    let arm_swing = 0.30 * c;
    // In VRM 1.0 right-handed coords (+X = avatar left, +Y = up, +Z = forward):
    // R_y(theta) maps +X -> (cos, 0, -sin) and -X -> (-cos, 0, +sin).
    // Thus forward elbow flexion is NEGATIVE Y for leftLowerArm and POSITIVE Y for rightLowerArm.
    let elbow_l = 0.18 + (-c).max(0.0) * 0.34;
    let elbow_r = 0.18 + c.max(0.0) * 0.34;

    let mut offsets = Pose::new();
    offsets.insert("leftUpperLeg".into(), [l_thigh, 0.0, 0.0]);
    offsets.insert("rightUpperLeg".into(), [r_thigh, 0.0, 0.0]);
    offsets.insert("leftLowerLeg".into(), [l_knee, 0.0, 0.0]);
    offsets.insert("rightLowerLeg".into(), [r_knee, 0.0, 0.0]);
    offsets.insert("leftFoot".into(), [l_foot, 0.0, 0.0]);
    offsets.insert("rightFoot".into(), [r_foot, 0.0, 0.0]);

    offsets.insert("leftUpperArm".into(), [arm_swing - 0.08, 0.0, 0.0]);
    offsets.insert("rightUpperArm".into(), [-arm_swing - 0.08, 0.0, 0.0]);
    offsets.insert("leftLowerArm".into(), [0.0, -elbow_l - 0.15, 0.0]);
    offsets.insert("rightLowerArm".into(), [0.0, elbow_r + 0.15, 0.0]);

    offsets.insert("hips".into(), [hips_x, -0.055 * c, 0.022 * s]);
    offsets.insert("spine".into(), [0.020, 0.040 * c, -0.014 * s]);
    offsets.insert("chest".into(), [0.012, 0.032 * c, -0.010 * s]);
    offsets.insert("head".into(), [-0.015, -0.018 * c, 0.0]);

    // Pelvis vertical compensation: drops at contact (cos(2*p) = +1) so planted feet stay on y=0,
    // rises smoothly back to 0 at mid-stance (cos(2*p) = -1).
    let bob_y = -0.017 - 0.019 * (2.0 * walk_phase).cos();
    (offsets, bob_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn ease_boundaries_and_midpoint() {
        assert_eq!(ease_in_out_cubic(-1.0), 0.0);
        assert_eq!(ease_in_out_cubic(0.0), 0.0);
        assert!((ease_in_out_cubic(0.5) - 0.5).abs() < 1e-6);
        assert_eq!(ease_in_out_cubic(1.0), 1.0);
        assert_eq!(ease_in_out_cubic(2.0), 1.0);
    }

    #[test]
    fn interpolate_pose_blends_bones() {
        let mut a = Pose::new();
        a.insert("head".into(), [0.0, 0.0, 0.0]);
        let mut b = Pose::new();
        b.insert("head".into(), [1.0, -2.0, 0.5]);
        let mid = interpolate_pose(&a, &b, 0.5);
        assert!((mid["head"][0] - 0.5).abs() < 1e-5);
        assert!((mid["head"][1] + 1.0).abs() < 1e-5);
        assert!((mid["head"][2] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn walk_cycle_swings_arms_opposite_to_legs_with_non_negative_knees() {
        let (off, bob) = compute_walk_offsets(0.2);
        assert!(off["leftUpperLeg"][0] * off["rightUpperLeg"][0] < 0.0);
        assert!(off["leftUpperArm"][0] * off["leftUpperLeg"][0] < 0.0);
        assert!(off["leftLowerLeg"][0] >= 0.0 && off["rightLowerLeg"][0] >= 0.0);
        assert!(bob <= 0.005 && bob >= -0.045, "bob={bob}");
    }

    #[test]
    fn walk_contact_drops_hips_and_bends_elbows_on_y_axis() {
        let (contact, bob_contact) = compute_walk_offsets(0.0);
        let (midstance, bob_mid) = compute_walk_offsets(std::f32::consts::FRAC_PI_2);
        assert!(
            bob_contact < bob_mid,
            "hips must drop at double-support contact ({bob_contact}) below mid-stance ({bob_mid}) so planted feet stay on floor"
        );
        let elbow_y_max = contact
            .get("leftLowerArm")
            .map(|v| v[1].abs())
            .unwrap_or(0.0)
            .max(
                midstance
                    .get("leftLowerArm")
                    .map(|v| v[1].abs())
                    .unwrap_or(0.0),
            );
        assert!(
            elbow_y_max > 0.05,
            "elbow bend in VRM normalized space must use Y axis, got {elbow_y_max}"
        );
        // Forward elbow flexion in VRM 1.0 (+X left, +Z forward): leftLowerArm Y < 0, rightLowerArm Y > 0
        assert!(contact["leftLowerArm"][1] < 0.0);
        assert!(contact["rightLowerArm"][1] > 0.0);
    }

    #[test]
    fn scale_pose_and_secondary_motion_animate_wave_and_sit() {
        let (walk, _) = compute_walk_offsets(0.3);
        assert!(scale_pose(&walk, 0.0).is_empty());
        let half = scale_pose(&walk, 0.5);
        assert!((half["leftUpperLeg"][0] - walk["leftUpperLeg"][0] * 0.5).abs() < 1e-5);

        let wave_a = compute_pose_secondary_motion("wave", 0.2);
        let wave_b = compute_pose_secondary_motion("wave", 0.65);
        assert!((wave_a["rightLowerArm"][2] - wave_b["rightLowerArm"][2]).abs() > 0.05);

        let sit_a = compute_pose_secondary_motion("sit", 0.5);
        assert!(sit_a["leftLowerLeg"][0] * sit_a["rightLowerLeg"][0] < 0.0);
    }

    #[test]
    fn loads_project_poses_json() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../poses.json");
        let poses = load_poses(&path).expect("poses.json should load");
        for name in [
            "idle", "wave", "cheer", "sit", "bow", "point", "think",
            "carry", "fall", "land", "jump", "stretch", "sleep", "sleep_sit", "react",
        ] {
            assert!(poses.contains_key(name), "missing pose {name}");
        }
        assert!(poses["cheer"].contains_key("leftUpperArm"));
        assert!(poses["cheer"].contains_key("rightUpperArm"));
    }

    #[test]
    fn secondary_motion_keeps_all_interactive_and_idle_poses_alive() {
        for pose_name in ["idle", "wave", "cheer", "sit", "carry", "fall", "stretch", "sleep", "react"] {
            let (a, bob_a) = compute_pose_secondary_motion_full(pose_name, 0.25, 0.5);
            let (b, bob_b) = compute_pose_secondary_motion_full(pose_name, 1.35, -0.5);
            let mut max_diff = (bob_a - bob_b).abs();
            for k in a.keys().chain(b.keys()) {
                let va = a.get(k).copied().unwrap_or([0.0; 3]);
                let vb = b.get(k).copied().unwrap_or([0.0; 3]);
                for i in 0..3 {
                    max_diff = max_diff.max((va[i] - vb[i]).abs());
                }
            }
            assert!(
                max_diff > 0.01,
                "pose '{pose_name}' secondary motion should animate smoothly (diff={max_diff})"
            );
        }
    }

    fn conv_frame() -> ConvFrame {
        let mut f = ConvFrame { active: true, mouth: [0.8, 0.0, 0.0, 0.0, 0.2], lids: 0.3, ..Default::default() };
        f.offsets.0[4] = [0.2, 0.0, 0.0];
        f.expr = [0.0, 0.0, 0.0, 0.4, 0.0];
        f
    }

    fn talking_blender() -> PoseBlender {
        PoseBlender::new(PoseMap::new(), "idle", "neutral", 0)
    }

    #[test]
    fn conversation_frame_reaches_mouth_bones_and_lids() {
        let mut b = talking_blender();
        let base = b.evaluate(false);
        assert!(!base.expressions.contains_key("aa"), "no lip sync without a conversation");
        b.set_conversation(conv_frame());
        let e = b.evaluate(false);
        assert!((e.expressions["aa"] - 0.8).abs() < 1e-5);
        assert!((e.expressions["oh"] - 0.2).abs() < 1e-5);
        assert!(e.expressions["relaxed"] >= 0.4);
        assert!(e.expressions["blink"] >= 0.3, "lid floor");
        assert!((e.pose["head"][0] - base.pose["head"][0] - 0.2).abs() < 1e-4);
    }

    #[test]
    fn inactive_conversation_frames_are_ignored() {
        let mut b = talking_blender();
        let base = b.evaluate(false);
        b.set_conversation(ConvFrame { active: false, ..conv_frame() });
        let e = b.evaluate(false);
        assert_eq!(e.pose["head"], base.pose["head"]);
        assert!(!e.expressions.contains_key("aa"));
    }

    #[test]
    fn a_big_smile_keeps_most_of_the_mouth_shape() {
        let mut b = PoseBlender::new(PoseMap::new(), "idle", "happy", 0);
        b.set_conversation(conv_frame());
        let aa = b.evaluate(false).expressions["aa"];
        assert!(aa >= 0.8 * 0.6 && aa < 0.8, "{aa}");
    }

    #[test]
    fn ambient_drowsiness_softens_the_face_and_yields_to_cursor_attention() {
        let mut blender = PoseBlender::new(PoseMap::new(), "idle", "neutral", 0);
        let mut input = BlenderInput {
            dt_sec: 1.0 / 60.0,
            target_pose: "idle",
            target_expression: "neutral",
            ambient_drowsiness: 1.0,
            ..Default::default()
        };
        for _ in 0..75 {
            blender.step(input);
        }
        assert!(blender.evaluate(false).expressions["relaxed"] > 0.25);
        input.cursor_attention = true;
        blender.step(input);
        assert!(blender.evaluate(false).expressions["relaxed"] > 0.15);
        for _ in 0..45 {
            blender.step(input);
        }
        assert!(blender.evaluate(false).expressions["relaxed"] < 0.02);
    }

    #[test]
    fn worried_mood_keeps_posture_and_tension_after_the_entry_beat() {
        let mut blender = PoseBlender::new(PoseMap::new(), "think", "sad", 0);
        let worried = BlenderInput {
            dt_sec: 1.0 / 60.0,
            target_pose: "think",
            target_expression: "sad",
            harness_mood: Some(HarnessMood::Worried),
            ..Default::default()
        };
        for _ in 0..240 {
            blender.step(worried);
        }
        let eval = blender.evaluate(false);
        assert!(eval.expressions["sad"] > 0.7);
        assert!(eval.expressions["angry"] > 0.10, "sustained brow tension");
        assert!(eval.pose["head"][2] < -0.05, "sustained head tilt");
        let clear = BlenderInput { harness_mood: None, ..worried };
        blender.step(clear);
        assert!(blender.evaluate(false).expressions["angry"] > 0.10);
        for _ in 0..60 {
            blender.step(clear);
        }
        assert!(blender.evaluate(false).expressions["angry"] < 0.005);
    }

    #[test]
    fn landing_has_a_single_upper_body_compression_and_rebound() {
        let mut blender = PoseBlender::new(PoseMap::new(), "fall", "neutral", 0);
        let input = BlenderInput {
            dt_sec: 1.0 / 60.0,
            target_pose: "land",
            target_expression: "neutral",
            ..Default::default()
        };
        let mut peak = 0.0_f32;
        let mut rebound = 0.0_f32;
        for _ in 0..26 {
            blender.step(input);
            let pitch = blender.life.offset("chest")[0];
            peak = peak.max(pitch);
            rebound = rebound.min(pitch);
        }
        assert!(peak > 0.07, "landing compression: {peak}");
        assert!(rebound < -0.015, "landing rebound: {rebound}");
        for _ in 0..90 {
            blender.step(input);
            assert!(blender.life.offset("chest")[0].abs() < 0.012);
        }
        assert_eq!(blender.life.offset("hips"), [0.0; 3]);
        assert_eq!(blender.life.offset("leftFoot"), [0.0; 3]);
    }

    #[test]
    fn looking_away_fades_when_another_animation_takes_over() {
        let mut blender = PoseBlender::new(PoseMap::new(), "idle", "neutral", 0);
        let mut input = BlenderInput {
            dt_sec: 1.0 / 60.0,
            target_pose: "idle",
            target_expression: "neutral",
            head_yaw: 0.25,
            eyes: [0.0, 0.08],
            cursor_attention: true,
            ..Default::default()
        };
        for _ in 0..30 {
            blender.step(input);
        }
        let previous = blender.evaluate(false);
        assert!(previous.pose["head"][1] > 0.2);
        assert!(previous.pose["leftEye"][1] > 0.07);
        input.head_yaw = 0.0;
        input.eyes = [0.0; 2];
        input.eyes_closed = true;
        blender.step(input);
        let first = blender.evaluate(false);
        assert!(first.pose["head"][1] > previous.pose["head"][1] - 0.075);
        assert!(first.pose["leftEye"][1] > 0.03);
        for _ in 0..45 {
            blender.step(input);
        }
        assert!(blender.evaluate(false).pose["head"][1].abs() < 0.02);
        assert!(!blender.evaluate(false).pose.contains_key("leftEye"));
    }

    #[test]
    fn pose_blender_transitions_smoothly_without_single_frame_snaps() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../poses.json");
        let poses = load_poses(&path).expect("poses.json should load");
        let mut blender = PoseBlender::new(poses, "idle", "neutral", 0);

        let sequence: &[(&'static str, &'static str, bool, f32, i8, bool, usize)] = &[
            ("idle", "neutral", false, 0.0, 0, false, 15),
            ("idle", "neutral", true, 1.2, 1, false, 30),   // start walking
            ("wave", "happy", false, 0.0, 0, false, 30),    // interrupt walk -> wave
            ("sit", "relaxed", false, 0.0, 0, false, 30),   // wave -> sit
            ("fall", "surprised", false, 0.0, 0, false, 20),// sit -> fall
            ("land", "relaxed", false, 0.0, 0, false, 20),  // fall -> land
            ("sleep", "relaxed", false, 0.0, 0, true, 25),  // land -> sleep (eyes close smoothly)
            ("stretch", "relaxed", false, 0.0, 0, false, 25), // sleep -> stretch
            ("idle", "neutral", false, 0.0, 0, false, 25),  // stretch -> idle
        ];

        let mut prev_pose = blender.evaluate(false).pose;
        let mut prev_blink = blender.evaluate(false).expressions["blink"];
        let mut max_bone_step = 0.0f32;
        let mut max_bone_info = String::new();
        let mut max_blink_step = 0.0f32;

        for &(pose, expr, walking, dx, facing, eyes_closed, frames) in sequence {
            for f in 0..frames {
                blender.step(BlenderInput {
                    dt_sec: 1.0 / 60.0,
                    target_pose: pose,
                    target_expression: expr,
                    walking,
                    walk_dx_px: dx,
                    walk_speed_factor: if walking { 1.0 } else { 0.0 },
                    scale: 1.0,
                    facing,
                    eyes_closed,
                    carry_sway: 0.0,
                    head_yaw: 0.0,
                    head_pitch: 0.0,
                    neck_yaw: 0.0,
                    neck_pitch: 0.0,
                    ..Default::default()
                });
                let eval = blender.evaluate(false);
                for k in prev_pose.keys().chain(eval.pose.keys()) {
                    let a = prev_pose.get(k).copied().unwrap_or([0.0; 3]);
                    let b = eval.pose.get(k).copied().unwrap_or([0.0; 3]);
                    for i in 0..3 {
                        let d = (b[i] - a[i]).abs();
                        if d > max_bone_step {
                            max_bone_step = d;
                            max_bone_info = format!("pose={pose} frame={f} bone={k}[{i}] {a:?} -> {b:?}");
                        }
                    }
                }
                let blink = eval.expressions.get("blink").copied().unwrap_or(0.0);
                max_blink_step = max_blink_step.max((blink - prev_blink).abs());
                prev_pose = eval.pose;
                prev_blink = blink;
            }
        }

        assert!(
            max_bone_step < 0.30,
            "state transitions must blend smoothly without jumps, max single-frame bone delta = {max_bone_step:.4} rad ({max_bone_info})"
        );
        assert!(
            max_blink_step < 0.35,
            "eyelid transitions must stay bounded without snaps, max blink delta = {max_blink_step:.4}"
        );

        // Specifically verify sleep eyes_closed ramps smoothly across multiple frames (never 0 -> 1 in 1 frame)
        let mut sleep_blender = PoseBlender::new(load_poses(&path).unwrap(), "idle", "neutral", 0);
        sleep_blender.step(BlenderInput {
            dt_sec: 1.0 / 60.0,
            target_pose: "sleep",
            target_expression: "relaxed",
            eyes_closed: true,
            scale: 1.0,
            ..Default::default()
        });
        let b1 = sleep_blender.evaluate(false).expressions["blink"];
        assert!(b1 < 0.5, "sleep eyes must close gradually, first frame blink={b1}");
        for _ in 0..20 {
            sleep_blender.step(BlenderInput {
                dt_sec: 1.0 / 60.0,
                target_pose: "sleep",
                target_expression: "relaxed",
                eyes_closed: true,
                scale: 1.0,
                ..Default::default()
            });
        }
        let b_done = sleep_blender.evaluate(false).expressions["blink"];
        assert!(b_done > 0.98, "sleep eyes must reach fully closed, got {b_done}");
    }
}

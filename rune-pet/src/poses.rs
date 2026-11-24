//! Pure humanoid VRM pose interpolation, idle breathing, grounded biomechanical walk cycle,
//! live pose secondary motion, and auto-blink.

use std::collections::{BTreeMap, BTreeSet};
use std::f32::consts::{PI, TAU};
use std::path::Path;

pub type Pose = BTreeMap<String, [f32; 3]>;
pub type PoseMap = BTreeMap<String, Pose>;

pub const HIP_SWING_RAD: f32 = 0.30;

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

/// Live secondary motion for non-walk poses so `wave`, `cheer`, and `sit` feel alive instead of static.
pub fn compute_pose_secondary_motion(pose_name: &str, time_sec: f32) -> Pose {
    let mut out = Pose::new();
    match pose_name {
        "wave" => {
            let w = (time_sec * 6.8).sin();
            out.insert("rightUpperArm".into(), [0.0, 0.0, w * 0.08]);
            out.insert("rightLowerArm".into(), [0.0, w * 0.16, w * 0.24]);
            out.insert("rightHand".into(), [0.0, 0.0, w * 0.18]);
            out.insert("head".into(), [0.0, 0.0, -w * 0.04]);
        }
        "cheer" => {
            let bounce = (time_sec * 5.2).sin();
            out.insert("leftUpperArm".into(), [0.0, 0.0, bounce * 0.07]);
            out.insert("rightUpperArm".into(), [0.0, 0.0, -bounce * 0.07]);
            out.insert("chest".into(), [-bounce.abs() * 0.03, 0.0, 0.0]);
            out.insert("head".into(), [-bounce.abs() * 0.03, 0.0, bounce * 0.03]);
        }
        "sit" => {
            // Gentle relaxed dangling leg swing while perched on a window top edge.
            let swing = (time_sec * 2.1).sin();
            out.insert("leftLowerLeg".into(), [swing * 0.14, 0.0, 0.0]);
            out.insert("rightLowerLeg".into(), [-swing * 0.14, 0.0, 0.0]);
            out.insert("leftFoot".into(), [0.12 + swing * 0.06, 0.0, 0.0]);
            out.insert("rightFoot".into(), [0.12 - swing * 0.06, 0.0, 0.0]);
        }
        _ => {}
    }
    out
}

pub fn compute_auto_blink(time_sec: f32, period_sec: f32, blink_duration_sec: f32) -> f32 {
    if time_sec < 0.0 || period_sec <= 0.0 || blink_duration_sec <= 0.0 {
        return 0.0;
    }
    let phase = time_sec % period_sec;
    if phase > blink_duration_sec {
        return 0.0;
    }
    ((phase / blink_duration_sec) * PI).sin()
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
    fn auto_blink_peaks_at_half_duration_and_stays_zero_outside() {
        assert!((compute_auto_blink(0.08, 3.6, 0.16) - 1.0).abs() < 1e-5);
        assert_eq!(compute_auto_blink(1.0, 3.6, 0.16), 0.0);
        assert_eq!(compute_auto_blink(-0.5, 3.6, 0.16), 0.0);
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
        for name in ["idle", "wave", "cheer", "sit", "bow", "point", "think"] {
            assert!(poses.contains_key(name), "missing pose {name}");
        }
        assert!(poses["cheer"].contains_key("leftUpperArm"));
        assert!(poses["cheer"].contains_key("rightUpperArm"));
    }
}

//! Pure humanoid VRM pose interpolation, idle breathing, natural walk stride, and auto-blink.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub type Pose = BTreeMap<String, [f32; 3]>;
pub type PoseMap = BTreeMap<String, Pose>;

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

pub fn compute_idle_breathing(time_sec: f32) -> Pose {
    let breath = (time_sec * 1.6).sin();
    let sway = (time_sec * 0.75).sin();
    let mut out = Pose::new();
    out.insert("spine".into(), [breath * 0.014, sway * 0.008, 0.0]);
    out.insert("chest".into(), [breath * 0.02, 0.0, sway * 0.006]);
    out.insert("head".into(), [-breath * 0.012, sway * 0.015, 0.0]);
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
    ((phase / blink_duration_sec) * std::f32::consts::PI).sin()
}

/// Natural bipedal walk cycle tied to horizontal stride phase:
/// opposite legs & arms swing in phase, trailing knees flex naturally, elbows soften on forward
/// swing, and torso adds a subtle counter-twist + forward lean.
pub fn compute_walk_offsets(walk_phase: f32) -> (Pose, f32) {
    let swing = walk_phase.sin();
    let knee_l = (walk_phase + 0.65).sin().max(0.0) * 0.56;
    let knee_r = (-(walk_phase + 0.65).sin()).max(0.0) * 0.56;
    let elbow_l = (-swing).max(0.0) * 0.26;
    let elbow_r = swing.max(0.0) * 0.26;

    let mut offsets = Pose::new();
    offsets.insert("leftUpperLeg".into(), [swing * 0.44, 0.0, 0.0]);
    offsets.insert("rightUpperLeg".into(), [-swing * 0.44, 0.0, 0.0]);
    offsets.insert("leftLowerLeg".into(), [knee_l, 0.0, 0.0]);
    offsets.insert("rightLowerLeg".into(), [knee_r, 0.0, 0.0]);
    offsets.insert("leftFoot".into(), [-knee_l * 0.28, 0.0, 0.0]);
    offsets.insert("rightFoot".into(), [-knee_r * 0.28, 0.0, 0.0]);
    offsets.insert("leftUpperArm".into(), [-swing * 0.30, 0.0, 0.0]);
    offsets.insert("rightUpperArm".into(), [swing * 0.30, 0.0, 0.0]);
    offsets.insert("leftLowerArm".into(), [-elbow_l, 0.0, 0.0]);
    offsets.insert("rightLowerArm".into(), [-elbow_r, 0.0, 0.0]);
    offsets.insert("hips".into(), [0.02, swing * 0.055, swing * 0.015]);
    offsets.insert("spine".into(), [0.035, -swing * 0.04, 0.0]);
    offsets.insert("chest".into(), [0.015, -swing * 0.035, 0.0]);
    let bob_y = walk_phase.sin().abs() * 0.024;
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
        let (off, bob) = compute_walk_offsets(1.3);
        assert!(off["leftUpperLeg"][0] * off["rightUpperLeg"][0] < 0.0);
        assert!(off["leftUpperArm"][0] * off["leftUpperLeg"][0] < 0.0);
        assert!(off["leftLowerLeg"][0] >= 0.0 && off["rightLowerLeg"][0] >= 0.0);
        assert!(bob >= 0.0 && bob <= 0.03);
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

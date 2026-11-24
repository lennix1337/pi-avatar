//! Pose-aware 2D silhouette hit-testing for per-avatar click-through (transparent margins pass
//! clicks through to underlying desktop windows, while the avatar body captures hover/click/drag).

use std::collections::HashMap;

use glam::{Mat4, Vec2, Vec3, Vec4};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClickZone {
    Head,
    Body,
    Feet,
}

#[derive(Clone, Copy, Debug)]
pub struct Capsule2D {
    pub a: Vec2,
    pub b: Vec2,
    pub radius: f32,
    pub zone: ClickZone,
}

#[derive(Clone, Debug, Default)]
pub struct AvatarHitMask {
    pub capsules: Vec<Capsule2D>,
    pub bubble_rect: Option<[f32; 4]>,
    pub aspect: f32,
    pub head_center_norm: Option<Vec2>,
}

fn project(view_proj: Mat4, p: Vec3) -> Option<Vec2> {
    let clip = view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
    if clip.w.abs() < 1e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5))
}

fn dist_sq_point_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq <= 1e-8 {
        return (p - a).length_squared();
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length_squared()
}

impl AvatarHitMask {
    pub fn from_world_bones(
        bone_positions: &[(&str, Vec3)],
        view_proj: Mat4,
        aspect: f32,
        bubble_rect: Option<[f32; 4]>,
    ) -> Self {
        let mut map: HashMap<&str, Vec2> = HashMap::with_capacity(bone_positions.len());
        let mut head_center_norm = None;
        for &(name, pos) in bone_positions {
            if let Some(p2) = project(view_proj, pos) {
                if name == "head" {
                    // Offset slightly upward to center on face/cranium
                    head_center_norm = Some(Vec2::new(p2.x, (p2.y - 0.045).clamp(0.0, 1.0)));
                }
                map.insert(name, p2);
            }
        }

        let mut capsules = Vec::with_capacity(14);
        // Head + hair halo capsule
        if let Some(hc) = head_center_norm {
            capsules.push(Capsule2D {
                a: hc + Vec2::new(0.0, -0.02),
                b: hc + Vec2::new(0.0, 0.045),
                radius: 0.085,
                zone: ClickZone::Head,
            });
        }
        let mut push_seg = |a_name: &str, b_name: &str, radius: f32, zone: ClickZone| {
            if let (Some(&a), Some(&b)) = (map.get(a_name), map.get(b_name)) {
                capsules.push(Capsule2D { a, b, radius, zone });
            }
        };

        // Torso / shirt / shorts
        push_seg("neck", "hips", 0.095, ClickZone::Body);
        push_seg("chest", "hips", 0.105, ClickZone::Body);
        // Arms & hands
        push_seg("leftUpperArm", "leftLowerArm", 0.052, ClickZone::Body);
        push_seg("leftLowerArm", "leftHand", 0.048, ClickZone::Body);
        push_seg("rightUpperArm", "rightLowerArm", 0.052, ClickZone::Body);
        push_seg("rightLowerArm", "rightHand", 0.048, ClickZone::Body);
        // Legs & shoes
        push_seg("leftUpperLeg", "leftLowerLeg", 0.055, ClickZone::Body);
        push_seg("leftLowerLeg", "leftFoot", 0.052, ClickZone::Feet);
        push_seg("rightUpperLeg", "rightLowerLeg", 0.055, ClickZone::Body);
        push_seg("rightLowerLeg", "rightFoot", 0.052, ClickZone::Feet);

        Self {
            capsules,
            bubble_rect,
            aspect: aspect.max(0.2),
            head_center_norm,
        }
    }

    pub fn hit_zone(&self, nx: f32, ny: f32) -> Option<ClickZone> {
        if !(0.0..=1.0).contains(&nx) || !(0.0..=1.0).contains(&ny) {
            return None;
        }
        let scale = Vec2::new(self.aspect.max(0.2), 1.0);
        let p = Vec2::new(nx, ny) * scale;
        let mut best: Option<ClickZone> = None;
        for c in &self.capsules {
            let a = c.a * scale;
            let b = c.b * scale;
            if dist_sq_point_segment(p, a, b) <= c.radius * c.radius {
                best = match (best, c.zone) {
                    (Some(ClickZone::Head), _) | (_, ClickZone::Head) => Some(ClickZone::Head),
                    (Some(ClickZone::Body), _) | (_, ClickZone::Body) => Some(ClickZone::Body),
                    (_, ClickZone::Feet) => Some(ClickZone::Feet),
                };
            }
        }
        if best.is_some() {
            return best;
        }
        if let Some([x0, y0, x1, y1]) = self.bubble_rect
            && nx >= x0
            && nx <= x1
            && ny >= y0
            && ny <= y1
        {
            return Some(ClickZone::Head);
        }
        None
    }

    pub fn hit_test(&self, nx: f32, ny: f32) -> bool {
        self.hit_zone(nx, ny).is_some()
    }

    pub fn head_center(&self) -> Option<Vec2> {
        self.head_center_norm
    }
}

#[derive(Clone, Debug, Default)]
pub struct ClickThroughController {
    click_through: Option<bool>,
}

impl ClickThroughController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply<E>(
        &mut self,
        cursor_norm: Option<(f32, f32)>,
        dragging: bool,
        mask: &AvatarHitMask,
        set: impl FnOnce(bool) -> Result<(), E>,
    ) -> Result<(), E> {
        let previous = self.click_through;
        if let Some(desired) = self.update(cursor_norm, dragging, mask)
            && let Err(error) = set(desired)
        {
            self.click_through = previous;
            return Err(error);
        }
        Ok(())
    }

    /// Returns `Some(new_click_through)` only when the Win32 `WS_EX_TRANSPARENT` state must change.
    pub fn update(
        &mut self,
        cursor_norm: Option<(f32, f32)>,
        dragging: bool,
        mask: &AvatarHitMask,
    ) -> Option<bool> {
        let desired_click_through = if dragging {
            false
        } else {
            match cursor_norm {
                Some((nx, ny)) => !mask.hit_test(nx, ny),
                None => true,
            }
        };
        if self.click_through == Some(desired_click_through) {
            None
        } else {
            self.click_through = Some(desired_click_through);
            Some(desired_click_through)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poses::load_poses;
    use crate::vrm::{RigFrame, VrmModel};
    use std::path::PathBuf;

    fn sample_mask(vrm: &VrmModel, pose: &crate::poses::Pose) -> AvatarHitMask {
        let aspect = 220.0 / 280.0;
        let vp = crate::renderer::pet_camera_view_proj(aspect);
        let mut frame = RigFrame::default();
        vrm.evaluate_frame(pose, 0.0, 0.0, &mut frame);
        AvatarHitMask::from_world_bones(&frame.hit_bones, vp, aspect, None)
    }

    #[test]
    fn idle_pose_hits_body_and_misses_transparent_corners() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let vrm = VrmModel::load(&root.join("models/avatar.vrm")).unwrap();
        let poses = load_poses(&root.join("poses.json")).unwrap();
        let mask = sample_mask(&vrm, &poses["idle"]);

        assert!(mask.hit_test(0.5, 0.52), "torso center should hit");
        assert!(mask.hit_test(0.5, 0.30), "head should hit");
        assert!(!mask.hit_test(0.05, 0.05), "top-left corner must be click-through");
        assert!(!mask.hit_test(0.95, 0.05), "top-right corner must be click-through");
        assert!(!mask.hit_test(0.05, 0.92), "bottom-left corner must be click-through");
        assert!(!mask.hit_test(0.95, 0.92), "bottom-right corner must be click-through");
    }

    #[test]
    fn failed_click_through_change_is_retried() {
        let mut ctrl = ClickThroughController::new();
        let mask = AvatarHitMask::default();
        assert!(ctrl.apply(None, false, &mask, |_| Err("failed")).is_err());
        let mut calls = 0;
        ctrl.apply(None, false, &mask, |desired| {
            calls += 1;
            assert!(desired);
            Ok::<_, &str>(())
        }).unwrap();
        assert_eq!(calls, 1, "failed change must be retried");
    }

    #[test]
    fn click_through_toggles_cleanly_and_never_drops_drag() {
        let mut mask = AvatarHitMask {
            capsules: vec![Capsule2D {
                a: Vec2::new(0.5, 0.3),
                b: Vec2::new(0.5, 0.8),
                radius: 0.12,
                zone: ClickZone::Body,
            }],
            bubble_rect: None,
            aspect: 220.0 / 280.0,
            head_center_norm: Some(Vec2::new(0.5, 0.3)),
        };
        let mut ctrl = ClickThroughController::new();
        assert_eq!(ctrl.update(Some((0.05, 0.05)), false, &mask), Some(true));
        assert_eq!(ctrl.update(Some((0.06, 0.06)), false, &mask), None);
        assert_eq!(ctrl.update(Some((0.5, 0.5)), false, &mask), Some(false));
        assert_eq!(ctrl.update(Some((0.02, 0.02)), true, &mask), None);
        assert_eq!(ctrl.update(Some((0.02, 0.02)), false, &mask), Some(true));
        // Speech bubble rect also accepts hits
        mask.bubble_rect = Some([0.1, 0.02, 0.9, 0.14]);
        assert_eq!(ctrl.update(Some((0.5, 0.08)), false, &mask), Some(false));
    }

    #[test]
    fn hit_zone_classifies_head_body_feet_and_keeps_corners_click_through() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let vrm = VrmModel::load(&root.join("models/avatar.vrm")).unwrap();
        let poses = load_poses(&root.join("poses.json")).unwrap();
        let mask = sample_mask(&vrm, &poses["idle"]);

        let head = mask.head_center().expect("head center present");
        assert_eq!(mask.hit_zone(head.x, head.y), Some(ClickZone::Head));
        assert_eq!(mask.hit_zone(0.5, 0.52), Some(ClickZone::Body));
        assert_eq!(mask.hit_zone(0.5, 0.88), Some(ClickZone::Feet));
        assert_eq!(mask.hit_zone(0.05, 0.05), None);
        assert_eq!(mask.hit_zone(0.95, 0.92), None);

        // Verify click-through outside avatar silhouette remains intact across all new poses
        for pose_name in ["carry", "fall", "land", "jump", "stretch", "sleep", "sleep_sit", "react"] {
            let m = sample_mask(&vrm, &poses[pose_name]);
            assert!(!m.hit_test(0.04, 0.04), "{pose_name}: top-left must stay click-through");
            assert!(!m.hit_test(0.96, 0.04), "{pose_name}: top-right must stay click-through");
            assert!(!m.hit_test(0.04, 0.95), "{pose_name}: bottom-left must stay click-through");
            assert!(!m.hit_test(0.96, 0.95), "{pose_name}: bottom-right must stay click-through");
            assert!(m.hit_test(0.5, 0.50), "{pose_name}: torso center must remain interactive");
        }
    }
}

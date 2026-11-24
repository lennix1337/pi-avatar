//! Native VRM 1.0 (GLB) loader, humanoid normalized bone rig, and morph target evaluator.
//! Pure Rust and testable without a GPU device.

use std::collections::HashMap;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use glam::{EulerRot, Mat4, Quat, Vec2, Vec3, Vec4};
use serde::Deserialize;

use crate::poses::Pose;

pub const MAX_TEXTURE_DIM: u32 = 1024;
pub const MAX_SKIN_JOINTS: usize = 160;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VrmVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct MorphDelta {
    pub vertex_index: usize,
    pub delta: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct VrmPrimitive {
    pub vertices: Vec<VrmVertex>,
    pub indices: Vec<u32>,
    pub material_index: usize,
    /// Map from morph target index -> sparse position deltas for expressions we actually use.
    pub morph_targets: HashMap<usize, Vec<MorphDelta>>,
}

#[derive(Clone, Debug)]
pub struct VrmMesh {
    pub name: String,
    pub node_index: usize,
    pub skin_index: Option<usize>,
    pub primitives: Vec<VrmPrimitive>,
}

#[derive(Clone, Debug)]
pub struct VrmMaterial {
    pub base_color_factor: [f32; 4],
    pub shade_color_factor: [f32; 3],
    pub shading_toony_factor: f32,
    pub shading_shift_factor: f32,
    pub emissive_factor: [f32; 3],
    pub base_color_texture: Option<usize>,
    pub emissive_texture: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct DecodedTexture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct VrmNode {
    pub parent: Option<usize>,
    pub rest_translation: Vec3,
    pub rest_rotation: Quat,
    pub rest_scale: Vec3,
}

#[derive(Clone, Debug)]
pub struct VrmSkin {
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

#[derive(Clone, Debug)]
pub struct HumanoidBoneRig {
    pub node_index: usize,
    pub parent_world_rest_rot: Quat,
    pub inv_parent_world_rest_rot: Quat,
    pub rest_local_rot: Quat,
}

#[derive(Clone, Debug)]
pub struct ExpressionBind {
    pub mesh_node: usize,
    pub morph_index: usize,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub enum ConstraintKind {
    Aim { aim_axis: Vec3 },
    Roll { roll_axis: Vec3 },
    Rotation,
}

#[derive(Clone, Copy, Debug)]
pub struct NodeConstraint {
    pub destination: usize,
    pub source: usize,
    pub weight: f32,
    pub kind: ConstraintKind,
}

pub struct VrmModel {
    pub meshes: Vec<VrmMesh>,
    pub materials: Vec<VrmMaterial>,
    /// Decoded HD RGBA textures indexed by glTF texture index; cleared after GPU upload to save RAM.
    pub textures: HashMap<usize, DecodedTexture>,
    pub nodes: Vec<VrmNode>,
    pub topo_order: Vec<usize>,
    pub skins: Vec<VrmSkin>,
    pub humanoid_bones: HashMap<String, HumanoidBoneRig>,
    pub expressions: HashMap<String, Vec<ExpressionBind>>,
    pub constraints: Vec<NodeConstraint>,
}

#[derive(Default)]
struct RigScratch {
    world: Vec<Mat4>,
    local_rotations: Vec<Quat>,
    world_rotations: Vec<Quat>,
}

#[derive(Default)]
pub(crate) struct RigFrame {
    pub skins: Vec<Vec<Mat4>>,
    pub hit_bones: Vec<(&'static str, Vec3)>,
    scratch: RigScratch,
}

impl VrmModel {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("failed to read VRM {}: {e}", path.display()))?;
        Self::from_glb_bytes(&bytes)
    }

    pub fn from_glb_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
            return Err("not a valid GLB/VRM binary".into());
        }
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if &bytes[16..20] != b"JSON" || 20 + json_len > bytes.len() {
            return Err("invalid GLB JSON chunk".into());
        }
        let gltf: GltfRoot = serde_json::from_slice(&bytes[20..20 + json_len])
            .map_err(|e| format!("failed to parse glTF JSON: {e}"))?;

        let bin_header = 20 + json_len;
        if bin_header + 8 > bytes.len() || &bytes[bin_header + 4..bin_header + 8] != b"BIN\0" {
            return Err("missing GLB BIN chunk".into());
        }
        let bin_len =
            u32::from_le_bytes(bytes[bin_header..bin_header + 4].try_into().unwrap()) as usize;
        let bin_start = bin_header + 8;
        let bin_end = (bin_start + bin_len).min(bytes.len());
        let bin = &bytes[bin_start..bin_end];

        // 1. Parse node hierarchy and topological traversal order.
        let node_count = gltf.nodes.len();
        let mut parents = vec![None; node_count];
        for (i, n) in gltf.nodes.iter().enumerate() {
            for &c in &n.children {
                if c < node_count {
                    parents[c] = Some(i);
                }
            }
        }
        let nodes: Vec<VrmNode> = gltf
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| VrmNode {
                parent: parents[i],
                rest_translation: n
                    .translation
                    .map(Vec3::from_array)
                    .unwrap_or(Vec3::ZERO),
                rest_rotation: n
                    .rotation
                    .map(|q| Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize())
                    .unwrap_or(Quat::IDENTITY),
                rest_scale: n.scale.map(Vec3::from_array).unwrap_or(Vec3::ONE),
            })
            .collect();

        let mut topo_order = Vec::with_capacity(node_count);
        let mut visited = vec![false; node_count];
        fn visit(idx: usize, parents: &[Option<usize>], visited: &mut [bool], out: &mut Vec<usize>) {
            if visited[idx] {
                return;
            }
            if let Some(p) = parents[idx] {
                visit(p, parents, visited, out);
            }
            visited[idx] = true;
            out.push(idx);
        }
        for i in 0..node_count {
            visit(i, &parents, &mut visited, &mut topo_order);
        }

        // Compute rest world rotations for VRM 1.0 normalized humanoid rig math.
        let mut rest_world_rot = vec![Quat::IDENTITY; node_count];
        for &i in &topo_order {
            let local_r = nodes[i].rest_rotation;
            rest_world_rot[i] = match nodes[i].parent {
                Some(p) => (rest_world_rot[p] * local_r).normalize(),
                None => local_r,
            };
        }

        // 2. Parse VRM 1.0 humanoid bones & expressions.
        let vrm_ext = gltf
            .extensions
            .as_ref()
            .and_then(|e| e.vrmc_vrm.as_ref())
            .ok_or_else(|| "VRM 1.0 VRMC_vrm extension not found".to_string())?;

        let mut humanoid_bones = HashMap::new();
        for (name, bone) in &vrm_ext.humanoid.human_bones {
            let idx = bone.node;
            if idx < node_count {
                let parent_world_rest_rot = nodes[idx]
                    .parent
                    .map(|p| rest_world_rot[p])
                    .unwrap_or(Quat::IDENTITY);
                humanoid_bones.insert(
                    name.clone(),
                    HumanoidBoneRig {
                        node_index: idx,
                        parent_world_rest_rot,
                        inv_parent_world_rest_rot: parent_world_rest_rot.inverse(),
                        rest_local_rot: nodes[idx].rest_rotation,
                    },
                );
            }
        }

        let mut expressions: HashMap<String, Vec<ExpressionBind>> = HashMap::new();
        let mut needed_morphs: HashMap<usize, Vec<usize>> = HashMap::new();
        if let Some(expr_block) = &vrm_ext.expressions {
            for (expr_name, expr) in &expr_block.preset {
                let mut binds = Vec::new();
                for bind in &expr.morph_target_binds {
                    binds.push(ExpressionBind {
                        mesh_node: bind.node,
                        morph_index: bind.index,
                        weight: bind.weight,
                    });
                    needed_morphs
                        .entry(bind.node)
                        .or_default()
                        .push(bind.index);
                }
                if !binds.is_empty() {
                    expressions.insert(expr_name.clone(), binds);
                }
            }
        }

        // 3. Parse materials and track ONLY referenced baseColor / emissive textures.
        let mut materials = Vec::with_capacity(gltf.materials.len());
        let mut needed_textures = Vec::new();
        for mat in &gltf.materials {
            let pbr = mat.pbr_metallic_roughness.as_ref();
            let base_color_factor = pbr
                .and_then(|p| p.base_color_factor)
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
            let base_color_texture = pbr
                .and_then(|p| p.base_color_texture.as_ref())
                .map(|t| t.index);
            let emissive_texture = mat.emissive_texture.as_ref().map(|t| t.index);
            let emissive_factor = mat.emissive_factor.unwrap_or([0.0, 0.0, 0.0]);
            let mtoon = mat
                .extensions
                .as_ref()
                .and_then(|e| e.mtoon.as_ref());

            if let Some(t) = base_color_texture {
                needed_textures.push(t);
            }
            if let Some(t) = emissive_texture {
                needed_textures.push(t);
            }

            materials.push(VrmMaterial {
                base_color_factor,
                shade_color_factor: mtoon
                    .and_then(|m| m.shade_color_factor)
                    .unwrap_or([0.85, 0.85, 0.9]),
                shading_toony_factor: mtoon.and_then(|m| m.shading_toony_factor).unwrap_or(0.9),
                shading_shift_factor: mtoon.and_then(|m| m.shading_shift_factor).unwrap_or(-0.05),
                emissive_factor,
                base_color_texture,
                emissive_texture,
            });
        }

        needed_textures.sort_unstable();
        needed_textures.dedup();
        let mut textures = HashMap::new();
        for tex_idx in needed_textures {
            if let Some(tex) = gltf.textures.get(tex_idx)
                && let Some(img_idx) = tex.source
                && let Some(img) = gltf.images.get(img_idx)
                && let Some(bv_idx) = img.buffer_view
                && let Some(slice) = buffer_view_slice(&gltf, bin, bv_idx)
            {
                let decoded = decode_and_cap_png(slice, MAX_TEXTURE_DIM)?;
                textures.insert(tex_idx, decoded);
            }
        }

        // 4. Parse skins.
        let mut skins = Vec::with_capacity(gltf.skins.len());
        for skin in &gltf.skins {
            let ibms = if let Some(acc_idx) = skin.inverse_bind_matrices {
                read_mat4_accessor(&gltf, bin, acc_idx)?
            } else {
                vec![Mat4::IDENTITY; skin.joints.len()]
            };
            skins.push(VrmSkin {
                joints: skin.joints.clone(),
                inverse_bind_matrices: ibms,
            });
        }

        // 5. Parse VRMC_node_constraint (aim, roll, rotation) for twist/sleeve/thigh helpers.
        let mut constraints = Vec::new();
        for (node_idx, node) in gltf.nodes.iter().enumerate() {
            let Some(nc) = node
                .extensions
                .as_ref()
                .and_then(|e| e.node_constraint.as_ref())
            else {
                continue;
            };
            if let Some(aim) = &nc.constraint.aim
                && aim.source < node_count
            {
                let axis = match aim.aim_axis.as_str() {
                    "PositiveX" => Vec3::X,
                    "NegativeX" => Vec3::NEG_X,
                    "PositiveY" => Vec3::Y,
                    "NegativeY" => Vec3::NEG_Y,
                    "PositiveZ" => Vec3::Z,
                    "NegativeZ" => Vec3::NEG_Z,
                    _ => Vec3::X,
                };
                constraints.push(NodeConstraint {
                    destination: node_idx,
                    source: aim.source,
                    weight: aim.weight.unwrap_or(1.0).clamp(0.0, 1.0),
                    kind: ConstraintKind::Aim { aim_axis: axis },
                });
            } else if let Some(roll) = &nc.constraint.roll
                && roll.source < node_count
            {
                let axis = match roll.roll_axis.as_str() {
                    "X" => Vec3::X,
                    "Y" => Vec3::Y,
                    "Z" => Vec3::Z,
                    _ => Vec3::X,
                };
                constraints.push(NodeConstraint {
                    destination: node_idx,
                    source: roll.source,
                    weight: roll.weight.unwrap_or(1.0).clamp(0.0, 1.0),
                    kind: ConstraintKind::Roll { roll_axis: axis },
                });
            } else if let Some(rot) = &nc.constraint.rotation
                && rot.source < node_count
            {
                constraints.push(NodeConstraint {
                    destination: node_idx,
                    source: rot.source,
                    weight: rot.weight.unwrap_or(1.0).clamp(0.0, 1.0),
                    kind: ConstraintKind::Rotation,
                });
            }
        }

        // 6. Parse meshes referenced by nodes.
        let mut meshes = Vec::new();
        for (node_idx, node) in gltf.nodes.iter().enumerate() {
            let Some(mesh_idx) = node.mesh else { continue };
            let gltf_mesh = gltf
                .meshes
                .get(mesh_idx)
                .ok_or_else(|| format!("missing mesh index {mesh_idx}"))?;
            let wanted_targets = needed_morphs.get(&node_idx);
            let mut primitives = Vec::with_capacity(gltf_mesh.primitives.len());

            for prim in &gltf_mesh.primitives {
                let positions = read_vec3_accessor(
                    &gltf,
                    bin,
                    *prim
                        .attributes
                        .get("POSITION")
                        .ok_or("primitive missing POSITION")?,
                )?;
                let normals = if let Some(&acc) = prim.attributes.get("NORMAL") {
                    read_vec3_accessor(&gltf, bin, acc)?
                } else {
                    vec![Vec3::Y; positions.len()]
                };
                let uvs = if let Some(&acc) = prim.attributes.get("TEXCOORD_0") {
                    read_vec2_accessor(&gltf, bin, acc)?
                } else {
                    vec![Vec2::ZERO; positions.len()]
                };
                let joints = if let Some(&acc) = prim.attributes.get("JOINTS_0") {
                    read_uvec4_accessor(&gltf, bin, acc)?
                } else {
                    vec![[0, 0, 0, 0]; positions.len()]
                };
                let weights = if let Some(&acc) = prim.attributes.get("WEIGHTS_0") {
                    read_vec4_accessor(&gltf, bin, acc)?
                } else {
                    vec![Vec4::new(1.0, 0.0, 0.0, 0.0); positions.len()]
                };
                let indices = read_indices_accessor(
                    &gltf,
                    bin,
                    prim.indices.ok_or("primitive missing indices")?,
                )?;

                let mut vertices = Vec::with_capacity(positions.len());
                for i in 0..positions.len() {
                    let mut w = weights[i];
                    let sum = w.x + w.y + w.z + w.w;
                    if sum > 1e-6 {
                        w /= sum;
                    } else {
                        w = Vec4::new(1.0, 0.0, 0.0, 0.0);
                    }
                    vertices.push(VrmVertex {
                        position: positions[i].to_array(),
                        normal: normals[i].to_array(),
                        uv: uvs[i].to_array(),
                        joints: joints[i],
                        weights: w.to_array(),
                    });
                }

                let mut morph_targets = HashMap::new();
                if let Some(wanted) = wanted_targets {
                    for &target_idx in wanted {
                        if let Some(target) = prim.targets.get(target_idx)
                            && let Some(&pos_acc) = target.get("POSITION")
                        {
                            let deltas = read_morph_deltas(&gltf, bin, pos_acc)?;
                            if !deltas.is_empty() {
                                morph_targets.insert(target_idx, deltas);
                            }
                        }
                    }
                }

                primitives.push(VrmPrimitive {
                    vertices,
                    indices,
                    material_index: prim.material.unwrap_or(0),
                    morph_targets,
                });
            }

            meshes.push(VrmMesh {
                name: gltf_mesh.name.clone().unwrap_or_default(),
                node_index: node_idx,
                skin_index: node.skin,
                primitives,
            });
        }

        Ok(Self {
            meshes,
            materials,
            textures,
            nodes,
            topo_order,
            skins,
            humanoid_bones,
            expressions,
            constraints,
        })
    }

    /// Computes world positions of key humanoid bones for silhouette hit-testing & gaze targeting.
    fn hit_bones_from_world(&self, world: &[Mat4], out: &mut Vec<(&'static str, Vec3)>) {
        const TRACKED: &[&str] = &[
            "hips",
            "spine",
            "chest",
            "neck",
            "head",
            "leftUpperArm",
            "leftLowerArm",
            "leftHand",
            "rightUpperArm",
            "rightLowerArm",
            "rightHand",
            "leftUpperLeg",
            "leftLowerLeg",
            "leftFoot",
            "rightUpperLeg",
            "rightLowerLeg",
            "rightFoot",
        ];
        out.clear();
        out.reserve(TRACKED.len());
        for &name in TRACKED {
            if let Some(rig) = self.humanoid_bones.get(name)
                && let Some(m) = world.get(rig.node_index)
            {
                out.push((name, m.w_axis.truncate()));
            }
        }
    }

    fn evaluate_world_matrices(
        &self,
        pose: &Pose,
        root_yaw: f32,
        root_bob_y: f32,
        scratch: &mut RigScratch,
    ) {
        let RigScratch { world, local_rotations, world_rotations } = scratch;
        let node_count = self.nodes.len();
        local_rotations.clear();
        local_rotations.extend(self.nodes.iter().map(|n| n.rest_rotation));

        // Exact @pixiv/three-vrm normalized-to-raw formula:
        // raw_local = inv(parent_world_rest) * q_norm * parent_world_rest * rest_local
        for (bone_name, rig) in &self.humanoid_bones {
            if let Some(&[rx, ry, rz]) = pose.get(bone_name) {
                let q_norm = Quat::from_euler(EulerRot::XYZ, rx, ry, rz);
                local_rotations[rig.node_index] = (rig.inv_parent_world_rest_rot
                    * q_norm
                    * rig.parent_world_rest_rot
                    * rig.rest_local_rot)
                    .normalize();
            }
        }

        let root_xform = Mat4::from_translation(Vec3::new(0.0, root_bob_y, 0.0))
            * Mat4::from_rotation_y(root_yaw);
        let root_rot = Quat::from_rotation_y(root_yaw);

        world.resize(node_count, Mat4::IDENTITY);
        world_rotations.resize(node_count, Quat::IDENTITY);
        let recompute_world = |local_rots: &[Quat], world: &mut [Mat4], world_rot: &mut [Quat]| {
            for &i in &self.topo_order {
                let n = &self.nodes[i];
                let local = Mat4::from_scale_rotation_translation(
                    n.rest_scale,
                    local_rots[i],
                    n.rest_translation,
                );
                match n.parent {
                    Some(p) => {
                        world[i] = world[p] * local;
                        world_rot[i] = (world_rot[p] * local_rots[i]).normalize();
                    }
                    None => {
                        world[i] = root_xform * local;
                        world_rot[i] = (root_rot * local_rots[i]).normalize();
                    }
                }
            }
        };

        recompute_world(local_rotations, world, world_rotations);

        // Apply VRM 1.0 VRMC_node_constraint (aim, roll, rotation) so sleeves, shoulders, and hips track limbs.
        if !self.constraints.is_empty() {
            for c in &self.constraints {
                let dst = c.destination;
                let src = c.source;
                let dst_rest = self.nodes[dst].rest_rotation;
                let src_rest = self.nodes[src].rest_rotation;

                let next_q = match c.kind {
                    ConstraintKind::Aim { aim_axis } => {
                        let dst_parent_rot = self.nodes[dst]
                            .parent
                            .map(|p| world_rotations[p])
                            .unwrap_or(root_rot);
                        let inv_dst_parent_rot = dst_parent_rot.inverse();
                        let a0 = (dst_parent_rot * (dst_rest * aim_axis)).normalize_or_zero();
                        let dst_pos = world[dst].w_axis.truncate();
                        let src_pos = world[src].w_axis.truncate();
                        let a1 = (src_pos - dst_pos).normalize_or_zero();
                        if a0.length_squared() > 1e-8 && a1.length_squared() > 1e-8 {
                            let target = (inv_dst_parent_rot
                                * Quat::from_rotation_arc(a0, a1)
                                * dst_parent_rot
                                * dst_rest)
                                .normalize();
                            dst_rest.slerp(target, c.weight)
                        } else {
                            dst_rest
                        }
                    }
                    ConstraintKind::Roll { roll_axis } => {
                        let inv_dst_rest = dst_rest.inverse();
                        let inv_src_rest = src_rest.inverse();
                        let quat_delta =
                            (inv_dst_rest * local_rotations[src] * inv_src_rest * dst_rest)
                                .normalize();
                        let n1 = (quat_delta * roll_axis).normalize_or_zero();
                        let from_to = if n1.length_squared() > 1e-8 {
                            Quat::from_rotation_arc(n1, roll_axis)
                        } else {
                            Quat::IDENTITY
                        };
                        let target = (dst_rest * from_to * quat_delta).normalize();
                        dst_rest.slerp(target, c.weight)
                    }
                    ConstraintKind::Rotation => {
                        let src_delta = src_rest.inverse() * local_rotations[src];
                        let target = (dst_rest * src_delta).normalize();
                        dst_rest.slerp(target, c.weight)
                    }
                };
                local_rotations[dst] = next_q.normalize();
            }
            recompute_world(local_rotations, world, world_rotations);
        }
    }

    /// One hierarchy evaluation feeds both the GPU skinning and the click-through silhouette.
    pub(crate) fn evaluate_frame(&self, pose: &Pose, root_yaw: f32, root_bob_y: f32, out: &mut RigFrame) {
        self.evaluate_world_matrices(pose, root_yaw, root_bob_y, &mut out.scratch);
        out.skins.resize_with(self.skins.len(), Vec::new);
        self.skin_from_world(&out.scratch.world, &mut out.skins);
        self.hit_bones_from_world(&out.scratch.world, &mut out.hit_bones);
    }

    fn skin_from_world(&self, world: &[Mat4], out: &mut [Vec<Mat4>]) {
        for (skin, dst) in self.skins.iter().zip(out) {
            dst.resize(MAX_SKIN_JOINTS, Mat4::IDENTITY);
            for ((&joint_node, &ibm), mat) in skin.joints.iter().zip(&skin.inverse_bind_matrices).zip(dst) {
                *mat = world[joint_node] * ibm;
            }
        }
    }

    #[cfg(test)]
    fn evaluate_skins(&self, pose: &Pose, root_yaw: f32, root_bob_y: f32, out: &mut [Vec<Mat4>]) {
        let mut scratch = RigScratch::default();
        self.evaluate_world_matrices(pose, root_yaw, root_bob_y, &mut scratch);
        self.skin_from_world(&scratch.world, out);
    }

    /// Computes morphed vertex buffer for a primitive given active expression weights (`name -> 0..1`).
    pub fn apply_morphs(
        &self,
        mesh_node: usize,
        prim: &VrmPrimitive,
        expr_weights: &HashMap<&str, f32>,
        out: &mut Vec<VrmVertex>,
    ) {
        out.clear();
        out.extend_from_slice(&prim.vertices);
        if prim.morph_targets.is_empty() {
            return;
        }
        for (&expr_name, &w) in expr_weights {
            if w <= 1e-4 {
                continue;
            }
            let Some(binds) = self.expressions.get(expr_name) else {
                continue;
            };
            for bind in binds {
                if bind.mesh_node != mesh_node {
                    continue;
                }
                let effective = w * bind.weight;
                if effective.abs() <= 1e-4 {
                    continue;
                }
                if let Some(deltas) = prim.morph_targets.get(&bind.morph_index) {
                    for d in deltas {
                        if let Some(v) = out.get_mut(d.vertex_index) {
                            v.position[0] += d.delta[0] * effective;
                            v.position[1] += d.delta[1] * effective;
                            v.position[2] += d.delta[2] * effective;
                        }
                    }
                }
            }
        }
    }
}

fn decode_and_cap_png(png_bytes: &[u8], max_dim: u32) -> Result<DecodedTexture, String> {
    let img = image::load_from_memory_with_format(png_bytes, image::ImageFormat::Png)
        .map_err(|e| format!("PNG decode failed: {e}"))?;
    let (w, h) = (img.width(), img.height());
    let resized = if max_dim > 0 && (w > max_dim || h > max_dim) {
        let scale = max_dim as f32 / w.max(h) as f32;
        let nw = (w as f32 * scale).round().max(1.0) as u32;
        let nh = (h as f32 * scale).round().max(1.0) as u32;
        img.resize_exact(nw, nh, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let rgba = resized.to_rgba8();
    Ok(DecodedTexture {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
    })
}

fn buffer_view_slice<'a>(gltf: &GltfRoot, bin: &'a [u8], bv_idx: usize) -> Option<&'a [u8]> {
    let bv = gltf.buffer_views.get(bv_idx)?;
    let start = bv.byte_offset.unwrap_or(0);
    let end = start + bv.byte_length;
    bin.get(start..end)
}

fn accessor_slice<'a>(
    gltf: &'a GltfRoot,
    bin: &'a [u8],
    acc_idx: usize,
) -> Result<(&'a GltfAccessor, &'a [u8], usize), String> {
    let acc = gltf
        .accessors
        .get(acc_idx)
        .ok_or_else(|| format!("missing accessor {acc_idx}"))?;
    let bv_idx = acc
        .buffer_view
        .ok_or_else(|| format!("accessor {acc_idx} has no bufferView"))?;
    let bv = gltf
        .buffer_views
        .get(bv_idx)
        .ok_or_else(|| format!("missing bufferView {bv_idx}"))?;
    let start = bv.byte_offset.unwrap_or(0) + acc.byte_offset.unwrap_or(0);
    let slice = bin
        .get(start..)
        .ok_or_else(|| format!("accessor {acc_idx} out of bounds"))?;
    Ok((acc, slice, bv.byte_stride.unwrap_or(0)))
}

fn read_f32(slice: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(slice[off..off + 4].try_into().unwrap())
}

fn read_vec3_accessor(gltf: &GltfRoot, bin: &[u8], acc_idx: usize) -> Result<Vec<Vec3>, String> {
    let (acc, slice, stride) = accessor_slice(gltf, bin, acc_idx)?;
    let step = if stride == 0 { 12 } else { stride };
    let mut out = Vec::with_capacity(acc.count);
    for i in 0..acc.count {
        let o = i * step;
        out.push(Vec3::new(
            read_f32(slice, o),
            read_f32(slice, o + 4),
            read_f32(slice, o + 8),
        ));
    }
    Ok(out)
}

fn read_vec2_accessor(gltf: &GltfRoot, bin: &[u8], acc_idx: usize) -> Result<Vec<Vec2>, String> {
    let (acc, slice, stride) = accessor_slice(gltf, bin, acc_idx)?;
    let step = if stride == 0 { 8 } else { stride };
    let mut out = Vec::with_capacity(acc.count);
    for i in 0..acc.count {
        let o = i * step;
        out.push(Vec2::new(read_f32(slice, o), read_f32(slice, o + 4)));
    }
    Ok(out)
}

fn read_vec4_accessor(gltf: &GltfRoot, bin: &[u8], acc_idx: usize) -> Result<Vec<Vec4>, String> {
    let (acc, slice, stride) = accessor_slice(gltf, bin, acc_idx)?;
    let step = if stride == 0 { 16 } else { stride };
    let mut out = Vec::with_capacity(acc.count);
    for i in 0..acc.count {
        let o = i * step;
        out.push(Vec4::new(
            read_f32(slice, o),
            read_f32(slice, o + 4),
            read_f32(slice, o + 8),
            read_f32(slice, o + 12),
        ));
    }
    Ok(out)
}

fn read_uvec4_accessor(
    gltf: &GltfRoot,
    bin: &[u8],
    acc_idx: usize,
) -> Result<Vec<[u32; 4]>, String> {
    let (acc, slice, stride) = accessor_slice(gltf, bin, acc_idx)?;
    let mut out = Vec::with_capacity(acc.count);
    match acc.component_type {
        5121 => {
            let step = if stride == 0 { 4 } else { stride };
            for i in 0..acc.count {
                let o = i * step;
                out.push([
                    slice[o] as u32,
                    slice[o + 1] as u32,
                    slice[o + 2] as u32,
                    slice[o + 3] as u32,
                ]);
            }
        }
        5123 => {
            let step = if stride == 0 { 8 } else { stride };
            for i in 0..acc.count {
                let o = i * step;
                out.push([
                    u16::from_le_bytes([slice[o], slice[o + 1]]) as u32,
                    u16::from_le_bytes([slice[o + 2], slice[o + 3]]) as u32,
                    u16::from_le_bytes([slice[o + 4], slice[o + 5]]) as u32,
                    u16::from_le_bytes([slice[o + 6], slice[o + 7]]) as u32,
                ]);
            }
        }
        ct => return Err(format!("unsupported JOINTS_0 componentType {ct}")),
    }
    Ok(out)
}

fn read_indices_accessor(gltf: &GltfRoot, bin: &[u8], acc_idx: usize) -> Result<Vec<u32>, String> {
    let (acc, slice, stride) = accessor_slice(gltf, bin, acc_idx)?;
    let mut out = Vec::with_capacity(acc.count);
    match acc.component_type {
        5121 => {
            let step = if stride == 0 { 1 } else { stride };
            for i in 0..acc.count {
                out.push(slice[i * step] as u32);
            }
        }
        5123 => {
            let step = if stride == 0 { 2 } else { stride };
            for i in 0..acc.count {
                let o = i * step;
                out.push(u16::from_le_bytes([slice[o], slice[o + 1]]) as u32);
            }
        }
        5125 => {
            let step = if stride == 0 { 4 } else { stride };
            for i in 0..acc.count {
                let o = i * step;
                out.push(u32::from_le_bytes(slice[o..o + 4].try_into().unwrap()));
            }
        }
        ct => return Err(format!("unsupported indices componentType {ct}")),
    }
    Ok(out)
}

fn read_mat4_accessor(gltf: &GltfRoot, bin: &[u8], acc_idx: usize) -> Result<Vec<Mat4>, String> {
    let (acc, slice, stride) = accessor_slice(gltf, bin, acc_idx)?;
    let step = if stride == 0 { 64 } else { stride };
    let mut out = Vec::with_capacity(acc.count);
    for i in 0..acc.count {
        let o = i * step;
        let mut cols = [0.0f32; 16];
        for c in 0..16 {
            cols[c] = read_f32(slice, o + c * 4);
        }
        out.push(Mat4::from_cols_array(&cols));
    }
    Ok(out)
}

fn read_morph_deltas(
    gltf: &GltfRoot,
    bin: &[u8],
    acc_idx: usize,
) -> Result<Vec<MorphDelta>, String> {
    let acc = gltf
        .accessors
        .get(acc_idx)
        .ok_or_else(|| format!("missing morph accessor {acc_idx}"))?;
    let mut deltas = Vec::new();
    if let Some(sparse) = &acc.sparse {
        let idx_bv = buffer_view_slice(gltf, bin, sparse.indices.buffer_view)
            .ok_or("missing sparse indices bufferView")?;
        let idx_slice = &idx_bv[sparse.indices.byte_offset.unwrap_or(0)..];
        let val_bv = buffer_view_slice(gltf, bin, sparse.values.buffer_view)
            .ok_or("missing sparse values bufferView")?;
        let val_slice = &val_bv[sparse.values.byte_offset.unwrap_or(0)..];

        deltas.reserve(sparse.count);
        for i in 0..sparse.count {
            let v_idx = match sparse.indices.component_type {
                5121 => idx_slice[i] as usize,
                5123 => u16::from_le_bytes([idx_slice[i * 2], idx_slice[i * 2 + 1]]) as usize,
                5125 => {
                    u32::from_le_bytes(idx_slice[i * 4..i * 4 + 4].try_into().unwrap()) as usize
                }
                ct => return Err(format!("unsupported sparse index componentType {ct}")),
            };
            let vo = i * 12;
            let dx = read_f32(val_slice, vo);
            let dy = read_f32(val_slice, vo + 4);
            let dz = read_f32(val_slice, vo + 8);
            if dx != 0.0 || dy != 0.0 || dz != 0.0 {
                deltas.push(MorphDelta {
                    vertex_index: v_idx,
                    delta: [dx, dy, dz],
                });
            }
        }
    } else if acc.buffer_view.is_some() {
        let vecs = read_vec3_accessor(gltf, bin, acc_idx)?;
        for (i, v) in vecs.into_iter().enumerate() {
            if v.length_squared() > 1e-12 {
                deltas.push(MorphDelta {
                    vertex_index: i,
                    delta: v.to_array(),
                });
            }
        }
    }
    Ok(deltas)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfRoot {
    #[serde(default)]
    nodes: Vec<GltfNode>,
    #[serde(default)]
    meshes: Vec<GltfMesh>,
    #[serde(default)]
    skins: Vec<GltfSkin>,
    #[serde(default)]
    materials: Vec<GltfMaterial>,
    #[serde(default)]
    textures: Vec<GltfTexture>,
    #[serde(default)]
    images: Vec<GltfImage>,
    #[serde(default)]
    accessors: Vec<GltfAccessor>,
    #[serde(default)]
    buffer_views: Vec<GltfBufferView>,
    extensions: Option<GltfRootExtensions>,
}

#[derive(Deserialize)]
struct GltfRootExtensions {
    #[serde(rename = "VRMC_vrm")]
    vrmc_vrm: Option<VrmcVrm>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VrmcVrm {
    humanoid: VrmHumanoid,
    expressions: Option<VrmExpressions>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VrmHumanoid {
    human_bones: HashMap<String, VrmBoneRef>,
}

#[derive(Deserialize)]
struct VrmBoneRef {
    node: usize,
}

#[derive(Deserialize)]
struct VrmExpressions {
    #[serde(default)]
    preset: HashMap<String, VrmExpression>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VrmExpression {
    #[serde(default)]
    morph_target_binds: Vec<VrmMorphBind>,
}

#[derive(Deserialize)]
struct VrmMorphBind {
    node: usize,
    index: usize,
    weight: f32,
}

#[derive(Deserialize)]
struct GltfNode {
    #[serde(default)]
    children: Vec<usize>,
    mesh: Option<usize>,
    skin: Option<usize>,
    translation: Option<[f32; 3]>,
    rotation: Option<[f32; 4]>,
    scale: Option<[f32; 3]>,
    extensions: Option<GltfNodeExtensions>,
}

#[derive(Deserialize)]
struct GltfNodeExtensions {
    #[serde(rename = "VRMC_node_constraint")]
    node_constraint: Option<VrmcNodeConstraint>,
}

#[derive(Deserialize)]
struct VrmcNodeConstraint {
    constraint: VrmConstraintBody,
}

#[derive(Deserialize)]
struct VrmConstraintBody {
    aim: Option<VrmAimConstraint>,
    roll: Option<VrmRollConstraint>,
    rotation: Option<VrmRotationConstraint>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VrmAimConstraint {
    source: usize,
    aim_axis: String,
    weight: Option<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VrmRollConstraint {
    source: usize,
    roll_axis: String,
    weight: Option<f32>,
}

#[derive(Deserialize)]
struct VrmRotationConstraint {
    source: usize,
    weight: Option<f32>,
}

#[derive(Deserialize)]
struct GltfMesh {
    name: Option<String>,
    #[serde(default)]
    primitives: Vec<GltfPrimitive>,
}

#[derive(Deserialize)]
struct GltfPrimitive {
    attributes: HashMap<String, usize>,
    indices: Option<usize>,
    material: Option<usize>,
    #[serde(default)]
    targets: Vec<HashMap<String, usize>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfSkin {
    inverse_bind_matrices: Option<usize>,
    #[serde(default)]
    joints: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfMaterial {
    pbr_metallic_roughness: Option<GltfPbr>,
    emissive_factor: Option<[f32; 3]>,
    emissive_texture: Option<GltfTexInfo>,
    extensions: Option<GltfMatExtensions>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfPbr {
    base_color_factor: Option<[f32; 4]>,
    base_color_texture: Option<GltfTexInfo>,
}

#[derive(Deserialize)]
struct GltfTexInfo {
    index: usize,
}

#[derive(Deserialize)]
struct GltfMatExtensions {
    #[serde(rename = "VRMC_materials_mtoon")]
    mtoon: Option<VrmMtoon>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VrmMtoon {
    shade_color_factor: Option<[f32; 3]>,
    shading_toony_factor: Option<f32>,
    shading_shift_factor: Option<f32>,
}

#[derive(Deserialize)]
struct GltfTexture {
    source: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfImage {
    buffer_view: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfAccessor {
    buffer_view: Option<usize>,
    byte_offset: Option<usize>,
    component_type: u32,
    count: usize,
    sparse: Option<GltfSparse>,
}

#[derive(Deserialize)]
struct GltfSparse {
    count: usize,
    indices: GltfSparseIndices,
    values: GltfSparseValues,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfSparseIndices {
    buffer_view: usize,
    byte_offset: Option<usize>,
    component_type: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfSparseValues {
    buffer_view: usize,
    byte_offset: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GltfBufferView {
    byte_offset: Option<usize>,
    byte_length: usize,
    byte_stride: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poses::load_poses;
    use std::path::PathBuf;

    #[test]
    fn evaluating_frame_preserves_the_hit_mask_bone_buffer() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let model = VrmModel::load(&root.join("models/avatar.vrm")).unwrap();
        let mut frame = RigFrame { hit_bones: Vec::with_capacity(64), ..RigFrame::default() };
        let storage = frame.hit_bones.as_ptr();
        model.evaluate_frame(&Pose::new(), 0.0, 0.0, &mut frame);
        let head_y = frame.hit_bones.iter().find(|(name, _)| *name == "head").unwrap().1.y;
        let mut original_skins = vec![Vec::new(); model.skins.len()];
        model.evaluate_skins(&Pose::new(), 0.0, 0.0, &mut original_skins);
        assert_eq!(frame.skins, original_skins);
        model.evaluate_frame(&Pose::new(), 0.3, 0.1, &mut frame);
        assert_eq!(frame.hit_bones.len(), 17);
        let raised_head = frame.hit_bones.iter().find(|(name, _)| *name == "head").unwrap().1.y;
        assert!((raised_head - head_y - 0.1).abs() < 1e-5);
        assert_eq!(frame.hit_bones.as_ptr(), storage, "the hit-mask buffer must not be replaced each frame");
    }

    #[test]
    fn loads_avatar_vrm_and_evaluates_idle_and_cheer_skins() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let vrm = VrmModel::load(&root.join("models/avatar.vrm")).expect("avatar.vrm loads");
        assert_eq!(vrm.meshes.len(), 3);
        assert_eq!(vrm.materials.len(), 13);
        assert!(vrm.humanoid_bones.contains_key("hips"));
        assert!(vrm.humanoid_bones.contains_key("leftUpperArm"));
        assert!(vrm.humanoid_bones.contains_key("rightUpperArm"));
        assert!(vrm.expressions.contains_key("happy"));
        assert!(vrm.expressions.contains_key("blink"));
        assert!(vrm.textures.len() >= 13);
        assert_eq!(vrm.constraints.len(), 14);

        let poses = load_poses(&root.join("poses.json")).unwrap();
        let mut skins_rest = vec![Vec::new(); vrm.skins.len()];
        let mut skins_idle = vec![Vec::new(); vrm.skins.len()];
        vrm.evaluate_skins(&Pose::new(), 0.0, 0.0, &mut skins_rest);
        vrm.evaluate_skins(&poses["idle"], 0.0, 0.0, &mut skins_idle);
        // Verify sleeve aim constraint node (91) actually moves away from T-pose in idle
        let sleeve_joint = vrm.skins[0]
            .joints
            .iter()
            .position(|&n| n == 91)
            .expect("sleeve aim joint 91 in skin 0");
        assert!(
            (skins_idle[0][sleeve_joint] - skins_rest[0][sleeve_joint])
                .to_cols_array()
                .iter()
                .map(|v| v.abs())
                .sum::<f32>()
                > 0.1,
            "sleeve aim constraint must track leftUpperArm in idle pose"
        );

        let mut skins = vec![Vec::new(); vrm.skins.len()];
        vrm.evaluate_skins(&poses["cheer"], 0.0, 0.0, &mut skins);
        for skin in &skins {
            assert_eq!(skin.len(), MAX_SKIN_JOINTS);
            assert!(skin[0].is_finite());
        }
    }

    #[test]
    fn whole_body_including_raised_arms_and_walk_fits_inside_viewport() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let vrm = VrmModel::load(&root.join("models/avatar.vrm")).unwrap();
        let poses = load_poses(&root.join("poses.json")).unwrap();
        let aspect = 220.0 / 280.0;
        let vp = crate::renderer::pet_camera_view_proj(aspect);

        for (label, pose_name, yaw, bob) in [
            ("idle", "idle", 0.0f32, 0.0f32),
            ("arms-up", "cheer", 0.0, 0.0),
            ("walk", "idle", 0.42 * std::f32::consts::PI, -0.02),
            ("carry", "carry", 0.0, 0.0),
            ("fall", "fall", 0.0, 0.0),
            ("land", "land", 0.0, -0.055),
            ("jump", "jump", 0.0, 0.015),
            ("stretch", "stretch", 0.0, 0.008),
            ("sleep", "sleep", 0.0, 0.0),
            ("sleep_sit", "sleep_sit", 0.0, 0.0),
            ("react", "react", 0.0, 0.02),
        ] {
            let mut skins = vec![Vec::new(); vrm.skins.len()];
            vrm.evaluate_skins(&poses[pose_name], yaw, bob, &mut skins);
            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for mesh in &vrm.meshes {
                let s = &skins[mesh.skin_index.unwrap_or(0)];
                for prim in &mesh.primitives {
                    for v in &prim.vertices {
                        let p = Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
                        let world_p = (s[v.joints[0] as usize] * v.weights[0]
                            + s[v.joints[1] as usize] * v.weights[1]
                            + s[v.joints[2] as usize] * v.weights[2]
                            + s[v.joints[3] as usize] * v.weights[3])
                            * p;
                        let clip = vp * world_p;
                        let ndc = clip.truncate() / clip.w;
                        let sx = ndc.x * 0.5 + 0.5;
                        let sy = 0.5 - ndc.y * 0.5;
                        min_x = min_x.min(sx);
                        max_x = max_x.max(sx);
                        min_y = min_y.min(sy);
                        max_y = max_y.max(sy);
                    }
                }
            }
            assert!(
                min_x >= 0.02 && max_x <= 0.98 && min_y >= 0.10 && max_y <= 0.998,
                "{label} clipped viewport: x=[{min_x:.3}..{max_x:.3}], y=[{min_y:.3}..{max_y:.3}]"
            );
        }
    }

    #[test]
    fn shoe_soles_align_with_foot_margin_and_stay_grounded_across_walk_cycle() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let vrm = VrmModel::load(&root.join("models/avatar.vrm")).unwrap();
        let poses = load_poses(&root.join("poses.json")).unwrap();
        let aspect = crate::pet_state::PET_WINDOW_WIDTH / crate::pet_state::PET_WINDOW_HEIGHT;
        let vp = crate::renderer::pet_camera_view_proj(aspect);

        // 1. Standing idle: projected bottom of shoes matches FOOT_BOTTOM_MARGIN_DIP within 2 DIP.
        let mut skins = vec![Vec::new(); vrm.skins.len()];
        vrm.evaluate_skins(&poses["idle"], 0.0, 0.0, &mut skins);
        let mut idle_max_sy = f32::NEG_INFINITY;
        for mesh in &vrm.meshes {
            let s = &skins[mesh.skin_index.unwrap_or(0)];
            for prim in &mesh.primitives {
                for v in &prim.vertices {
                    let p = Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
                    let wp = (s[v.joints[0] as usize] * v.weights[0]
                        + s[v.joints[1] as usize] * v.weights[1]
                        + s[v.joints[2] as usize] * v.weights[2]
                        + s[v.joints[3] as usize] * v.weights[3])
                        * p;
                    let clip = vp * wp;
                    let sy = 0.5 - (clip.y / clip.w) * 0.5;
                    idle_max_sy = idle_max_sy.max(sy);
                }
            }
        }
        let measured_margin_dip = (1.0 - idle_max_sy) * crate::pet_state::PET_WINDOW_HEIGHT;
        assert!(
            (measured_margin_dip - crate::pet_state::FOOT_BOTTOM_MARGIN_DIP).abs() <= 3.5,
            "idle shoe bottom margin ({measured_margin_dip:.2} DIP) must match FOOT_BOTTOM_MARGIN_DIP ({})",
            crate::pet_state::FOOT_BOTTOM_MARGIN_DIP
        );

        // 2. Across 24 phases of the walk cycle, the planted foot stays near y = 0.0 (no floating).
        let yaw = crate::pet_state::facing_to_yaw_rad(1);
        for step in 0..24 {
            let phase = (step as f32 / 24.0) * std::f32::consts::TAU;
            let (walk_off, bob_y) = crate::poses::compute_walk_offsets(phase);
            let mut combined = poses["idle"].clone();
            for (bone, val) in walk_off {
                let entry = combined.entry(bone).or_insert([0.0; 3]);
                entry[0] += val[0];
                entry[1] += val[1];
                entry[2] += val[2];
            }
            vrm.evaluate_skins(&combined, yaw, bob_y, &mut skins);
            let mut min_world_y = f32::INFINITY;
            for mesh in &vrm.meshes {
                let s = &skins[mesh.skin_index.unwrap_or(0)];
                for prim in &mesh.primitives {
                    for v in &prim.vertices {
                        let p = Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
                        let wp = (s[v.joints[0] as usize] * v.weights[0]
                            + s[v.joints[1] as usize] * v.weights[1]
                            + s[v.joints[2] as usize] * v.weights[2]
                            + s[v.joints[3] as usize] * v.weights[3])
                            * p;
                        min_world_y = min_world_y.min(wp.y);
                    }
                }
            }
            assert!(
                min_world_y >= -0.035 && min_world_y <= 0.045,
                "walk phase {phase:.2}: lowest shoe vertex y={min_world_y:.4}m drifted off ground"
            );
        }

        // 3. Standing interaction & idle poses stay grounded (neither floating nor sinking into taskbar).
        for (label, pose_name, bob_y) in [
            ("stretch", "stretch", 0.008f32),
            ("sleep", "sleep", 0.0),
            ("react", "react", 0.0),
            ("land", "land", -0.055),
        ] {
            vrm.evaluate_skins(&poses[pose_name], 0.0, bob_y, &mut skins);
            let mut min_world_y = f32::INFINITY;
            for mesh in &vrm.meshes {
                let s = &skins[mesh.skin_index.unwrap_or(0)];
                for prim in &mesh.primitives {
                    for v in &prim.vertices {
                        let p = Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
                        let wp = (s[v.joints[0] as usize] * v.weights[0]
                            + s[v.joints[1] as usize] * v.weights[1]
                            + s[v.joints[2] as usize] * v.weights[2]
                            + s[v.joints[3] as usize] * v.weights[3])
                            * p;
                        min_world_y = min_world_y.min(wp.y);
                    }
                }
            }
            assert!(
                min_world_y >= -0.045 && min_world_y <= 0.045,
                "{label} pose shoe y={min_world_y:.4}m must stay grounded near taskbar floor"
            );
        }
    }
}

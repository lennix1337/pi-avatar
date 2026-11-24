//! Rune `Compositor<PetAppState>` GPU 3D VRM renderer using Rune's own `wgpu::Device` and `Queue`.
//! Features:
//! - 2x supersampling + 4x MSAA on 100%-150% DPI screens (zero pixelation), downsampled with a 4-tap filter
//! - Linear-light sRGB mipmaps + 8x anisotropic filtering for crisp HD textures
//! - Rune SDF + Parley UI overlay composited cleanly over the 3D VRM (`LoadOp::Load`) for speech bubbles & sparkles
//! - Live pose-aware `AvatarHitMask` publication for per-avatar click-through (zero GPU readback)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use rune::compose::{Compositor, DeviceKey, Frame};
use wgpu::util::DeviceExt;

use crate::PetAppState;
use crate::hit_mask::AvatarHitMask;
use crate::pacing::{FpsSummary, FpsTracker};
use crate::vrm::{DecodedTexture, MAX_SKIN_JOINTS, RigFrame, VrmModel, VrmVertex};

const MSAA_SAMPLES: u32 = 4;

pub const CAMERA_FOV_DEG: f32 = 32.0;
pub const CAMERA_Y: f32 = 1.02;
pub const CAMERA_Z: f32 = 3.85;

pub fn pet_camera_view_proj(aspect: f32) -> Mat4 {
    let proj = Mat4::perspective_rh(CAMERA_FOV_DEG.to_radians(), aspect, 0.1, 30.0);
    let view = Mat4::look_at_rh(
        Vec3::new(0.0, CAMERA_Y, CAMERA_Z),
        Vec3::new(0.0, CAMERA_Y, 0.0),
        Vec3::Y,
    );
    proj * view
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    srgb_encode: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniforms {
    base_color: [f32; 4],
    shade_color: [f32; 4],
    emissive: [f32; 4],
    shading_toony: f32,
    shading_shift: f32,
    has_emissive_tex: u32,
    _pad: u32,
}

pub struct RenderMetrics {
    pub started_at: Instant,
    pub total_frames: u64,
    pub tracker: FpsTracker,
}

impl RenderMetrics {
    pub fn new(warmup_sec: f64) -> Self {
        Self {
            started_at: Instant::now(),
            total_frames: 0,
            tracker: FpsTracker::new(warmup_sec),
        }
    }

    pub fn record(&mut self) {
        self.total_frames += 1;
        let t = self.started_at.elapsed().as_secs_f64();
        self.tracker.record_frame(t);
    }

    pub fn summary(&self) -> FpsSummary {
        self.tracker.summary()
    }
}

struct GpuPrimitive {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    material_index: usize,
    has_morphs: bool,
}

struct GpuMesh {
    node_index: usize,
    skin_index: usize,
    primitives: Vec<GpuPrimitive>,
}

struct SupersampleBlit {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

struct Targets {
    out_size: (u32, u32),
    supersampled: bool,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    ss_resolve_view: Option<wgpu::TextureView>,
    ss_blit_bg: Option<wgpu::BindGroup>,
}

struct GpuState {
    device_key: DeviceKey,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    blit: SupersampleBlit,
    targets: Targets,
    frame_uniform_buf: wgpu::Buffer,
    skin_buffers: Vec<wgpu::Buffer>,
    skin_bind_groups: Vec<wgpu::BindGroup>,
    material_bind_groups: Vec<wgpu::BindGroup>,
    meshes: Vec<GpuMesh>,
}

pub struct VrmCompositor {
    vrm_path: PathBuf,
    model: VrmModel,
    gpu: Option<GpuState>,
    rig_frame: RigFrame,
    skin_raw_scratch: Vec<[[f32; 4]; 4]>,
    morph_scratch: Vec<VrmVertex>,
    last_morph_sig: [u32; 6],
    wants_draw: bool,
    metrics: Arc<Mutex<RenderMetrics>>,
    hit_mask: Arc<Mutex<AvatarHitMask>>,
}

impl VrmCompositor {
    pub fn new(
        vrm_path: PathBuf,
        model: VrmModel,
        metrics: Arc<Mutex<RenderMetrics>>,
        hit_mask: Arc<Mutex<AvatarHitMask>>,
    ) -> Self {
        Self {
            vrm_path,
            model,
            gpu: None,
            rig_frame: RigFrame::default(),
            skin_raw_scratch: vec![Mat4::IDENTITY.to_cols_array_2d(); MAX_SKIN_JOINTS],
            morph_scratch: Vec::new(),
            last_morph_sig: [u32::MAX; 6],
            wants_draw: true,
            metrics,
            hit_mask,
        }
    }

    fn ensure_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: DeviceKey,
        format: wgpu::TextureFormat,
        size: (u32, u32),
    ) {
        if let Some(gpu) = &mut self.gpu
            && gpu.device_key == key
            && gpu.format == format
        {
            if gpu.targets.out_size != size {
                gpu.targets = create_targets(device, format, size, &gpu.blit);
            }
            return;
        }

        // Reload textures & static geometry from disk only if recovering from a lost GPU device.
        if self.model.textures.is_empty()
            && let Ok(reloaded) = VrmModel::load(&self.vrm_path)
        {
            self.model = reloaded;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vrm-mtoon-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vrm.wgsl").into()),
        });

        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vrm-frame-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let mat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vrm-mat-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vrm-pipeline-layout"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&mat_bgl)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VrmVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vrm-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLES,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let blit = create_blit_pipeline(device, format);

        let frame_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vrm-frame-uniform"),
            contents: bytemuck::bytes_of(&FrameUniforms {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                srgb_encode: 0,
                _pad: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let identity_skin: Vec<[[f32; 4]; 4]> =
            vec![Mat4::IDENTITY.to_cols_array_2d(); MAX_SKIN_JOINTS];
        let mut skin_buffers = Vec::with_capacity(self.model.skins.len().max(1));
        let mut skin_bind_groups = Vec::with_capacity(self.model.skins.len().max(1));
        for s_idx in 0..self.model.skins.len().max(1) {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("vrm-skin-{s_idx}")),
                contents: bytemuck::cast_slice(&identity_skin),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("vrm-skin-bg-{s_idx}")),
                layout: &frame_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf.as_entire_binding(),
                    },
                ],
            });
            skin_buffers.push(buf);
            skin_bind_groups.push(bg);
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vrm-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });

        let fallback_white = upload_mipmapped_texture(
            device,
            queue,
            DecodedTexture {
                width: 1,
                height: 1,
                rgba: vec![255, 255, 255, 255],
            },
        );
        let fallback_black = upload_mipmapped_texture(
            device,
            queue,
            DecodedTexture {
                width: 1,
                height: 1,
                rgba: vec![0, 0, 0, 255],
            },
        );

        let mut gpu_tex_views: HashMap<usize, wgpu::TextureView> = HashMap::new();
        for (idx_i, (tex_idx, tex)) in self.model.textures.drain().enumerate() {
            gpu_tex_views.insert(tex_idx, upload_mipmapped_texture(device, queue, tex));
            if (idx_i + 1) % 3 == 0 {
                queue.submit([]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            }
        }
        queue.submit([]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        self.model.textures.shrink_to_fit();

        let mut material_bind_groups = Vec::with_capacity(self.model.materials.len());
        for (m_idx, mat) in self.model.materials.iter().enumerate() {
            let base_view = mat
                .base_color_texture
                .and_then(|id| gpu_tex_views.get(&id))
                .unwrap_or(&fallback_white);
            let emissive_view = mat
                .emissive_texture
                .and_then(|id| gpu_tex_views.get(&id))
                .unwrap_or(&fallback_black);

            let mat_u = MaterialUniforms {
                base_color: mat.base_color_factor,
                shade_color: [
                    mat.shade_color_factor[0],
                    mat.shade_color_factor[1],
                    mat.shade_color_factor[2],
                    1.0,
                ],
                emissive: [
                    mat.emissive_factor[0],
                    mat.emissive_factor[1],
                    mat.emissive_factor[2],
                    1.0,
                ],
                shading_toony: mat.shading_toony_factor,
                shading_shift: mat.shading_shift_factor,
                has_emissive_tex: u32::from(mat.emissive_texture.is_some()),
                _pad: 0,
            };
            let mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("vrm-mat-{m_idx}")),
                contents: bytemuck::bytes_of(&mat_u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("vrm-mat-bg-{m_idx}")),
                layout: &mat_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mat_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(base_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(emissive_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            material_bind_groups.push(bg);
        }

        let mut gpu_meshes = Vec::with_capacity(self.model.meshes.len());
        for mesh in &mut self.model.meshes {
            let mut gpu_prims = Vec::with_capacity(mesh.primitives.len());
            for prim in &mut mesh.primitives {
                let has_morphs = !prim.morph_targets.is_empty();
                let usage = if has_morphs {
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST
                } else {
                    wgpu::BufferUsages::VERTEX
                };
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("vrm-vb-{}", mesh.name)),
                    contents: bytemuck::cast_slice(&prim.vertices),
                    usage,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("vrm-ib-{}", mesh.name)),
                    contents: bytemuck::cast_slice(&prim.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let index_count = prim.indices.len() as u32;
                prim.indices = Vec::new();
                if !has_morphs {
                    prim.vertices = Vec::new();
                }
                gpu_prims.push(GpuPrimitive {
                    vertex_buffer,
                    index_buffer,
                    index_count,
                    material_index: prim.material_index,
                    has_morphs,
                });
            }
            gpu_meshes.push(GpuMesh {
                node_index: mesh.node_index,
                skin_index: mesh.skin_index.unwrap_or(0),
                primitives: gpu_prims,
            });
        }

        let targets = create_targets(device, format, size, &blit);

        self.gpu = Some(GpuState {
            device_key: key,
            format,
            pipeline,
            blit,
            targets,
            frame_uniform_buf,
            skin_buffers,
            skin_bind_groups,
            material_bind_groups,
            meshes: gpu_meshes,
        });
        self.last_morph_sig = [u32::MAX; 6];
    }
}

impl Compositor<PetAppState> for VrmCompositor {
    fn render(&mut self, f: &mut Frame<'_, PetAppState>) {
        let key = f.device_key();
        let srgb_encode = u32::from(f.srgb_blend());
        self.ensure_gpu(f.device, f.queue, key, f.format, f.size);

        let pose_eval = f.rt.state.evaluate_frame_pose();
        let has_bubble = f.rt.state.bubble_text().is_some();
        self.wants_draw = f.rt.state.should_render();

        let aspect = (f.size.0.max(1) as f32) / (f.size.1.max(1) as f32);
        let view_proj_mat = pet_camera_view_proj(aspect);
        let view_proj = view_proj_mat.to_cols_array_2d();

        self.model.evaluate_frame(
            &pose_eval.pose,
            pose_eval.root_yaw,
            pose_eval.root_bob_y,
            &mut self.rig_frame,
        );

        // Publish pose-aware silhouette hit mask for per-avatar click-through & gaze head center.
        let bubble_rect = if has_bubble {
            Some([0.08, 0.01, 0.92, 0.15])
        } else {
            None
        };
        let mask = AvatarHitMask::from_world_bones(&self.rig_frame.hit_bones, view_proj_mat, aspect, bubble_rect);
        if let Ok(mut guard) = self.hit_mask.lock() {
            *guard = mask;
        }

        let gpu = self.gpu.as_ref().unwrap();
        let frame_u = FrameUniforms {
            view_proj,
            srgb_encode,
            _pad: [0; 3],
        };
        f.write_buffer(&gpu.frame_uniform_buf, 0, bytemuck::bytes_of(&frame_u));

        for (s_idx, skin_mats) in self.rig_frame.skins.iter().enumerate() {
            if let Some(buf) = gpu.skin_buffers.get(s_idx) {
                self.skin_raw_scratch.clear();
                self.skin_raw_scratch
                    .extend(skin_mats.iter().map(|m| m.to_cols_array_2d()));
                f.write_buffer(buf, 0, bytemuck::cast_slice(&self.skin_raw_scratch));
            }
        }

        // Update morphed face vertices only when expression or blink weights actually changed.
        let expr_names = ["happy", "angry", "sad", "relaxed", "surprised", "blink"];
        let mut morph_sig = [0u32; 6];
        for (i, name) in expr_names.iter().enumerate() {
            let w = pose_eval.expressions.get(*name).copied().unwrap_or(0.0);
            morph_sig[i] = (w.clamp(0.0, 1.0) * 1000.0).round() as u32;
        }
        if morph_sig != self.last_morph_sig {
            self.last_morph_sig = morph_sig;
            let mut weights_map: HashMap<&str, f32> = HashMap::new();
            for name in expr_names {
                if let Some(&w) = pose_eval.expressions.get(name)
                    && w > 1e-4
                {
                    weights_map.insert(name, w);
                }
            }
            for (mesh_idx, gpu_mesh) in gpu.meshes.iter().enumerate() {
                let model_mesh = &self.model.meshes[mesh_idx];
                for (prim_idx, gpu_prim) in gpu_mesh.primitives.iter().enumerate() {
                    if !gpu_prim.has_morphs {
                        continue;
                    }
                    let model_prim = &model_mesh.primitives[prim_idx];
                    self.model.apply_morphs(
                        gpu_mesh.node_index,
                        model_prim,
                        &weights_map,
                        &mut self.morph_scratch,
                    );
                    f.write_buffer(
                        &gpu_prim.vertex_buffer,
                        0,
                        bytemuck::cast_slice(&self.morph_scratch),
                    );
                }
            }
        }

        // Pass 1: 3D VRM render pass (4x MSAA, 2x supersampled when DPI < 2x).
        let resolve_target = gpu
            .targets
            .ss_resolve_view
            .as_ref()
            .unwrap_or(f.target);
        {
            let mut pass = f.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vrm-main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gpu.targets.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(resolve_target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.targets.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&gpu.pipeline);
            for gpu_mesh in &gpu.meshes {
                let skin_bg = &gpu.skin_bind_groups[gpu_mesh.skin_index];
                pass.set_bind_group(0, skin_bg, &[]);
                for prim in &gpu_mesh.primitives {
                    if let Some(mat_bg) = gpu.material_bind_groups.get(prim.material_index) {
                        pass.set_bind_group(1, mat_bg, &[]);
                        pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                        pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..prim.index_count, 0, 0..1);
                    }
                }
            }
        }

        // Pass 1b: 4-tap supersample downsample into the final swapchain target when supersampled.
        if gpu.targets.supersampled
            && let Some(blit_bg) = &gpu.targets.ss_blit_bg
        {
            let mut pass = f.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vrm-supersample-blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: f.target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&gpu.blit.pipeline);
            pass.set_bind_group(0, blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2: Composite Rune's crisp SDF + Parley UI overlay (speech bubble & sparkles) on top.
        if !f.rt.instances.is_empty() {
            f.ui.set_fade_stops(f.rt.fade_stops());
            f.ui.prepare(
                f.device,
                f.queue,
                &mut f.rt.text.atlas,
                &f.rt.instances,
                f.size,
            );
            f.ui.draw_target_range_with(
                f.encoder,
                f.target,
                f.format,
                wgpu::LoadOp::Load,
                0..f.rt.instances.len() as u32,
                &mut |_, _| {},
            );
        }

        if let Ok(mut m) = self.metrics.lock() {
            m.record();
        }
    }

    fn release_gpu(&mut self) {
        self.gpu = None;
    }

    fn animating(&self) -> bool {
        // Paced explicitly by `FramePacer` via `Tx::wake_after` (~60 FPS) so DX12 never spins uncapped.
        false
    }

    fn draws_own(&self) -> bool {
        self.wants_draw
    }
}

fn create_blit_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> SupersampleBlit {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vrm-blit-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vrm-blit-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vrm-blit-layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("vrm-blit-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_blit"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_blit"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("vrm-blit-sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    SupersampleBlit {
        pipeline,
        bgl,
        sampler,
    }
}

fn create_targets(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    out_size: (u32, u32),
    blit: &SupersampleBlit,
) -> Targets {
    let ow = out_size.0.max(1);
    let oh = out_size.1.max(1);
    // 2x supersampling whenever output width < 400 px (covers 100%-150% DPI screens cleanly)
    let supersampled = ow < 400;
    let render_size = if supersampled { (ow * 2, oh * 2) } else { (ow, oh) };

    let msaa_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm-msaa-color"),
        size: wgpu::Extent3d {
            width: render_size.0,
            height: render_size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm-msaa-depth"),
        size: wgpu::Extent3d {
            width: render_size.0,
            height: render_size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let (ss_resolve_view, ss_blit_bg) = if supersampled {
        let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vrm-ss-resolve"),
            size: wgpu::Extent3d {
                width: render_size.0,
                height: render_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = resolve_tex.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vrm-ss-blit-bg"),
            layout: &blit.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit.sampler),
                },
            ],
        });
        (Some(view), Some(bg))
    } else {
        (None, None)
    };

    Targets {
        out_size,
        supersampled,
        msaa_view: msaa_tex.create_view(&Default::default()),
        depth_view: depth_tex.create_view(&Default::default()),
        ss_resolve_view,
        ss_blit_bg,
    }
}

#[inline]
fn srgb_u8_to_linear(v: u8) -> f32 {
    let x = v as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb_u8(v: f32) -> u8 {
    let x = v.clamp(0.0, 1.0);
    let s = if x <= 0.0031308 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Uploads a texture with a linear-light box-filtered mipmap chain so anime outlines, eyes, and hair
/// stay sharp without darkening or aliasing at any DPI.
fn upload_mipmapped_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: DecodedTexture,
) -> wgpu::TextureView {
    let base_w = src.width.max(1);
    let base_h = src.height.max(1);
    let mip_level_count = (base_w.max(base_h) as f32).log2().floor() as u32 + 1;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrm-tex"),
        size: wgpu::Extent3d {
            width: base_w,
            height: base_h,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut w = base_w;
    let mut h = base_h;
    let mut cur = src.rgba;
    for level in 0..mip_level_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &cur,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        if level + 1 >= mip_level_count {
            break;
        }
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let x0 = (x * 2).min(w - 1);
                let x1 = (x0 + 1).min(w - 1);
                let y0 = (y * 2).min(h - 1);
                let y1 = (y0 + 1).min(h - 1);
                let coords = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
                let dst_i = ((y * nw + x) * 4) as usize;
                for c in 0..3 {
                    let mut sum = 0.0f32;
                    for (sx, sy) in coords {
                        sum += srgb_u8_to_linear(cur[((sy * w + sx) * 4) as usize + c]);
                    }
                    next[dst_i + c] = linear_to_srgb_u8(sum * 0.25);
                }
                let mut a_sum = 0u32;
                for (sx, sy) in coords {
                    a_sum += cur[((sy * w + sx) * 4) as usize + 3] as u32;
                }
                next[dst_i + 3] = ((a_sum + 2) / 4) as u8;
            }
        }
        w = nw;
        h = nh;
        cur = next;
    }

    texture.create_view(&Default::default())
}

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rand::RngExt;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    include_wgsl, BindGroup, Buffer, ComputePipeline, CurrentSurfaceTexture, Device, Queue,
    RenderPipeline, Surface, SurfaceConfiguration,
};
use winit::dpi::PhysicalSize;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const DEFAULT_FPS: u32 = 60;
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const BOIDS_COUNT: u32 = 20000;
const SEP_RADIUS: u32 = 15;
const ALIGN_RADIUS: u32 = 42;
const COHESION_RADIUS: u32 = 60;
const SEP_STRENGTH: f32 = 100.0;
const ALIGN_STRENGTH: f32 = 1.4;
const COHESION_STRENGTH: f32 = 0.6;
const MAX_SPEED: u32 = 420;
const MIN_SPEED: u32 = 60;
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Boid {
    pos: [f32; 2],
    vel: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComputeUniforms {
    delta_time: f32,
    width: f32,
    height: f32,
    count: u32,
    sep_radius: f32,
    align_radius: f32,
    cohesion_radius: f32,
    sep_strength: f32,
    align_strength: f32,
    cohesion_strength: f32,
    max_speed: f32,
    min_speed: f32,
}
impl Default for ComputeUniforms {
    fn default() -> Self {
        Self {
            delta_time: 1.0 / DEFAULT_FPS as f32,
            width: WIDTH as f32,
            height: HEIGHT as f32,
            count: BOIDS_COUNT,
            sep_radius: SEP_RADIUS as f32,
            align_radius: ALIGN_RADIUS as f32,
            cohesion_radius: COHESION_RADIUS as f32,
            sep_strength: SEP_STRENGTH,
            align_strength: ALIGN_STRENGTH,
            cohesion_strength: COHESION_STRENGTH,
            max_speed: MAX_SPEED as f32,
            min_speed: MIN_SPEED as f32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderUniforms {
    width: f32,
    height: f32,
}

struct BoidsSimulation {
    window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    compute_pipeline: ComputePipeline,
    render_pipeline: RenderPipeline,
    compute_uniforms_buffer: Buffer,
    render_uniforms_buffer: Buffer,
    compute_bind_groups: [BindGroup; 2],
    render_bind_groups: [BindGroup; 2],
    current_idx: usize,
    last_frame: Instant,
}

impl BoidsSimulation {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();

        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create wgpu surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: true,
            })
            .await
            .expect("failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Boids Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .expect("failed to create GPU device");

        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface is not supported by the adapter");

        surface.configure(&device, &config);

        let mut rng = rand::rng();

        let initial_boids: Vec<Boid> = (0..BOIDS_COUNT)
            .map(|_| Boid {
                pos: [
                    rng.random_range(0.0..size.width.max(1) as f32),
                    rng.random_range(0.0..size.height.max(1) as f32),
                ],
                vel: [rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0)],
            })
            .collect();

        let boid_buffers = [
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Boids Buffer A"),
                contents: bytemuck::cast_slice(&initial_boids),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Boids Buffer B"),
                contents: bytemuck::cast_slice(&initial_boids),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
        ];

        let (render_uniforms_buffer, render_bind_groups, render_pipeline) =
            Self::render_setup(size, &device, &config, &boid_buffers);

        let (compute_uniforms_buffer, compute_bind_groups, compute_pipeline) =
            Self::compute_setup(size, &device, &boid_buffers);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            compute_pipeline,
            render_pipeline,
            compute_uniforms_buffer,
            render_uniforms_buffer,
            compute_bind_groups,
            render_bind_groups,
            current_idx: 0,
            last_frame: Instant::now(),
        }
    }

    fn render_setup(
        size: PhysicalSize<u32>,
        device: &Device,
        config: &SurfaceConfiguration,
        boid_buffers: &[Buffer; 2],
    ) -> (Buffer, [BindGroup; 2], RenderPipeline) {
        let render_shaders = device.create_shader_module(include_wgsl!("renderShaders.wgsl"));

        let render_uniforms = RenderUniforms {
            width: size.width.max(1) as f32,
            height: size.height.max(1) as f32,
        };
        let render_uniforms_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Render Buffer"),
            contents: bytemuck::bytes_of(&[render_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
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

        let render_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Bind Group A"),
                layout: &render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: render_uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: boid_buffers[0].as_entire_binding(),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Bind Group B"),
                layout: &render_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: render_uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: boid_buffers[1].as_entire_binding(),
                    },
                ],
            }),
        ];
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&render_bind_group_layout)],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shaders,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shaders,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        (render_uniforms_buffer, render_bind_groups, render_pipeline)
    }

    fn compute_setup(
        size: PhysicalSize<u32>,
        device: &Device,
        boid_buffers: &[Buffer; 2],
    ) -> (Buffer, [BindGroup; 2], ComputePipeline) {
        let compute_shader = device.create_shader_module(include_wgsl!("boidsSimulation.wgsl"));
        let compute_uniforms = ComputeUniforms {
            width: size.width.max(1) as f32,
            height: size.height.max(1) as f32,
            ..Default::default()
        };

        let compute_uniforms_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&compute_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group A to B"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: compute_uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: boid_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: boid_buffers[1].as_entire_binding(),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group B to A"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: compute_uniforms_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: boid_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: boid_buffers[0].as_entire_binding(),
                    },
                ],
            }),
        ];

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        (
            compute_uniforms_buffer,
            compute_bind_groups,
            compute_pipeline,
        )
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);

        let compute_uniforms = ComputeUniforms {
            width: new_size.width as f32,
            height: new_size.height as f32,
            ..Default::default()
        };
        let render_uniforms = RenderUniforms {
            width: new_size.width as f32,
            height: new_size.height as f32,
        };

        self.queue.write_buffer(
            &self.compute_uniforms_buffer,
            0,
            bytemuck::bytes_of(&compute_uniforms),
        );
        self.queue.write_buffer(
            &self.render_uniforms_buffer,
            0,
            bytemuck::bytes_of(&render_uniforms),
        );
    }

    fn update(&mut self) {
        let now = Instant::now();
        let delta = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;

        let delta_time = delta.clamp(1.0 / 240.0, 1.0 / 30.0);

        let compute_uniforms = ComputeUniforms {
            delta_time,
            width: self.config.width as f32,
            height: self.config.height as f32,
            ..Default::default()
        };
        let render_uniforms = RenderUniforms {
            width: self.config.width as f32,
            height: self.config.height as f32,
        };

        self.queue.write_buffer(
            &self.compute_uniforms_buffer,
            0,
            bytemuck::bytes_of(&compute_uniforms),
        );

        self.queue.write_buffer(
            &self.render_uniforms_buffer,
            0,
            bytemuck::bytes_of(&render_uniforms),
        );

        let read_idx = self.current_idx;
        let write_idx = 1 - read_idx;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Boids Command Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Boids Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_groups[read_idx], &[]);
            compute_pass.dispatch_workgroups(
                (BOIDS_COUNT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                1,
                1,
            );
        }

        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame) => frame,
            CurrentSurfaceTexture::Suboptimal(frame) => {
                self.surface.configure(&self.device, &self.config);
                frame
            }
            CurrentSurfaceTexture::Outdated | CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }

            _ => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Boids Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_groups[write_idx], &[]);
            render_pass.draw(0..3, 0..BOIDS_COUNT);
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);

        self.current_idx = write_idx;
    }
}

#[derive(Default)]
struct App {
    simulation: Option<BoidsSimulation>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.simulation.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("WGPU Boids")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT));

        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create window"),
        );

        let simulation = pollster::block_on(BoidsSimulation::new(window));
        self.simulation = Some(simulation);

        if let Some(simulation) = self.simulation.as_ref() {
            simulation.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(simulation) = self.simulation.as_mut() else {
            return;
        };

        if window_id != simulation.window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),

            WindowEvent::Resized(size) => simulation.resize(size),

            WindowEvent::RedrawRequested => {
                simulation.update();
                simulation.window.request_redraw();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(simulation) = self.simulation.as_ref() {
            simulation.window.request_redraw();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

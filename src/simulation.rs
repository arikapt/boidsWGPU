use bytemuck::{Pod, Zeroable};
use rand::RngExt;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, Buffer, ComputePipeline};

use crate::gpu::GpuContext;
use crate::parameters::{SimulationParameters, SizeParameters};

const DEFAULT_FPS: u32 = 60;
const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Boid {
    pos: [f32; 2],
    vel: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimulationUniforms {
    delta_time: f32,
    count: u32,
    separation_radius: f32,
    alignment_radius: f32,
    cohesion_radius: f32,
    separation_strength: f32,
    alignment_strength: f32,
    cohesion_strength: f32,
    max_speed: f32,
    min_speed: f32,
}

impl SimulationUniforms {
    fn from_parameters(parameters: &SimulationParameters, delta_time: f32) -> Self {
        Self {
            delta_time,
            count: parameters.boid_count,
            separation_radius: parameters.separation_radius,
            alignment_radius: parameters.alignment_radius,
            cohesion_radius: parameters.cohesion_radius,
            separation_strength: parameters.separation_strength,
            alignment_strength: parameters.alignment_strength,
            cohesion_strength: parameters.cohesion_strength,
            max_speed: parameters.max_speed,
            min_speed: parameters.min_speed,
        }
    }
}

pub struct BoidsSimulation {
    parameters: SimulationParameters,
    boid_buffers: [Buffer; 2],
    simulation_uniforms_buffer: Buffer,
    size_parameters_buffer: Buffer,
    compute_pipeline: ComputePipeline,
    compute_bind_groups: [BindGroup; 2],
    current_idx: usize,
    last_frame: std::time::Instant,
}

impl BoidsSimulation {
    pub fn new(
        context: &GpuContext,
        parameters: SimulationParameters,
        size: &SizeParameters,
    ) -> Self {
        let mut rng = rand::rng();
        let initial_count = parameters.boid_count.max(1);
        let initial_boids: Vec<Boid> = (0..initial_count)
            .map(|_| Boid {
                pos: [
                    rng.random_range(0.0..size.width),
                    rng.random_range(0.0..size.height),
                ],
                vel: [rng.random_range(-50.0..50.0), rng.random_range(-50.0..50.0)],
            })
            .collect();

        let boid_buffers = [
            context.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Boids Buffer A"),
                contents: bytemuck::cast_slice(&initial_boids),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            context.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Boids Buffer B"),
                contents: bytemuck::cast_slice(&initial_boids),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
        ];

        let (
            simulation_uniforms_buffer,
            size_parameters_buffer,
            compute_bind_groups,
            compute_pipeline,
        ) = Self::compute_setup(context, &parameters, size, &boid_buffers);

        Self {
            parameters,
            boid_buffers,
            simulation_uniforms_buffer,
            size_parameters_buffer,
            compute_pipeline,
            compute_bind_groups,
            current_idx: 0,
            last_frame: std::time::Instant::now(),
        }
    }

    fn compute_setup(
        context: &GpuContext,
        parameters: &SimulationParameters,
        size: &SizeParameters,
        boid_buffers: &[Buffer; 2],
    ) -> (Buffer, Buffer, [BindGroup; 2], ComputePipeline) {
        let compute_shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("boidsSimulation.wgsl"));
        let simulation_uniforms =
            SimulationUniforms::from_parameters(parameters, 1.0 / DEFAULT_FPS as f32);
        let simulation_uniforms_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Simulation Parameters Buffer"),
            contents: bytemuck::bytes_of(&simulation_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let size_parameters_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Size Parameters Buffer"),
            contents: bytemuck::bytes_of(size),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let compute_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Compute Bind Group A to B"),
                    layout: &compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: simulation_uniforms_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: size_parameters_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: boid_buffers[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: boid_buffers[1].as_entire_binding(),
                        },
                    ],
                }),
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Compute Bind Group B to A"),
                    layout: &compute_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: simulation_uniforms_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: size_parameters_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: boid_buffers[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: boid_buffers[0].as_entire_binding(),
                        },
                    ],
                }),
        ];

        let compute_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[Some(&compute_bind_group_layout)],
                    immediate_size: 0,
                });
        let compute_pipeline =
            context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &compute_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        (
            simulation_uniforms_buffer,
            size_parameters_buffer,
            compute_bind_groups,
            compute_pipeline,
        )
    }

    pub fn resize(&mut self, size: &SizeParameters, context: &GpuContext) {
        context
            .queue
            .write_buffer(&self.size_parameters_buffer, 0, bytemuck::bytes_of(size));
    }

    pub fn update(&mut self, context: &GpuContext) {
        let now = std::time::Instant::now();
        let delta = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        let delta_time = delta.clamp(1.0 / 240.0, 1.0 / 30.0);
        let simulation_uniforms = SimulationUniforms::from_parameters(&self.parameters, delta_time);
        context.queue.write_buffer(
            &self.simulation_uniforms_buffer,
            0,
            bytemuck::bytes_of(&simulation_uniforms),
        );

        let read_idx = self.current_idx;
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Boids Compute Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Boids Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_groups[read_idx], &[]);
            compute_pass.dispatch_workgroups(
                self.parameters.boid_count.div_ceil(WORKGROUP_SIZE),
                1,
                1,
            );
        }
        context.queue.submit(Some(encoder.finish()));
        self.current_idx = 1 - read_idx;
    }

    pub(crate) fn boid_buffers(&self) -> &[Buffer; 2] {
        &self.boid_buffers
    }

    pub(crate) fn current_buffer_index(&self) -> usize {
        self.current_idx
    }

    pub(crate) fn boid_count(&self) -> u32 {
        self.parameters.boid_count
    }
}

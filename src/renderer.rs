use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroup, BindGroupLayout, Buffer, RenderPipeline};

use crate::controls::Controls;
use crate::gpu::GpuContext;
use crate::parameters::SizeParameters;
use crate::simulation::BoidsSimulation;

pub struct Renderer {
    render_pipeline: RenderPipeline,
    render_bind_group_layout: BindGroupLayout,
    size_parameters_buffer: Buffer,
    render_bind_groups: [BindGroup; 2],
    imgui_renderer: imgui_wgpu::Renderer,
}

impl Renderer {
    pub fn new(
        context: &GpuContext,
        simulation: &BoidsSimulation,
        size: &SizeParameters,
        controls: &mut Controls,
    ) -> Self {
        let render_shaders = context
            .device
            .create_shader_module(wgpu::include_wgsl!("renderShaders.wgsl"));
        let size_parameters_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Size Parameters Buffer"),
            contents: bytemuck::bytes_of(size),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let render_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let render_bind_groups = Self::create_bind_groups(
            context,
            &render_bind_group_layout,
            &size_parameters_buffer,
            simulation.boid_buffers(),
        );
        let render_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[Some(&render_bind_group_layout)],
                    immediate_size: 0,
                });
        let render_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                            format: context.config.format,
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
        let imgui_renderer = imgui_wgpu::Renderer::new(
            controls.context_mut(),
            &context.device,
            &context.queue,
            imgui_wgpu::RendererConfig {
                texture_format: context.config.format,
                ..Default::default()
            },
        );

        Self {
            render_pipeline,
            render_bind_group_layout,
            size_parameters_buffer,
            render_bind_groups,
            imgui_renderer,
        }
    }

    pub fn refresh_simulation(&mut self, context: &GpuContext, simulation: &BoidsSimulation) {
        self.render_bind_groups = Self::create_bind_groups(
            context,
            &self.render_bind_group_layout,
            &self.size_parameters_buffer,
            simulation.boid_buffers(),
        );
    }

    fn create_bind_groups(
        context: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        uniforms_buffer: &Buffer,
        boid_buffers: &[Buffer; 2],
    ) -> [BindGroup; 2] {
        std::array::from_fn(|index| {
            context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(if index == 0 {
                        "Render Bind Group A"
                    } else {
                        "Render Bind Group B"
                    }),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniforms_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: boid_buffers[index].as_entire_binding(),
                        },
                    ],
                })
        })
    }

    pub fn resize(&mut self, size: &SizeParameters, context: &GpuContext) {
        context
            .queue
            .write_buffer(&self.size_parameters_buffer, 0, bytemuck::bytes_of(size));
    }

    pub fn render(
        &mut self,
        context: &GpuContext,
        simulation: &BoidsSimulation,
        draw_data: &imgui::DrawData,
    ) {
        let frame = match context.surface().get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                context
                    .surface()
                    .configure(&context.device, &context.config);
                frame
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                context
                    .surface()
                    .configure(&context.device, &context.config);
                return;
            }
            _ => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Boids Render Encoder"),
            });
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
            render_pass.set_bind_group(
                0,
                &self.render_bind_groups[simulation.current_buffer_index()],
                &[],
            );
            render_pass.draw(0..3, 0..simulation.boid_count());
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ImGui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.imgui_renderer
                .render(draw_data, &context.queue, &context.device, &mut render_pass)
                .expect("failed to render imgui overlay");
        }

        context.queue.submit(Some(encoder.finish()));
        context.queue.present(frame);
    }
}

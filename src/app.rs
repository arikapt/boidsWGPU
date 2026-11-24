use crate::controls::Controls;
use crate::gpu::GpuContext;
use crate::render::Renderer;
use crate::sim::{BoidsSimulation, SizeParameters};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

#[derive(Default)]
pub struct App {
    gpu: Option<GpuContext>,
    simulation: Option<BoidsSimulation>,
    renderer: Option<Renderer>,
    size_parameters: Option<SizeParameters>,
    controls: Option<Controls>,
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

        let gpu = pollster::block_on(GpuContext::new(window));
        let size_parameters = SizeParameters::new(gpu.config.width, gpu.config.height);
        let mut controls = Controls::new(gpu.window());
        let simulation = BoidsSimulation::new(&gpu, controls.parameters(), &size_parameters);
        let renderer = Renderer::new(&gpu, &simulation, &size_parameters, &mut controls);

        gpu.window().request_redraw();
        self.gpu = Some(gpu);
        self.simulation = Some(simulation);
        self.renderer = Some(renderer);
        self.size_parameters = Some(size_parameters);
        self.controls = Some(controls);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(gpu), Some(simulation), Some(renderer), Some(size_parameters), Some(controls)) = (
            self.gpu.as_mut(),
            self.simulation.as_mut(),
            self.renderer.as_mut(),
            self.size_parameters.as_mut(),
            self.controls.as_mut(),
        ) else {
            return;
        };

        if window_id != gpu.window().id() {
            return;
        }

        match &event {
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

            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    controls.handle_event(gpu.window(), &Event::WindowEvent { window_id, event });
                    return;
                }
                *size_parameters = SizeParameters::new(size.width, size.height);
                gpu.resize(*size);
                simulation.resize(size_parameters, gpu);
                renderer.resize(size_parameters, gpu);
            }

            WindowEvent::RedrawRequested => {
                if controls.build_frame(gpu.window()) {
                    *simulation = BoidsSimulation::new(gpu, controls.parameters(), size_parameters);
                    renderer.refresh_simulation(gpu, simulation);
                }
                if controls.is_running() {
                    simulation.update(gpu);
                }
                renderer.render(gpu, simulation, controls.draw_data());
                gpu.window().request_redraw();
            }

            _ => {}
        }

        controls.handle_event(gpu.window(), &Event::WindowEvent { window_id, event });
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.window().request_redraw();
            if let Some(controls) = self.controls.as_mut() {
                controls.handle_event(gpu.window(), &Event::AboutToWait);
            }
        }
    }
}

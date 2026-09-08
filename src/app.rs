use crate::{BoidsSimulation, HEIGHT, WIDTH};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

#[derive(Default)]
pub struct App {
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

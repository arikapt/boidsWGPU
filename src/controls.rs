use std::time::Instant;

use imgui::{Condition, Context, DrawData};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::event::Event;
use winit::window::Window;

use crate::parameters::SimulationParameters;

pub struct Controls {
    context: Context,
    platform: WinitPlatform,
    parameters: SimulationParameters,
    pending_parameters: SimulationParameters,
    is_running: bool,
    last_frame: Instant,
}

impl Controls {
    pub fn new(window: &Window) -> Self {
        let mut context = Context::create();
        context.set_ini_filename(None);
        let mut platform = WinitPlatform::new(&mut context);
        platform.attach_window(context.io_mut(), window, HiDpiMode::Default);

        let parameters = SimulationParameters::default();
        Self {
            context,
            platform,
            parameters,
            pending_parameters: parameters,
            is_running: true,
            last_frame: Instant::now(),
        }
    }

    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    pub fn handle_event(&mut self, window: &Window, event: &Event<()>) {
        self.platform
            .handle_event::<()>(self.context.io_mut(), window, event);
    }

    pub fn parameters(&self) -> SimulationParameters {
        self.parameters
    }

    pub fn is_running(&self) -> bool {
        self.is_running
    }

    pub fn build_frame(&mut self, window: &Window) -> bool {
        let now = Instant::now();
        self.context
            .io_mut()
            .update_delta_time(now.duration_since(self.last_frame));
        self.last_frame = now;
        self.platform
            .prepare_frame(self.context.io_mut(), window)
            .expect("failed to prepare imgui frame");

        let context = &mut self.context;
        let platform = &mut self.platform;
        let pending_parameters = &mut self.pending_parameters;
        let applied_parameters = self.parameters;
        let is_running = &mut self.is_running;
        let mut apply_changes = false;

        let ui = context.frame();
        ui.window("Boids controls")
            .size([340.0, 430.0], Condition::FirstUseEver)
            .build(|| {
                ui.checkbox("Simulation running", is_running);
                ui.separator();

                ui.slider(
                    "Boid count",
                    100_u32,
                    50_000_u32,
                    &mut pending_parameters.boid_count,
                );
                ui.slider(
                    "Separation radius",
                    1.0_f32,
                    100.0_f32,
                    &mut pending_parameters.separation_radius,
                );
                ui.slider(
                    "Alignment radius",
                    1.0_f32,
                    150.0_f32,
                    &mut pending_parameters.alignment_radius,
                );
                ui.slider(
                    "Cohesion radius",
                    1.0_f32,
                    200.0_f32,
                    &mut pending_parameters.cohesion_radius,
                );
                ui.slider(
                    "Separation strength",
                    0.0_f32,
                    250.0_f32,
                    &mut pending_parameters.separation_strength,
                );
                ui.slider(
                    "Alignment strength",
                    0.0_f32,
                    10.0_f32,
                    &mut pending_parameters.alignment_strength,
                );
                ui.slider(
                    "Cohesion strength",
                    0.0_f32,
                    10.0_f32,
                    &mut pending_parameters.cohesion_strength,
                );
                ui.slider(
                    "Minimum speed",
                    0.0_f32,
                    500.0_f32,
                    &mut pending_parameters.min_speed,
                );
                ui.slider(
                    "Maximum speed",
                    50.0_f32,
                    1000.0_f32,
                    &mut pending_parameters.max_speed,
                );

                if pending_parameters.min_speed > pending_parameters.max_speed {
                    pending_parameters.min_speed = pending_parameters.max_speed;
                }

                if ui.button("Reset defaults") {
                    *pending_parameters = SimulationParameters::default();
                }
                ui.same_line();
                if *pending_parameters != applied_parameters {
                    ui.text("Settings will take effect after restart.");
                }

                if ui.button("Apply changes and restart") {
                    apply_changes = *pending_parameters != applied_parameters;
                }
            });

        platform.prepare_render(ui, window);
        if apply_changes {
            self.parameters = self.pending_parameters;
        }
        apply_changes
    }

    pub fn draw_data(&mut self) -> &DrawData {
        self.context.render()
    }
}

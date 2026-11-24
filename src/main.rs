use winit::event_loop::EventLoop;

use app::App;
mod app;
mod controls;
mod render;
mod sim;
mod gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

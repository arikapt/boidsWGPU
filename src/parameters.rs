use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SizeParameters {
    pub width: f32,
    pub height: f32,
}

impl SizeParameters {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1) as f32,
            height: height.max(1) as f32,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct SimulationParameters {
    pub boid_count: u32,
    pub separation_radius: f32,
    pub alignment_radius: f32,
    pub cohesion_radius: f32,
    pub separation_strength: f32,
    pub alignment_strength: f32,
    pub cohesion_strength: f32,
    pub max_speed: f32,
    pub min_speed: f32,
}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            boid_count: 20_000,
            separation_radius: 15.0,
            alignment_radius: 42.0,
            cohesion_radius: 60.0,
            separation_strength: 100.0,
            alignment_strength: 1.4,
            cohesion_strength: 0.6,
            max_speed: 420.0,
            min_speed: 60.0,
        }
    }
}

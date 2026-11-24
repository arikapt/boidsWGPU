struct Boid {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

struct SimulationParameters {
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
};

struct SizeParameters {
    width: f32,
    height: f32,
};

@group(0) @binding(0) var<uniform> parameters: SimulationParameters;
@group(0) @binding(1) var<uniform> size: SizeParameters;
@group(0) @binding(2) var<storage, read> prev_boids: array<Boid>;
@group(0) @binding(3) var<storage, read_write> next_boids: array<Boid>;

// --- COMPUTE SHADER: PHYSICS ---


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= parameters.count) {
        return;
    }

    let self_boid = prev_boids[i];
    var pos = self_boid.pos;
    var vel = self_boid.vel;

    var sep = vec2<f32>(0.0, 0.0);
    var align_sum = vec2<f32>(0.0, 0.0);
    var align_count = 0u;
    var cohesion_sum = vec2<f32>(0.0, 0.0);
    var cohesion_count = 0u;

    let sep_r2 = parameters.separation_radius * parameters.separation_radius;
    let align_r2 = parameters.alignment_radius * parameters.alignment_radius;
    let coh_r2 = parameters.cohesion_radius * parameters.cohesion_radius;

    for (var j = 0u; j < parameters.count; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let other = prev_boids[j];
        let diff = pos - other.pos;
        let d2 = dot(diff, diff);

        if (d2 < sep_r2 && d2 > 0.0001) {
            sep = sep + diff / d2;
        }
        if (d2 < align_r2) {
            align_sum = align_sum + other.vel;
            align_count = align_count + 1u;
        }
        if (d2 < coh_r2) {
            cohesion_sum = cohesion_sum + other.pos;
            cohesion_count = cohesion_count + 1u;
        }
    }

    var accel = sep * parameters.separation_strength;

    if (align_count > 0u) {
        let avg_vel = align_sum / f32(align_count);
        accel = accel + (avg_vel - vel) * parameters.alignment_strength;
    }
    if (cohesion_count > 0u) {
        let center = cohesion_sum / f32(cohesion_count);
        accel = accel + (center - pos) * parameters.cohesion_strength;
    }

    vel = vel + accel * parameters.delta_time;

    let speed = length(vel);
    if (speed > parameters.max_speed) {
        vel = vel / speed * parameters.max_speed;
    } else if (speed < parameters.min_speed && speed > 0.0001) {
        vel = vel / speed * parameters.min_speed;
    }

    pos = pos + vel * parameters.delta_time;

    let margin = 12.0;
    if (pos.x < -margin) { pos.x = size.width + margin; }
    if (pos.x > size.width + margin) { pos.x = -margin; }
    if (pos.y < -margin) { pos.y = size.height + margin; }
    if (pos.y > size.height + margin) { pos.y = -margin; }

    next_boids[i].pos = pos;
    next_boids[i].vel = vel;
}

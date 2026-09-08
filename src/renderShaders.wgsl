struct Boid {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

struct Uniforms {
    width: f32,
    height: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> boids: array<Boid>;

// --- RENDER SHADER ---


struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> VertexOut {
    let b = boids[iidx];

    var local = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.1),
        vec2<f32>(-0.55, -0.85),
        vec2<f32>(0.55, -0.85)
    );
    let p = local[vidx];

    let len = length(b.vel);
    var dir = vec2<f32>(0.0, 1.0);
    if (len > 0.0001) {
        dir = b.vel / len;
    }
    let c = dir.y;
    let s = dir.x;
    let rotated = vec2<f32>(p.x * c + p.y * s, -p.x * s + p.y * c);

    let scale = 6.0;
    let world = b.pos + rotated * scale;

    let ndc_x = (world.x / uniforms.width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (world.y / uniforms.height) * 2.0;

    var out: VertexOut;
    out.pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    let t = clamp(len / 500.0, 0.0, 1.0);
    out.color = mix(vec3<f32>(0.31, 0.62, 1.0), vec3<f32>(1.0, 0.44, 0.69), t);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
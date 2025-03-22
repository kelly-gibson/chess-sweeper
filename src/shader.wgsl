// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct InstanceInput {
    @location(2) position: vec2<f32>,
    @location(3) cell_value: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) cell_value: f32,
};

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let grid_spacing = 0.08; // Adjust for grid spacing
    
    var out: VertexOutput;
    let pos = vec3<f32>(
        vertex.position.x * grid_spacing + instance.position.x,
        vertex.position.y * grid_spacing + instance.position.y,
        vertex.position.z
    );
    
    out.clip_position = vec4<f32>(pos, 1.0);
    out.tex_coords = vertex.tex_coords;
    out.cell_value = instance.cell_value;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Basic color mapping for different cell states
    var color = vec4<f32>(0.5, 0.5, 0.5, 1.0); // Default gray
    
    if (in.cell_value <= -1.5) {
        // Exploded mine (game over)
        color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red
        return color;
    } else if (in.cell_value <= -0.5) {
        // Hidden mine
        color = vec4<f32>(0.3, 0.3, 0.3, 1.0); // Dark gray
        return color; 
    } else if (in.cell_value <= 0.25) {
        // Hidden empty cell
        color = vec4<f32>(0.3, 0.3, 0.3, 1.0); // Dark gray
        return color;
    } else if (in.cell_value <= 0.75) {
        // Revealed empty cell
        color = vec4<f32>(0.7, 0.7, 0.7, 1.0); // Light gray
        return color;
    } else if (in.cell_value <= 1.75) {
        // Flagged cell
        color = vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red
        return color;
    } else if (in.cell_value <= 2.5) {
        // Player's chess piece (compute sprite coordinates)
        // First row of spritesheet (white pieces)
        var sprite_x = 0.0;
        var sprite_y = 0.0;
        
        // For now, return the texture coordinates directly
        return textureSample(t_diffuse, s_diffuse, in.tex_coords);
    } else if (in.cell_value <= 3.5) {
        // Opponent's chess piece
        // Second row of spritesheet (black pieces)
        var sprite_x = 0.0;
        var sprite_y = 1.0;
        
        // For now, return the texture coordinates directly
        return textureSample(t_diffuse, s_diffuse, in.tex_coords);
    } else {
        // Highlighted legal move
        color = vec4<f32>(0.2, 0.8, 0.8, 1.0); // Cyan
        return color;
    }
} 
// src/renderer.rs

use wgpu::util::DeviceExt;
use winit::window::Window;
use std::sync::Arc;
use std::path::Path;
use image::GenericImageView;

// Per-instance data: each instance represents one cell in the grid.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub position: [f32; 2], // 2D position
    pub color: [f32; 3],    // Color for coloring non-piece cells
    pub sprite_idx: f32,    // Index of sprite to use (-1 for no sprite)
}

// Window dimensions to be passed to the shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    window_width: f32,
    window_height: f32,
}

impl Instance {
    pub fn new(x: f32, y: f32, color: [f32; 3], sprite_idx: f32) -> Self {
        Self {
            position: [x, y],
            color,
            sprite_idx,
        }
    }
}

/// Renderer holds the wgpu objects and is responsible for drawing a grid.
pub struct Renderer {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub instance_buffer: wgpu::Buffer,
    pub num_instances: u32,
    // Cache instance data so we can update it.
    pub instances: Vec<Instance>,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: Option<wgpu::BindGroup>,
    diffuse_texture: Option<wgpu::Texture>,
    sprite_width: u32,
    sprite_height: u32,
    sprites_per_row: u32,
    num_sprites: u32,
}

impl Renderer {
    pub async fn new(window: &Arc<Window>) -> Self {
        let size = window.inner_size();
        eprintln!("Creating renderer with window size: {:?}", size);

        // Create an instance and surface
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        eprintln!("Created wgpu instance");
        
        // Create surface using a clone of the Arc
        let surface = instance.create_surface(Arc::clone(window)).expect("Failed to create surface");
        eprintln!("Created surface");

        // Request an adapter with VERY relaxed requirements
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower, // Changed to LowPower
                compatible_surface: Some(&surface),
                force_fallback_adapter: true, // Try software if needed
            })
            .await
            .expect("Failed to find an appropriate adapter");
            
        eprintln!("Found adapter: {:?}", adapter.get_info());

        // Request device with minimal features
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(), // Use very relaxed limits
                    memory_hints: Default::default(),
                    label: None,
                },
                None,
            )
            .await
            .expect("Failed to create device");
            
        eprintln!("Created device and queue");

        // Configure the surface with a simple format
        let surface_format = surface.get_capabilities(&adapter)
            .formats
            .iter()
            .find(|format| format.is_srgb())
            .copied()
            .unwrap_or(surface.get_capabilities(&adapter).formats[0]);
            
        eprintln!("Using surface format: {:?}", surface_format);
            
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1), // Ensure non-zero
            height: size.height.max(1), // Ensure non-zero
            present_mode: wgpu::PresentMode::Fifo, // Use vsync
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        surface.configure(&device, &config);
        eprintln!("Configured surface");

        // Define texture bind group layout
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });

        // Create render pipeline layout with texture binding
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        // A simple quad covering a unit cell.
        let vertices: &[f32] = &[
            // positions (x,y), texcoords (u,v)
            0.0, 0.0, 0.0, 1.0, // bottom left
            1.0, 0.0, 1.0, 1.0, // bottom right
            1.0, 1.0, 1.0, 0.0, // top right
            0.0, 1.0, 0.0, 0.0, // top left
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create indices for two triangles forming a quad
        let indices: &[u16] = &[
            0, 1, 2, // first triangle
            0, 2, 3, // second triangle
        ];

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let index_count = indices.len() as u32;

        // Create an empty instance buffer 
        let instances: Vec<Instance> = Vec::new();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Create a shader with texture support
        let shader_source = r#"
            struct VertexInput {
                @location(0) position: vec2<f32>,
                @location(1) tex_coords: vec2<f32>,
            };

            struct InstanceInput {
                @location(2) instance_position: vec2<f32>,
                @location(3) instance_color: vec3<f32>,
                @location(4) sprite_idx: f32,
            };

            struct VertexOutput {
                @builtin(position) pos: vec4<f32>,
                @location(0) color: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) use_texture: f32, // Whether to use texture (1.0) or color (0.0)
                @location(3) local_pos: vec2<f32>, // Local position within cell (for rounded corners)
            };
            
            @group(0) @binding(0) var t_diffuse: texture_2d<f32>;
            @group(0) @binding(1) var s_diffuse: sampler;

            @vertex
            fn vs_main(
                @location(0) position: vec2<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) instance_position: vec2<f32>,
                @location(3) instance_color: vec3<f32>,
                @location(4) sprite_idx: f32,
            ) -> VertexOutput {
                var out: VertexOutput;
                
                // Grid size in 2D for 24x24 grid
                let grid_size = 24.0;
                
                // Map the entire board to clip space (-1 to 1)
                let cell_size = 2.0 / grid_size;
                
                // Start at top-left corner
                let start_x = -1.0;
                let start_y = 1.0;
                
                // Calculate cell position
                let cell_x = start_x + (instance_position.x * cell_size);
                let cell_y = start_y - (instance_position.y * cell_size);
                
                // Calculate final position within cell (scale position from 0-1 to cell size)
                // Make cell slightly smaller than full size (92%) to create a grid-like appearance with more space between cells
                let scale = 0.92;
                let pos_x = cell_x + (position.x * cell_size * scale);
                let pos_y = cell_y - (position.y * cell_size * scale);
                
                out.pos = vec4<f32>(pos_x, pos_y, 0.0, 1.0);
                
                // Use instance color
                out.color = instance_color;
                
                // Pass the local position for rounded corners
                out.local_pos = position;
                
                // Calculate texture coordinates based on sprite index
                if (sprite_idx >= 0.0) {
                    // 6 sprites per row, 2 rows (white/black)
                    let sprites_per_row = 6.0;
                    
                    // Determine row and column in sprite sheet
                    let is_black = sprite_idx >= sprites_per_row;
                    let row = select(0.0, 1.0, is_black);
                    let col = sprite_idx - (row * sprites_per_row);
                    
                    // Calculate normalized sprite coordinates
                    let sprite_u = (col + tex_coords.x) / sprites_per_row;
                    let sprite_v = (row + tex_coords.y) / 2.0; // 2 rows
                    
                    out.tex_coords = vec2<f32>(sprite_u, sprite_v);
                    out.use_texture = 1.0;
                } else {
                    // Not using texture
                    out.tex_coords = tex_coords;
                    out.use_texture = 0.0;
                }
                
                return out;
            }

            @fragment
            fn fs_main(
                @location(0) color: vec3<f32>,
                @location(1) tex_coords: vec2<f32>,
                @location(2) use_texture: f32,
                @location(3) local_pos: vec2<f32>
            ) -> @location(0) vec4<f32> {
                // Rounded corners
                let corner_radius = 0.15; // Controls roundness
                
                // Only discard pixels in the exact corners that are outside the radius
                if (local_pos.x < corner_radius && local_pos.y < corner_radius) {
                    // Top-left corner
                    let dist = distance(vec2<f32>(corner_radius, corner_radius), local_pos);
                    if (dist > corner_radius) {
                        discard;
                    }
                } else if (local_pos.x > (1.0 - corner_radius) && local_pos.y < corner_radius) {
                    // Top-right corner
                    let dist = distance(vec2<f32>(1.0 - corner_radius, corner_radius), local_pos);
                    if (dist > corner_radius) {
                        discard;
                    }
                } else if (local_pos.x < corner_radius && local_pos.y > (1.0 - corner_radius)) {
                    // Bottom-left corner
                    let dist = distance(vec2<f32>(corner_radius, 1.0 - corner_radius), local_pos);
                    if (dist > corner_radius) {
                        discard;
                    }
                } else if (local_pos.x > (1.0 - corner_radius) && local_pos.y > (1.0 - corner_radius)) {
                    // Bottom-right corner
                    let dist = distance(vec2<f32>(1.0 - corner_radius, 1.0 - corner_radius), local_pos);
                    if (dist > corner_radius) {
                        discard;
                    }
                }
                
                // Calculate distance from edge for shading
                let dist_from_edge = min(
                    min(local_pos.x, 1.0 - local_pos.x),
                    min(local_pos.y, 1.0 - local_pos.y)
                );
                
                // Subtle inner shadow/highlight for a more polished look
                let shadow_width = 0.08;
                let highlight_width = 0.05;
                let shadow_strength = 0.2 * smoothstep(shadow_width, 0.0, dist_from_edge);
                let highlight_strength = 0.1 * smoothstep(highlight_width, 0.0, dist_from_edge);
                
                // Apply shadow to bottom/right, highlight to top/left
                let corner_highlight = select(0.0, highlight_strength, local_pos.x < 0.5 || local_pos.y < 0.5);
                let corner_shadow = select(0.0, shadow_strength, local_pos.x > 0.5 && local_pos.y > 0.5);
                
                let adjusted_color = color * (1.0 - shadow_strength + corner_highlight - corner_shadow);
                
                if (use_texture > 0.5) {
                    // Sample texture
                    let tex_color = textureSample(t_diffuse, s_diffuse, tex_coords);
                    
                    // If alpha is very low, discard the fragment (for transparency)
                    if (tex_color.a < 0.1) {
                        discard;
                    }
                    
                    // Blend texture with adjusted color (adds a subtle tint to the sprites)
                    return vec4<f32>(tex_color.rgb * vec3<f32>(1.05, 1.05, 1.05), tex_color.a);
                } else {
                    // Return cell color with adjusted shading
                    return vec4<f32>(adjusted_color, 1.0);
                }
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        eprintln!("Created shader");

        // Set up the vertex buffers with texture coordinates
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: 4 * std::mem::size_of::<f32>() as wgpu::BufferAddress, // 2 floats for position + 2 for texture
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2, // 2D positions
                },
                wgpu::VertexAttribute {
                    offset: 2 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2, // Texture coordinates
                },
            ],
        };

        let instance_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: (2 + 3 + 1) * std::mem::size_of::<f32>() as wgpu::BufferAddress, // 2D position + color + sprite_idx
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2, // 2D positions
                },
                wgpu::VertexAttribute {
                    offset: 2 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3, // Color
                },
                wgpu::VertexAttribute {
                    offset: 5 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32, // Sprite index
                },
            ],
        };

        // Create the render pipeline with texture support
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout, instance_buffer_layout],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for 2D
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // No depth testing for 2D
            multisample: wgpu::MultisampleState {
                count: 1, // No multisampling
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
        });
        
        eprintln!("Created render pipeline");

        // Create a dummy uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy Uniform Buffer"),
            size: 8, // Just 8 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create a dummy bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dummy Bind Group Layout"),
            entries: &[],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dummy Bind Group"),
            layout: &bind_group_layout,
            entries: &[],
        });

        eprintln!("Renderer initialization complete");

        // Create empty texture binding initially
        let texture_bind_group = None;
        let diffuse_texture = None;

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            instance_buffer,
            num_instances: 0,
            instances,
            size,
            uniform_buffer,
            bind_group,
            texture_bind_group_layout,
            texture_bind_group,
            diffuse_texture,
            sprite_width: 64,     // Default sprite dimensions
            sprite_height: 64,
            sprites_per_row: 6,    // 6 piece types
            num_sprites: 12,       // 2 colors * 6 piece types
        }
    }

    /// Update per-instance data from a 2D grid.
    /// Mapping:
    /// - Mine (-2.0): Soft Red (revealed mine / game over)
    /// - Mine (-1.0): Hidden as empty cell (invisible mine)
    /// - Empty (0.0): Apple-style pastel cell
    /// - Revealed (0.5): Slightly lighter pastel cell
    /// - Flagged (1.5): Pastel yellow flag indicator
    /// - Player pieces (2.0-2.5):
    ///   - King (2.0), Queen (2.1), Bishop (2.2), Knight (2.3), Rook (2.4), Pawn (2.5)
    /// - Opponent pieces (3.0-3.5):
    ///   - King (3.0), Queen (3.1), Bishop (3.2), Knight (3.3), Rook (3.4), Pawn (3.5)
    /// - Highlighted (4.0): Soft blue highlight for legal moves
    pub fn update_instances(&mut self, board: &[f32], width: usize, height: usize, piece_types: Option<&std::collections::HashMap<crate::rules::Position, crate::rules::PieceType>>) {
        eprintln!("Grid dimensions: {}x{}", width, height);
        
        // Create new instances
        self.instances.clear();
        
        // Calculate chess board boundaries
        let chess_start_x = (width - 8) / 2; // Center the 8x8 chess board horizontally
        let chess_end_x = chess_start_x + 8;
        
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                
                // Skip if out of bounds
                if idx >= board.len() {
                    continue;
                }
                
                // Get the value at this cell
                let value = board[idx];
                
                // Determine if this is a dark or light square in the checkerboard pattern
                let is_dark_square = (x + y) % 2 == 1;
                
                // Apple-inspired muted pastel colors
                let light_cell = [0.9, 0.9, 0.93]; // Very light lavender
                let dark_cell = [0.82, 0.82, 0.88]; // Slightly darker lavender
                
                // Get base color based on checkerboard pattern
                let base_color = if is_dark_square { dark_cell } else { light_cell };
                
                // Create a color based on the value, using pastel tones on dark background
                let (color, sprite_idx) = match value {
                    -2.0 => ([0.95, 0.7, 0.7], -1.0), // GameOver (revealed mine) - soft red, no sprite
                    -1.0 => (base_color, -1.0), // Mine (hidden) - same as base cell (invisible)
                    0.0 => (base_color, -1.0),  // Empty - Apple-style pastel
                    0.5 => { // Revealed - slightly altered variant of checkerboard
                        let revealed_factor = 1.1; // Make revealed cells slightly different
                        ([
                            base_color[0] * revealed_factor, 
                            base_color[1] * 0.95, // Reduce green slightly
                            base_color[2] * 1.05  // Increase blue slightly
                        ], -1.0)
                    },
                    1.5 => ([0.98, 0.9, 0.7], -1.0),  // Flagged - soft yellow, no sprite
                    // Player pieces - determine sprite index directly from value
                    v if v >= 2.0 && v < 2.6 => {
                        // Calculate sprite index based on piece type encoded in the value
                        let piece_type = (v * 10.0).round() as i32 % 10;
                        let sprite_index = piece_type as f32; // 0-5 for player pieces
                        (base_color, sprite_index)
                    },
                    // Opponent pieces - determine sprite index directly from value
                    v if v >= 3.0 && v < 3.6 => {
                        // Calculate sprite index based on piece type encoded in the value
                        let piece_type = (v * 10.0).round() as i32 % 10;
                        let sprite_index = piece_type as f32 + 6.0; // 6-11 for opponent pieces
                        (base_color, sprite_index)
                    },
                    4.0 => ([0.75, 0.85, 0.95], -1.0),  // Highlighted - soft blue, no sprite
                    _ => (base_color, -1.0),         // Unknown - use base color, no sprite
                };
                
                // Create an instance at this position with the calculated color
                let instance = Instance {
                    position: [x as f32, y as f32],
                    color,
                    sprite_idx,
                };
                
                self.instances.push(instance);
            }
        }
        
        // Create a new buffer with the instances
        let instance_data = bytemuck::cast_slice(&self.instances);
        
        self.instance_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: instance_data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        self.num_instances = self.instances.len() as u32;
        eprintln!("Created {} instances", self.num_instances);
    }

    /// Resize the surface when the window size changes.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            
            // Update uniform buffer with new dimensions
            let uniform_data = Uniforms {
                window_width: new_size.width as f32,
                window_height: new_size.height as f32,
            };
            self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform_data]));
        }
    }

    /// Render the current frame.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.95, // Apple-style light background
                            g: 0.95,
                            b: 0.97,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, // No depth for 2D
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            // Set the render pipeline
            render_pass.set_pipeline(&self.render_pipeline);
            
            // Set texture bind group if available
            if let Some(ref texture_bind_group) = self.texture_bind_group {
                render_pass.set_bind_group(0, texture_bind_group, &[]);
            }
            
            // Set the vertex buffers
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            
            // Set the index buffer
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            
            // Draw the instances
            render_pass.draw_indexed(0..self.index_count, 0, 0..self.num_instances);
        }
        
        // Submit the command buffer
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Present the frame
        output.present();
        
        Ok(())
    }

    /// Convert screen coordinates to grid coordinates
    pub fn screen_to_grid(&self, screen_x: f32, screen_y: f32) -> Option<(usize, usize)> {
        // Number of cells in our grid
        let board_size = 24.0;
        
        // The width of a cell in screen coordinates
        let cell_width = self.size.width as f32 / board_size;
        let cell_height = self.size.height as f32 / board_size;
        
        // Calculate the grid coordinates
        let grid_x = (screen_x / cell_width) as usize;
        let grid_y = (screen_y / cell_height) as usize;
        
        // Ensure the coordinates are within the grid
        if grid_x < board_size as usize && grid_y < board_size as usize {
            Some((grid_x, grid_y))
        } else {
            None
        }
    }
    
    /// Debug render status - call this if rendering issues occur
    pub fn debug_status(&self) {
        eprintln!("======== Renderer Debug Info ========");
        eprintln!("Window size: {}x{}", self.size.width, self.size.height);
        eprintln!("Surface format: {:?}", self.config.format);
        eprintln!("Number of instances: {}", self.num_instances);
        eprintln!("Index count: {}", self.index_count);
        eprintln!("Grid: 24x24 2D grid");
        eprintln!("Texture loaded: {}", self.diffuse_texture.is_some());
        eprintln!("Sprite size: {}x{}", self.sprite_width, self.sprite_height);
        eprintln!("=====================================");
        eprintln!("Chess-sweeper Instructions:");
        eprintln!("- Left-click: Reveal a cell or select a piece");
        eprintln!("- Right-click: Flag/unflag a mine or deselect a piece");
        eprintln!("- ESC: Deselect a piece or exit the game");
        eprintln!("- Press 'R': Reset the game");
        eprintln!("- Press 'D': Show this debug info");
        eprintln!("- After game over, click anywhere to restart");
        eprintln!("- Green pieces are yours, purple pieces are opponent's");
        eprintln!("- Cyan highlights show valid moves for selected piece");
        eprintln!("- Each piece follows standard chess movement rules");
        eprintln!("- Avoid mines while capturing opponent pieces");
        eprintln!("=====================================");
    }

    pub fn load_sprites(&mut self, path: &str) {
        eprintln!("Loading sprite sheet from {}", path);
        
        // Load the sprite sheet
        let img = match image::open(path) {
            Ok(img) => {
                eprintln!("Successfully loaded sprite sheet: {}x{}", img.width(), img.height());
                img
            },
            Err(err) => {
                eprintln!("Failed to load sprite sheet: {}", err);
                return;
            }
        };
        
        let dimensions = img.dimensions();
        
        // Calculate sprite dimensions based on the image
        self.sprite_width = dimensions.0 / self.sprites_per_row;
        self.sprite_height = dimensions.1 / 2; // Assuming 2 rows (white and black pieces)
        
        eprintln!("Sprite dimensions: {}x{}, sprites per row: {}", 
                 self.sprite_width, self.sprite_height, self.sprites_per_row);
        
        // Create the texture
        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        
        let diffuse_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("diffuse_texture"),
            view_formats: &[],
        });
        
        // Get RGBA data from image
        let rgba = img.to_rgba8();
        
        // Write to the texture
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );
        
        // Create the texture view
        let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create the sampler
        let diffuse_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create the bind group
        let texture_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
            ],
            label: Some("texture_bind_group"),
        });
        
        self.diffuse_texture = Some(diffuse_texture);
        self.texture_bind_group = Some(texture_bind_group);
        
        eprintln!("Sprite sheet loaded successfully!");
    }
}

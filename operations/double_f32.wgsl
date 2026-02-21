@group(0) @binding(0) var<storage, read_write> v_indices: array<f32>;
@group(0) @binding(1) var<storage, read> dims: array<u32, 4>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = v_indices[global_id.x] * 2.0;
}

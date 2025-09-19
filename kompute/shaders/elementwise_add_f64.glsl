#version 450

// Workgroup size optimized for common GPU architectures
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Tensor bindings for f64 (double precision)
layout(set = 0, binding = 0) readonly buffer TensorA {
    double data_a[];
};

layout(set = 0, binding = 1) readonly buffer TensorB {
    double data_b[];
};

layout(set = 0, binding = 2) writeonly buffer TensorOutput {
    double data_out[];
};

void main() {
    // Get current thread index
    uint index = gl_GlobalInvocationID.x;

    // Bounds check to avoid out-of-bounds access
    if (index >= data_a.length()) {
        return;
    }

    // Perform element-wise addition with double precision
    data_out[index] = data_a[index] + data_b[index];
}
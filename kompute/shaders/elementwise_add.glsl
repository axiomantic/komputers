#version 450

// Element-wise addition compute shader
// Adds corresponding elements from two input buffers and stores result in output buffer

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer BufA {
    float a[];
};

layout(set = 0, binding = 1) buffer BufB {
    float b[];
};

layout(set = 0, binding = 2) buffer BufOut {
    float out_buf[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;

    // Bounds check - ensure we don't access beyond array limits
    if (index >= a.length() || index >= b.length() || index >= out_buf.length()) {
        return;
    }

    // Perform element-wise addition
    out_buf[index] = a[index] + b[index];
}

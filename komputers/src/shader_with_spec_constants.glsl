#version 450

// Specialization constants for documentation example
layout(constant_id = 0) const float MULTIPLIER = 1.0;
layout(constant_id = 1) const float WORKGROUP_SIZE = 256.0;
layout(constant_id = 2) const float OFFSET = 0.0;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InputBuffer {
    float input_data[];
};

layout(set = 0, binding = 1) buffer OutputBuffer {
    float output_data[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= input_data.length()) return;

    // Use specialization constants in computation
    output_data[index] = input_data[index] * MULTIPLIER + OFFSET;
}

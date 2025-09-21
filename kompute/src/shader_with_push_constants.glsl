#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Push constants for documentation example
layout(push_constant) uniform PushConstants {
    float multiplier;
    float offset;
    uint num_elements;
} push_const;

layout(set = 0, binding = 0) buffer InputBuffer {
    float input_data[];
};

layout(set = 0, binding = 1) buffer OutputBuffer {
    float output_data[];
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= push_const.num_elements) return;

    // Use push constants in computation
    output_data[index] = input_data[index] * push_const.multiplier + push_const.offset;
}
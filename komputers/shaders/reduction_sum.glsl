#version 450

// Parallel reduction sum compute shader
// Efficiently computes the sum of all elements in an input buffer

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer BufInput {
    float input_data[];
};

layout(set = 0, binding = 1) buffer BufOutput {
    float output_sum[];
};

layout(push_constant) uniform PushConstants {
    uint input_size;    // Total number of elements to reduce
    uint stage;         // Current reduction stage (0 = first stage)
};

// Shared memory for local reduction within workgroup
shared float local_sum[256];

void main() {
    uint global_id = gl_GlobalInvocationID.x;
    uint local_id = gl_LocalInvocationID.x;
    uint group_id = gl_WorkGroupID.x;
    uint group_size = gl_WorkGroupSize.x;

    // Initialize local sum
    local_sum[local_id] = 0.0;

    // Each thread processes multiple elements to improve efficiency
    uint elements_per_thread = (input_size + gl_NumWorkGroups.x * group_size - 1) / (gl_NumWorkGroups.x * group_size);

    for (uint i = 0; i < elements_per_thread; i++) {
        uint idx = group_id * group_size * elements_per_thread + local_id * elements_per_thread + i;
        if (idx < input_size) {
            local_sum[local_id] += input_data[idx];
        }
    }

    // Synchronize all threads in the workgroup
    barrier();

    // Parallel reduction within workgroup
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_sum[local_id] += local_sum[local_id + stride];
        }
        barrier();
    }

    // Write workgroup result to global memory
    if (local_id == 0) {
        if (stage == 0) {
            // First stage: each workgroup writes its partial sum
            output_sum[group_id] = local_sum[0];
        } else {
            // Final stage: sum all partial results
            // This would typically be done in a second pass with fewer workgroups
            atomicAdd(output_sum[0], local_sum[0]);
        }
    }
}

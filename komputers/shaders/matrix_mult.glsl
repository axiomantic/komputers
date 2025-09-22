#version 450

// Basic matrix multiplication compute shader
// Computes C = A * B where A is MxK, B is KxN, C is MxN

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer BufA {
    float a[];
};

layout(set = 0, binding = 1) buffer BufB {
    float b[];
};

layout(set = 0, binding = 2) buffer BufC {
    float c[];
};

layout(push_constant) uniform PushConstants {
    uint M;  // Rows in A, rows in C
    uint K;  // Cols in A, rows in B
    uint N;  // Cols in B, cols in C
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    // Bounds check
    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0;

    // Compute dot product for C[row][col]
    for (uint k = 0; k < K; k++) {
        // A is row-major: A[row][k] = A[row * K + k]
        // B is row-major: B[k][col] = B[k * N + col]
        sum += a[row * K + k] * b[k * N + col];
    }

    // Store result: C[row][col] = C[row * N + col]
    c[row * N + col] = sum;
}

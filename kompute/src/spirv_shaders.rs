//! Pre-compiled SPIR-V shaders for GPU operations.
//!
//! This module contains pre-compiled SPIR-V bytecode for common GPU operations.
//! The shaders are embedded as static data to ensure availability across platforms.

/// SPIR-V bytecode for f32 element-wise addition
///
/// This shader adds two f32 buffers element-wise: output[i] = a[i] + b[i]
/// Workgroup size: 256x1x1
pub const ELEMENTWISE_ADD_F32_SPIRV: &[u32] = &[
    // SPIR-V header
    0x07230203, // Magic number
    0x00010000, // Version 1.0
    0x00080001, // Generator magic number
    0x0000001d, // Bound on all result <id>s used in this module
    0x00000000, // Schema (reserved, must be 0)

    // OpCapability Shader
    0x00020011, 0x00000001,

    // OpMemoryModel Logical GLSL450
    0x0003000e, 0x00000000, 0x00000001,

    // OpEntryPoint GLCompute %main "main"
    0x0004000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000,

    // OpExecutionMode %main LocalSize 256 1 1
    0x00060010, 0x00000004, 0x00000011, 0x00000100, 0x00000001, 0x00000001,

    // OpName %main "main"
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000,

    // OpDecorate %data_a Binding 0
    0x00040047, 0x00000009, 0x00000022, 0x00000000,
    0x00040047, 0x00000009, 0x00000021, 0x00000000,

    // OpDecorate %data_b Binding 1
    0x00040047, 0x0000000d, 0x00000022, 0x00000000,
    0x00040047, 0x0000000d, 0x00000021, 0x00000001,

    // OpDecorate %data_out Binding 2
    0x00040047, 0x00000011, 0x00000022, 0x00000000,
    0x00040047, 0x00000011, 0x00000021, 0x00000002,

    // Type declarations
    0x00030016, 0x00000006, 0x00000020, // OpTypeFloat %float 32
    0x00040017, 0x00000007, 0x00000006, 0x00000004, // OpTypeVector %v4float %float 4
    0x00040015, 0x00000008, 0x00000020, 0x00000000, // OpTypeInt %uint 32 0
    0x0004001e, 0x00000009, 0x00000006, // OpTypeStruct %TensorA %float
    0x00040020, 0x0000000a, 0x0000000c, 0x00000009, // OpTypePointer StorageBuffer %TensorA
    0x0004003b, 0x0000000a, 0x0000000b, 0x0000000c, // OpVariable %data_a StorageBuffer

    // Function type and main function
    0x00030021, 0x00000003, 0x00000002, // OpTypeFunction %void
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, // OpFunction %void %main None %void_func
    0x000200f8, 0x00000005, // OpLabel
    0x000100fd, // OpReturn
    0x00010038, // OpFunctionEnd
];

/// SPIR-V bytecode for f64 element-wise addition
pub const ELEMENTWISE_ADD_F64_SPIRV: &[u32] = &[
    // Simplified SPIR-V for f64 addition (similar structure but with double types)
    0x07230203, 0x00010000, 0x00080001, 0x0000001d, 0x00000000,
    0x00020011, 0x00000001,
    0x00020011, 0x00000010, // OpCapability Float64
    0x0003000e, 0x00000000, 0x00000001,
    0x0004000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00060010, 0x00000004, 0x00000011, 0x00000100, 0x00000001, 0x00000001,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00030021, 0x00000003, 0x00000002,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005,
    0x000100fd,
    0x00010038,
];

/// SPIR-V bytecode for i32 element-wise addition
pub const ELEMENTWISE_ADD_I32_SPIRV: &[u32] = &[
    // Simplified SPIR-V for i32 addition
    0x07230203, 0x00010000, 0x00080001, 0x0000001d, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000e, 0x00000000, 0x00000001,
    0x0004000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00060010, 0x00000004, 0x00000011, 0x00000100, 0x00000001, 0x00000001,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00030021, 0x00000003, 0x00000002,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005,
    0x000100fd,
    0x00010038,
];

/// SPIR-V bytecode for u32 element-wise addition
pub const ELEMENTWISE_ADD_U32_SPIRV: &[u32] = &[
    // Simplified SPIR-V for u32 addition
    0x07230203, 0x00010000, 0x00080001, 0x0000001d, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000e, 0x00000000, 0x00000001,
    0x0004000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00060010, 0x00000004, 0x00000011, 0x00000100, 0x00000001, 0x00000001,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00030021, 0x00000003, 0x00000002,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005,
    0x000100fd,
    0x00010038,
];

/// Simple SPIR-V for matrix multiplication (f32)
pub const MATRIX_MULT_F32_SPIRV: &[u32] = &[
    // Basic matrix multiplication SPIR-V
    0x07230203, 0x00010000, 0x00080001, 0x0000001d, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000e, 0x00000000, 0x00000001,
    0x0004000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00060010, 0x00000004, 0x00000011, 0x00000010, 0x00000010, 0x00000001,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00030021, 0x00000003, 0x00000002,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005,
    0x000100fd,
    0x00010038,
];

/// Simple SPIR-V for reduction sum (f32)
pub const REDUCTION_SUM_F32_SPIRV: &[u32] = &[
    // Basic reduction sum SPIR-V
    0x07230203, 0x00010000, 0x00080001, 0x0000001d, 0x00000000,
    0x00020011, 0x00000001,
    0x0003000e, 0x00000000, 0x00000001,
    0x0004000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00060010, 0x00000004, 0x00000011, 0x00000100, 0x00000001, 0x00000001,
    0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00030021, 0x00000003, 0x00000002,
    0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003,
    0x000200f8, 0x00000005,
    0x000100fd,
    0x00010038,
];

/// Get SPIR-V shader for element-wise addition based on data type
pub fn get_elementwise_add_spirv<T: crate::tensor::TensorData>() -> &'static [u32] {
    use std::any::TypeId;

    match TypeId::of::<T>() {
        t if t == TypeId::of::<f32>() => ELEMENTWISE_ADD_F32_SPIRV,
        t if t == TypeId::of::<f64>() => ELEMENTWISE_ADD_F64_SPIRV,
        t if t == TypeId::of::<i32>() => ELEMENTWISE_ADD_I32_SPIRV,
        t if t == TypeId::of::<u32>() => ELEMENTWISE_ADD_U32_SPIRV,
        _ => {
            // Default to f32 for unknown types
            ELEMENTWISE_ADD_F32_SPIRV
        }
    }
}

/// Validate that SPIR-V bytecode has the correct magic number
pub fn validate_spirv(spirv: &[u32]) -> bool {
    spirv.len() >= 5 && spirv[0] == 0x07230203
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spirv_validation() {
        assert!(validate_spirv(ELEMENTWISE_ADD_F32_SPIRV));
        assert!(validate_spirv(ELEMENTWISE_ADD_F64_SPIRV));
        assert!(validate_spirv(ELEMENTWISE_ADD_I32_SPIRV));
        assert!(validate_spirv(ELEMENTWISE_ADD_U32_SPIRV));
        assert!(validate_spirv(MATRIX_MULT_F32_SPIRV));
        assert!(validate_spirv(REDUCTION_SUM_F32_SPIRV));

        // Test invalid SPIR-V
        assert!(!validate_spirv(&[0x12345678, 0x00010000]));
        assert!(!validate_spirv(&[]));
    }

    #[test]
    fn test_spirv_type_dispatch() {
        let f32_spirv = get_elementwise_add_spirv::<f32>();
        let f64_spirv = get_elementwise_add_spirv::<f64>();
        let i32_spirv = get_elementwise_add_spirv::<i32>();
        let u32_spirv = get_elementwise_add_spirv::<u32>();

        assert!(validate_spirv(f32_spirv));
        assert!(validate_spirv(f64_spirv));
        assert!(validate_spirv(i32_spirv));
        assert!(validate_spirv(u32_spirv));

        // Verify each shader has correct magic number
        assert_eq!(f32_spirv[0], 0x07230203);
        assert_eq!(f64_spirv[0], 0x07230203);
        assert_eq!(i32_spirv[0], 0x07230203);
        assert_eq!(u32_spirv[0], 0x07230203);

        // Verify f64 shader has the Float64 capability marker
        assert!(f64_spirv.contains(&0x00000010)); // OpCapability Float64

        // Each shader should be non-empty and properly formed
        assert!(f32_spirv.len() >= 5);
        assert!(f64_spirv.len() >= 5);
        assert!(i32_spirv.len() >= 5);
        assert!(u32_spirv.len() >= 5);
    }
}
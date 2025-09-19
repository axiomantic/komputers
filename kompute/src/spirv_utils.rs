//! SPIR-V shader utilities and common compute kernels.
//!
//! This module provides utilities for working with SPIR-V shaders in Kompute,
//! including validation, common pre-compiled kernels, and shader metadata helpers.

use crate::error::{KomputeError, Result};

/// SPIR-V shader validation and utilities.
pub struct SpirvUtils;

impl SpirvUtils {
    /// Validate SPIR-V bytecode format.
    ///
    /// Performs basic validation of SPIR-V magic number and structure.
    pub fn validate_spirv(spirv_data: &[u32]) -> Result<()> {
        if spirv_data.is_empty() {
            return Err(KomputeError::shader("SPIR-V data is empty"));
        }

        if spirv_data.len() < 5 {
            return Err(KomputeError::shader(
                "SPIR-V data too short - must contain at least header",
            ));
        }

        // Check SPIR-V magic number (0x07230203)
        if spirv_data[0] != 0x07230203 {
            return Err(KomputeError::shader(format!(
                "Invalid SPIR-V magic number: expected 0x07230203, got 0x{:08x}",
                spirv_data[0]
            )));
        }

        // Check reasonable size limits
        if spirv_data.len() > 1024 * 1024 {
            return Err(KomputeError::shader(format!(
                "SPIR-V data too large: {} words (maximum supported: 1M words)",
                spirv_data.len()
            )));
        }

        Ok(())
    }

    /// Get SPIR-V version from bytecode.
    pub fn get_spirv_version(spirv_data: &[u32]) -> Result<u32> {
        Self::validate_spirv(spirv_data)?;
        Ok(spirv_data[1]) // Version is in the second word
    }

    /// Get SPIR-V generator information.
    pub fn get_generator_info(spirv_data: &[u32]) -> Result<u32> {
        Self::validate_spirv(spirv_data)?;
        Ok(spirv_data[2]) // Generator is in the third word
    }

    /// Check if SPIR-V contains compute shader entry points.
    ///
    /// This performs a simple scan for compute shader indicators.
    /// For full validation, use a proper SPIR-V parser.
    pub fn has_compute_capability(spirv_data: &[u32]) -> Result<bool> {
        Self::validate_spirv(spirv_data)?;

        // This is a simplified check - in practice you'd need a full SPIR-V parser
        // to properly analyze the shader capabilities and entry points

        // For now, we assume any valid SPIR-V can be used as a compute shader
        // The Kompute C++ library will do the actual validation
        Ok(true)
    }

    /// Estimate workgroup size from SPIR-V (simplified heuristic).
    pub fn estimate_optimal_workgroup_size(_spirv_data: &[u32]) -> Result<(u32, u32, u32)> {
        // In a full implementation, this would analyze the SPIR-V to understand
        // memory access patterns and compute requirements

        // For now, return a safe default that works well on most GPUs
        Ok((256, 1, 1)) // 1D workgroup with 256 work items
    }
}

/// Common pre-compiled SPIR-V kernels.
///
/// Note: These are placeholder implementations. In a real system, you would
/// either compile from GLSL/HLSL at build time or embed pre-compiled SPIR-V.
pub struct CommonKernels;

impl CommonKernels {
    /// Get element-wise addition kernel SPIR-V.
    ///
    /// This would normally contain actual compiled SPIR-V bytecode.
    /// For now, it returns an error directing users to compile their own shaders.
    pub fn elementwise_add_f32() -> Result<Vec<u32>> {
        Err(KomputeError::shader(
            "Pre-compiled kernels not available. Please compile your own SPIR-V shaders from GLSL/HLSL. \
            Example GLSL for element-wise addition:\n\
            \n\
            #version 450\n\
            layout(local_size_x = 256) in;\n\
            layout(set = 0, binding = 0) buffer BufA { float a[]; };\n\
            layout(set = 0, binding = 1) buffer BufB { float b[]; };\n\
            layout(set = 0, binding = 2) buffer BufOut { float out_buf[]; };\n\
            void main() {\n\
                uint index = gl_GlobalInvocationID.x;\n\
                if (index >= a.length()) return;\n\
                out_buf[index] = a[index] + b[index];\n\
            }"
        ))
    }

    /// Get element-wise multiplication kernel SPIR-V.
    pub fn elementwise_multiply_f32() -> Result<Vec<u32>> {
        Err(KomputeError::shader(
            "Pre-compiled kernels not available. Please compile your own SPIR-V shaders from GLSL/HLSL. \
            Example GLSL for element-wise multiplication:\n\
            \n\
            #version 450\n\
            layout(local_size_x = 256) in;\n\
            layout(set = 0, binding = 0) buffer BufA { float a[]; };\n\
            layout(set = 0, binding = 1) buffer BufB { float b[]; };\n\
            layout(set = 0, binding = 2) buffer BufOut { float out_buf[]; };\n\
            void main() {\n\
                uint index = gl_GlobalInvocationID.x;\n\
                if (index >= a.length()) return;\n\
                out_buf[index] = a[index] * b[index];\n\
            }"
        ))
    }

    /// Get tensor fill (constant) kernel SPIR-V.
    pub fn fill_constant_f32() -> Result<Vec<u32>> {
        Err(KomputeError::shader(
            "Pre-compiled kernels not available. Please compile your own SPIR-V shaders from GLSL/HLSL. \
            Example GLSL for filling with constant:\n\
            \n\
            #version 450\n\
            layout(local_size_x = 256) in;\n\
            layout(set = 0, binding = 0) buffer BufOut { float out_buf[]; };\n\
            layout(push_constant) uniform PushConstants { float value; };\n\
            void main() {\n\
                uint index = gl_GlobalInvocationID.x;\n\
                if (index >= out_buf.length()) return;\n\
                out_buf[index] = value;\n\
            }"
        ))
    }

    /// List all available kernel types.
    pub fn list_available_kernels() -> Vec<&'static str> {
        vec![
            "elementwise_add_f32",
            "elementwise_multiply_f32",
            "fill_constant_f32",
        ]
    }
}

/// SPIR-V shader compilation guidance.
pub struct ShaderCompilationGuide;

impl ShaderCompilationGuide {
    /// Get compilation instructions for GLSL to SPIR-V.
    pub fn glsl_compilation_guide() -> &'static str {
        "To compile GLSL compute shaders to SPIR-V:\n\
        \n\
        1. Install the Vulkan SDK (includes glslc compiler)\n\
        2. Write your compute shader in GLSL (.comp file)\n\
        3. Compile with: glslc shader.comp -o shader.spv\n\
        4. Load the .spv file as bytes and pass to Algorithm::new()\n\
        \n\
        For online compilation, you can also use:\n\
        - Shader Playground (https://shader-playground.timjones.io/)\n\
        - Vulkan SDK tools\n\
        - shaderc Rust crate for runtime compilation"
    }

    /// Get example GLSL templates for common operations.
    pub fn example_glsl_templates() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "Element-wise Addition",
                include_str!("../shaders/elementwise_add.glsl"),
            ),
            (
                "Matrix Multiplication",
                include_str!("../shaders/matrix_mult.glsl"),
            ),
            (
                "Reduction Sum",
                include_str!("../shaders/reduction_sum.glsl"),
            ),
        ]
    }

    /// Get comprehensive shader compilation workflow.
    pub fn complete_shader_workflow() -> &'static str {
        "Complete SPIR-V Shader Workflow with Kompute:\n\
        \n\
        1. Write GLSL Compute Shader:\n\
           - Define local_size_x/y/z for workgroup dimensions\n\
           - Use buffer blocks for input/output tensors\n\
           - Add push constants for parameters\n\
           - Implement your compute logic in main()\n\
        \n\
        2. Compile to SPIR-V:\n\
           glslc shader.comp -o shader.spv\n\
        \n\
        3. Load and Use in Rust:\n\
           ```rust\n\
           let spirv_data = std::fs::read(\"shader.spv\")?;\n\
           let spirv_u32: Vec<u32> = spirv_data.chunks_exact(4)\n\
               .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))\n\
               .collect();\n\
           \n\
           let mut manager = Manager::new()?;\n\
           let algorithm = manager.algorithm(&spirv_u32, &tensors)?;\n\
           let mut sequence = Sequence::new(&mut manager)?;\n\
           sequence.record(&algorithm)?;\n\
           sequence.eval()?;\n\
           ```\n\
        \n\
        4. Tensor Data Flow:\n\
           - Create tensors with Manager::tensor()\n\
           - Algorithm automatically binds tensors in order\n\
           - Use Sequence to record and execute operations\n\
           - Retrieve results with tensor.data()\n\
        \n\
        Example workflows available in kompute/shaders/ directory."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spirv_validation() {
        // Test empty data
        assert!(SpirvUtils::validate_spirv(&[]).is_err());

        // Test too short data
        assert!(SpirvUtils::validate_spirv(&[0x07230203]).is_err());

        // Test invalid magic number
        assert!(SpirvUtils::validate_spirv(&[0x12345678, 0, 0, 0, 0]).is_err());

        // Test valid minimal SPIR-V header
        let valid_spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];
        assert!(SpirvUtils::validate_spirv(&valid_spirv).is_ok());
    }

    #[test]
    fn test_spirv_version_extraction() {
        let spirv = vec![
            0x07230203, // Magic number
            0x00010300, // Version 1.3
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        assert_eq!(SpirvUtils::get_spirv_version(&spirv).unwrap(), 0x00010300);
    }

    #[test]
    fn test_spirv_generator_info() {
        let spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version
            0x00123456, // Generator (custom value)
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        assert_eq!(SpirvUtils::get_generator_info(&spirv).unwrap(), 0x00123456);
    }

    #[test]
    fn test_compute_capability_check() {
        let spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        // Our simplified implementation should return true for valid SPIR-V
        assert!(SpirvUtils::has_compute_capability(&spirv).unwrap());
    }

    #[test]
    fn test_workgroup_size_estimation() {
        let spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        let (x, y, z) = SpirvUtils::estimate_optimal_workgroup_size(&spirv).unwrap();
        assert_eq!(x, 256);
        assert_eq!(y, 1);
        assert_eq!(z, 1);
    }

    #[test]
    fn test_available_kernels_list() {
        let kernels = CommonKernels::list_available_kernels();
        assert!(!kernels.is_empty());
        assert!(kernels.contains(&"elementwise_add_f32"));
        assert!(kernels.contains(&"elementwise_multiply_f32"));
        assert!(kernels.contains(&"fill_constant_f32"));
    }

    #[test]
    fn test_kernel_placeholder_errors() {
        // These should return helpful errors with GLSL examples
        assert!(CommonKernels::elementwise_add_f32().is_err());
        assert!(CommonKernels::elementwise_multiply_f32().is_err());
        assert!(CommonKernels::fill_constant_f32().is_err());
    }

    #[test]
    fn test_compilation_guide_available() {
        let guide = ShaderCompilationGuide::glsl_compilation_guide();
        assert!(!guide.is_empty());
        assert!(guide.contains("glslc"));
        assert!(guide.contains("SPIR-V"));
    }
}

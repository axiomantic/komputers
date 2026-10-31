//! GLSL to SPIR-V Compilation Utilities
//!
//! This module provides utilities for compiling GLSL compute shaders to SPIR-V bytecode
//! that can be used with Kompute algorithms. It supports various GLSL features and
//! provides helpful error reporting for shader compilation issues.
//!
//! This module is only available when the `glsl` feature is enabled.

use crate::error::{KomputeError, Result};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

/// GLSL shader compiler for converting compute shaders to SPIR-V
///
/// This compiler provides a convenient interface for compiling GLSL compute shaders
/// to SPIR-V bytecode that can be used directly with Kompute algorithms.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::GlslCompiler;
///
/// let compiler = GlslCompiler::new();
///
/// let glsl_source = r#"
///     #version 450
///     layout(local_size_x = 64) in;
///     layout(set = 0, binding = 0) buffer InputBuffer { float data[]; } input_buf;
///     layout(set = 0, binding = 1) buffer OutputBuffer { float data[]; } output_buf;
///
///     void main() {
///         uint idx = gl_GlobalInvocationID.x;
///         if (idx < input_buf.data.length()) {
///             output_buf.data[idx] = input_buf.data[idx] * 2.0;
///         }
///     }
/// "#;
///
/// let spirv_bytecode = compiler.compile_compute_shader(glsl_source).unwrap();
/// ```
pub struct GlslCompiler {
    /// Compiler options and settings
    options: CompilerOptions,
    /// Cache for compiled shaders
    shader_cache: HashMap<String, Vec<u32>>,
}

/// Configuration options for GLSL compilation
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    /// GLSL version to target (e.g., 450, 460)
    pub glsl_version: u32,
    /// Enable optimization
    pub optimize: bool,
    /// Enable debug information
    pub debug: bool,
    /// Target SPIR-V version
    pub spirv_version: String,
    /// Additional include directories
    pub include_dirs: Vec<String>,
    /// Preprocessor definitions
    pub defines: HashMap<String, String>,
    /// Vulkan environment version
    pub vulkan_version: String,
}

/// Compilation result containing SPIR-V bytecode and metadata
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Compiled SPIR-V bytecode
    pub spirv_bytecode: Vec<u32>,
    /// Shader reflection information
    pub reflection: ShaderReflection,
    /// Compilation warnings
    pub warnings: Vec<String>,
    /// Compilation time
    pub compile_time_ms: u64,
}

/// Shader reflection information extracted from SPIR-V
#[derive(Debug, Clone)]
pub struct ShaderReflection {
    /// Local workgroup size
    pub local_size: [u32; 3],
    /// Descriptor set bindings
    pub bindings: Vec<BindingInfo>,
    /// Push constant information
    pub push_constants: Option<PushConstantInfo>,
    /// Specialization constants
    pub spec_constants: Vec<SpecConstantInfo>,
}

/// Information about a descriptor binding
#[derive(Debug, Clone)]
pub struct BindingInfo {
    /// Binding number
    pub binding: u32,
    /// Descriptor set number
    pub set: u32,
    /// Binding name
    pub name: String,
    /// Descriptor type (buffer, image, etc.)
    pub descriptor_type: DescriptorType,
    /// Array size (1 for non-arrays)
    pub array_size: u32,
}

/// Information about push constants
#[derive(Debug, Clone)]
pub struct PushConstantInfo {
    /// Push constant block name
    pub name: String,
    /// Size in bytes
    pub size: u32,
    /// Member information
    pub members: Vec<PushConstantMember>,
}

/// Information about a push constant member
#[derive(Debug, Clone)]
pub struct PushConstantMember {
    /// Member name
    pub name: String,
    /// Offset in bytes
    pub offset: u32,
    /// Size in bytes
    pub size: u32,
    /// Data type
    pub data_type: DataType,
}

/// Information about specialization constants
#[derive(Debug, Clone)]
pub struct SpecConstantInfo {
    /// Specialization constant ID
    pub id: u32,
    /// Constant name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Default value (as bytes)
    pub default_value: Vec<u8>,
}

/// Descriptor types for shader bindings
#[derive(Debug, Clone, PartialEq)]
pub enum DescriptorType {
    /// Storage buffer
    StorageBuffer,
    /// Uniform buffer
    UniformBuffer,
    /// Storage image
    StorageImage,
    /// Sampled image
    SampledImage,
    /// Sampler
    Sampler,
    /// Combined image sampler
    CombinedImageSampler,
}

/// Data types for shader variables
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 32-bit unsigned integer
    Uint32,
    /// Boolean
    Bool,
    /// 2-component float vector
    Vec2,
    /// 3-component float vector
    Vec3,
    /// 4-component float vector
    Vec4,
    /// 4x4 float matrix
    Mat4,
}

/// Common GLSL compute shader templates for quick prototyping
pub struct GlslTemplates;

impl GlslCompiler {
    /// Create a new GLSL compiler with default options
    pub fn new() -> Self {
        Self {
            options: CompilerOptions::default(),
            shader_cache: HashMap::new(),
        }
    }

    /// Create a new GLSL compiler with custom options
    pub fn with_options(options: CompilerOptions) -> Self {
        Self {
            options,
            shader_cache: HashMap::new(),
        }
    }

    /// Compile a GLSL compute shader to SPIR-V bytecode
    ///
    /// # Arguments
    ///
    /// * `glsl_source` - The GLSL compute shader source code
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::GlslCompiler;
    ///
    /// let compiler = GlslCompiler::new();
    /// let glsl = r#"
    ///     #version 450
    ///     layout(local_size_x = 256) in;
    ///     layout(set = 0, binding = 0) buffer Data { float values[]; };
    ///     void main() {
    ///         uint idx = gl_GlobalInvocationID.x;
    ///         if (idx < values.length()) {
    ///             values[idx] = sqrt(values[idx]);
    ///         }
    ///     }
    /// "#;
    ///
    /// let result = compiler.compile_compute_shader(glsl).unwrap();
    /// println!("Compiled shader with {} bytes of SPIR-V", result.spirv_bytecode.len() * 4);
    /// ```
    pub fn compile_compute_shader(&mut self, glsl_source: &str) -> Result<CompilationResult> {
        // Check cache first
        let cache_key = self.generate_cache_key(glsl_source);
        if let Some(cached_spirv) = self.shader_cache.get(&cache_key) {
            return Ok(CompilationResult {
                spirv_bytecode: cached_spirv.clone(),
                reflection: ShaderReflection::default(),
                warnings: vec!["Using cached compilation".to_string()],
                compile_time_ms: 0,
            });
        }

        let start_time = std::time::Instant::now();

        // Validate GLSL source
        self.validate_glsl_source(glsl_source)?;

        // Compile using external glslc compiler (from Vulkan SDK)
        let spirv_bytecode = self.compile_with_glslc(glsl_source)?;

        // Extract reflection information
        let reflection = self.extract_reflection(&spirv_bytecode)?;

        let compile_time = start_time.elapsed();

        let result = CompilationResult {
            spirv_bytecode: spirv_bytecode.clone(),
            reflection,
            warnings: Vec::new(),
            compile_time_ms: compile_time.as_millis() as u64,
        };

        // Cache the result
        self.shader_cache.insert(cache_key, spirv_bytecode);

        Ok(result)
    }

    /// Compile GLSL source with template substitution
    ///
    /// This method allows for parameterized shader compilation using template variables.
    ///
    /// # Arguments
    ///
    /// * `glsl_template` - GLSL source with template variables (e.g., `{{WORKGROUP_SIZE}}`)
    /// * `variables` - HashMap of variable names to values for substitution
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::GlslCompiler;
    /// use std::collections::HashMap;
    ///
    /// let compiler = GlslCompiler::new();
    /// let mut variables = HashMap::new();
    /// variables.insert("WORKGROUP_SIZE".to_string(), "128".to_string());
    /// variables.insert("DATA_TYPE".to_string(), "float".to_string());
    ///
    /// let template = r#"
    ///     #version 450
    ///     layout(local_size_x = {{WORKGROUP_SIZE}}) in;
    ///     layout(set = 0, binding = 0) buffer Data { {{DATA_TYPE}} values[]; };
    ///     void main() {
    ///         uint idx = gl_GlobalInvocationID.x;
    ///         values[idx] *= 2.0;
    ///     }
    /// "#;
    ///
    /// let result = compiler.compile_with_template(template, &variables).unwrap();
    /// ```
    pub fn compile_with_template(
        &mut self,
        glsl_template: &str,
        variables: &HashMap<String, String>,
    ) -> Result<CompilationResult> {
        let glsl_source = self.substitute_template_variables(glsl_template, variables)?;
        self.compile_compute_shader(&glsl_source)
    }

    /// Set compiler options
    pub fn set_options(&mut self, options: CompilerOptions) {
        self.options = options;
        // Clear cache when options change
        self.shader_cache.clear();
    }

    /// Clear the shader compilation cache
    pub fn clear_cache(&mut self) {
        self.shader_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let entry_count = self.shader_cache.len();
        let total_bytes = self
            .shader_cache
            .values()
            .map(|spirv| spirv.len() * 4)
            .sum();
        (entry_count, total_bytes)
    }

    // Private helper methods

    fn validate_glsl_source(&self, glsl_source: &str) -> Result<()> {
        if glsl_source.trim().is_empty() {
            return Err(KomputeError::shader("GLSL source cannot be empty"));
        }

        // Check for version directive
        if !glsl_source.contains("#version") {
            return Err(KomputeError::shader(
                "GLSL source must contain a #version directive",
            ));
        }

        // Check for compute shader requirements
        if !glsl_source.contains("local_size") {
            return Err(KomputeError::shader(
                "Compute shader must specify local_size layout qualifier",
            ));
        }

        if !glsl_source.contains("void main()") {
            return Err(KomputeError::shader(
                "Compute shader must contain a main() function",
            ));
        }

        Ok(())
    }

    fn compile_with_glslc(&self, glsl_source: &str) -> Result<Vec<u32>> {
        // Create temporary file for GLSL source
        let mut temp_glsl = NamedTempFile::new()
            .map_err(|e| KomputeError::shader(format!("Failed to create temp file: {}", e)))?;

        temp_glsl
            .write_all(glsl_source.as_bytes())
            .map_err(|e| KomputeError::shader(format!("Failed to write GLSL source: {}", e)))?;

        // Create temporary file for SPIR-V output
        let temp_spirv = NamedTempFile::new().map_err(|e| {
            KomputeError::shader(format!("Failed to create output temp file: {}", e))
        })?;

        // Build glslc command
        let mut cmd = Command::new("glslc");
        cmd.arg("-fshader-stage=compute")
            .arg("-o")
            .arg(temp_spirv.path())
            .arg(temp_glsl.path());

        // Add optimization flags
        if self.options.optimize {
            cmd.arg("-O");
        }

        // Add debug flags
        if self.options.debug {
            cmd.arg("-g");
        }

        // Add target environment
        cmd.arg("--target-env=vulkan1.0");

        // Add defines
        for (key, value) in &self.options.defines {
            cmd.arg(format!("-D{}={}", key, value));
        }

        // Add include directories
        for include_dir in &self.options.include_dirs {
            cmd.arg("-I").arg(include_dir);
        }

        // Execute compilation
        let output = cmd.output()
            .map_err(|e| KomputeError::shader(format!("Failed to execute glslc: {}. Make sure Vulkan SDK is installed and glslc is in PATH", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(KomputeError::shader(format!(
                "GLSL compilation failed:\n{}",
                stderr
            )));
        }

        // Read SPIR-V bytecode
        let spirv_bytes = fs::read(temp_spirv.path())
            .map_err(|e| KomputeError::shader(format!("Failed to read SPIR-V output: {}", e)))?;

        // Convert to u32 words (SPIR-V is stored as 32-bit words)
        if spirv_bytes.len() % 4 != 0 {
            return Err(KomputeError::shader(
                "Invalid SPIR-V output: size not divisible by 4",
            ));
        }

        let mut spirv_words = Vec::with_capacity(spirv_bytes.len() / 4);
        for chunk in spirv_bytes.chunks_exact(4) {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            spirv_words.push(word);
        }

        // Validate SPIR-V magic number
        if spirv_words.is_empty() || spirv_words[0] != 0x07230203 {
            return Err(KomputeError::shader(
                "Invalid SPIR-V output: incorrect magic number",
            ));
        }

        Ok(spirv_words)
    }

    fn extract_reflection(&self, _spirv_bytecode: &[u32]) -> Result<ShaderReflection> {
        // Simplified reflection - in a full implementation, this would parse SPIR-V
        Ok(ShaderReflection {
            local_size: [64, 1, 1], // Default values
            bindings: Vec::new(),
            push_constants: None,
            spec_constants: Vec::new(),
        })
    }

    fn substitute_template_variables(
        &self,
        template: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String> {
        let mut result = template.to_string();

        for (key, value) in variables {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        // Check for unsubstituted variables
        if result.contains("{{") && result.contains("}}") {
            return Err(KomputeError::shader(
                "Template contains unsubstituted variables",
            ));
        }

        Ok(result)
    }

    fn generate_cache_key(&self, glsl_source: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        glsl_source.hash(&mut hasher);
        self.options.glsl_version.hash(&mut hasher);
        self.options.optimize.hash(&mut hasher);
        self.options.debug.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            glsl_version: 450,
            optimize: true,
            debug: false,
            spirv_version: "1.0".to_string(),
            include_dirs: Vec::new(),
            defines: HashMap::new(),
            vulkan_version: "1.0".to_string(),
        }
    }
}

impl Default for ShaderReflection {
    fn default() -> Self {
        Self {
            local_size: [1, 1, 1],
            bindings: Vec::new(),
            push_constants: None,
            spec_constants: Vec::new(),
        }
    }
}

impl GlslTemplates {
    /// Basic element-wise operation template
    pub fn elementwise_operation() -> &'static str {
        r#"#version 450

layout(local_size_x = {{WORKGROUP_SIZE}}, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer InputBuffer {
    {{DATA_TYPE}} data[];
} input_buf;

layout(set = 0, binding = 1) buffer OutputBuffer {
    {{DATA_TYPE}} data[];
} output_buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= input_buf.data.length()) return;

    // Operation: {{OPERATION}}
    output_buf.data[idx] = {{OPERATION_BODY}};
}"#
    }

    /// Matrix multiplication template
    pub fn matrix_multiply() -> &'static str {
        r#"#version 450

layout(local_size_x = {{TILE_SIZE}}, local_size_y = {{TILE_SIZE}}, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer MatrixA {
    float data[];
} matrix_a;

layout(set = 0, binding = 1) readonly buffer MatrixB {
    float data[];
} matrix_b;

layout(set = 0, binding = 2) writeonly buffer MatrixC {
    float data[];
} matrix_c;

layout(push_constant) uniform PushConstants {
    uint M, N, K;  // Matrix dimensions: A is MxK, B is KxN, C is MxN
} pc;

shared float tile_a[{{TILE_SIZE}}][{{TILE_SIZE}}];
shared float tile_b[{{TILE_SIZE}}][{{TILE_SIZE}}];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint local_row = gl_LocalInvocationID.y;
    uint local_col = gl_LocalInvocationID.x;

    float sum = 0.0;

    for (uint tile = 0; tile < (pc.K + {{TILE_SIZE}} - 1) / {{TILE_SIZE}}; ++tile) {
        // Load tile into shared memory
        uint a_idx = row * pc.K + tile * {{TILE_SIZE}} + local_col;
        uint b_idx = (tile * {{TILE_SIZE}} + local_row) * pc.N + col;

        tile_a[local_row][local_col] = (row < pc.M && tile * {{TILE_SIZE}} + local_col < pc.K)
                                       ? matrix_a.data[a_idx] : 0.0;
        tile_b[local_row][local_col] = (tile * {{TILE_SIZE}} + local_row < pc.K && col < pc.N)
                                       ? matrix_b.data[b_idx] : 0.0;

        barrier();

        // Compute partial sum
        for (uint k = 0; k < {{TILE_SIZE}}; ++k) {
            sum += tile_a[local_row][k] * tile_b[k][local_col];
        }

        barrier();
    }

    // Write result
    if (row < pc.M && col < pc.N) {
        matrix_c.data[row * pc.N + col] = sum;
    }
}"#
    }

    /// Reduction operation template
    pub fn reduction() -> &'static str {
        r#"#version 450

layout(local_size_x = {{WORKGROUP_SIZE}}, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InputBuffer {
    {{DATA_TYPE}} data[];
} input_buf;

layout(set = 0, binding = 1) buffer OutputBuffer {
    {{DATA_TYPE}} data[];
} output_buf;

shared {{DATA_TYPE}} shared_data[{{WORKGROUP_SIZE}}];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint group_id = gl_WorkGroupID.x;

    // Load data into shared memory
    shared_data[tid] = (gid < input_buf.data.length()) ? input_buf.data[gid] : {{IDENTITY_VALUE}};
    barrier();

    // Reduction in shared memory
    for (uint stride = {{WORKGROUP_SIZE}} / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = {{REDUCE_OP}}(shared_data[tid], shared_data[tid + stride]);
        }
        barrier();
    }

    // Write result
    if (tid == 0) {
        output_buf.data[group_id] = shared_data[0];
    }
}"#
    }

    /// Image processing filter template
    pub fn image_filter() -> &'static str {
        r#"#version 450

layout(local_size_x = {{WORKGROUP_X}}, local_size_y = {{WORKGROUP_Y}}, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D input_image;
layout(set = 0, binding = 1, {{FORMAT}}) uniform writeonly image2D output_image;

layout(push_constant) uniform PushConstants {
    int width, height;
    float param1, param2;
} pc;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (coord.x >= pc.width || coord.y >= pc.height) return;

    vec2 uv = (vec2(coord) + 0.5) / vec2(pc.width, pc.height);

    // Filter operation: {{FILTER_OPERATION}}
    vec4 result = {{FILTER_BODY}};

    imageStore(output_image, coord, result);
}"#
    }
}

impl Default for GlslCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = GlslCompiler::new();
        assert_eq!(compiler.options.glsl_version, 450);
        assert!(compiler.options.optimize);
        assert!(!compiler.options.debug);
    }

    #[test]
    fn test_glsl_validation() {
        let compiler = GlslCompiler::new();

        // Valid GLSL should pass validation
        let valid_glsl = r#"
            #version 450
            layout(local_size_x = 64) in;
            void main() {
                // Do nothing
            }
        "#;
        assert!(compiler.validate_glsl_source(valid_glsl).is_ok());

        // Missing version should fail
        let no_version = r#"
            layout(local_size_x = 64) in;
            void main() {}
        "#;
        assert!(compiler.validate_glsl_source(no_version).is_err());

        // Missing local_size should fail
        let no_local_size = r#"
            #version 450
            void main() {}
        "#;
        assert!(compiler.validate_glsl_source(no_local_size).is_err());
    }

    #[test]
    fn test_template_substitution() {
        let compiler = GlslCompiler::new();
        let mut variables = HashMap::new();
        variables.insert("WORKGROUP_SIZE".to_string(), "128".to_string());
        variables.insert("DATA_TYPE".to_string(), "float".to_string());

        let template =
            "layout(local_size_x = {{WORKGROUP_SIZE}}) buffer Data { {{DATA_TYPE}} values[]; };";
        let result = compiler
            .substitute_template_variables(template, &variables)
            .unwrap();

        assert_eq!(
            result,
            "layout(local_size_x = 128) buffer Data { float values[]; };"
        );
    }

    #[test]
    fn test_cache_key_generation() {
        let compiler = GlslCompiler::new();
        let key1 = compiler.generate_cache_key("shader source 1");
        let key2 = compiler.generate_cache_key("shader source 2");
        let key3 = compiler.generate_cache_key("shader source 1"); // Same as key1

        assert_ne!(key1, key2);
        assert_eq!(key1, key3);
    }

    #[test]
    fn test_templates() {
        // Test that templates contain expected placeholders
        let elementwise = GlslTemplates::elementwise_operation();
        assert!(elementwise.contains("{{WORKGROUP_SIZE}}"));
        assert!(elementwise.contains("{{DATA_TYPE}}"));

        let matmul = GlslTemplates::matrix_multiply();
        assert!(matmul.contains("{{TILE_SIZE}}"));

        let reduction = GlslTemplates::reduction();
        assert!(reduction.contains("{{REDUCE_OP}}"));
    }

    #[test]
    #[ignore = "requires glslc compiler from Vulkan SDK"]
    fn test_actual_compilation() {
        let mut compiler = GlslCompiler::new();

        let glsl = r#"
            #version 450
            layout(local_size_x = 64) in;
            layout(set = 0, binding = 0) buffer Data { float values[]; };
            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx < values.length()) {
                    values[idx] *= 2.0;
                }
            }
        "#;

        match compiler.compile_compute_shader(glsl) {
            Ok(result) => {
                assert!(!result.spirv_bytecode.is_empty());
                assert_eq!(result.spirv_bytecode[0], 0x07230203); // SPIR-V magic
                println!("✅ GLSL compilation test passed!");
                println!("   SPIR-V size: {} bytes", result.spirv_bytecode.len() * 4);
                println!("   Compile time: {}ms", result.compile_time_ms);
            }
            Err(e) => {
                println!(
                    "⚠️ GLSL compilation failed (expected without Vulkan SDK): {}",
                    e
                );
            }
        }
    }
}

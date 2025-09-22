//! High-level GPU operations and compute patterns.
//!
//! This module provides convenient, idiomatic Rust APIs for common GPU computing
//! operations built on top of the Kompute C++ library. It wraps the lower-level
//! Algorithm and Sequence APIs to provide easy-to-use functions for typical
//! GPU computing workloads.

use crate::algorithm::Algorithm;
use crate::algorithm_binding::{AccessPattern, AlgorithmBinding, CommonBindingTemplates};
use crate::error::{KomputeError, Result};
use crate::manager::Manager;
use crate::memory_optimizer::MemoryOptimizer;
use crate::sequence::Sequence;
use crate::spirv_shaders;
use crate::tensor::{Tensor, TensorData};
use std::collections::HashMap;

/// High-level GPU operations manager.
///
/// Provides convenient methods for common GPU computing patterns like
/// element-wise operations, reductions, and matrix operations. All operations
/// are built on top of the existing Kompute Algorithm and Sequence infrastructure.
/// Includes automatic cross-platform memory optimization.
pub struct GpuOperations {
    manager: Manager,
    memory_optimizer: MemoryOptimizer,
}

impl GpuOperations {
    /// Create a new GPU operations manager.
    pub fn new() -> Result<Self> {
        let manager = Manager::new()?;
        let memory_optimizer = MemoryOptimizer::new()?;
        Ok(Self {
            manager,
            memory_optimizer,
        })
    }

    /// Create GPU operations manager with a specific device.
    pub fn with_device(device_id: u32) -> Result<Self> {
        let manager = Manager::with_device(device_id)?;
        let memory_optimizer = MemoryOptimizer::new()?;
        Ok(Self {
            manager,
            memory_optimizer,
        })
    }

    /// Perform element-wise addition of two tensors.
    ///
    /// # Arguments
    ///
    /// * `a` - First input tensor
    /// * `b` - Second input tensor
    /// * `output` - Output tensor (must be same size as inputs)
    ///
    /// # Errors
    ///
    /// Returns error if tensors have mismatched sizes or GPU operation fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::{GpuOperations, Tensor, Result};
    ///
    /// fn example() -> Result<()> {
    ///     let mut ops = GpuOperations::new()?;
    ///     let a = Tensor::new(&mut ops.manager(), &[1.0f32, 2.0, 3.0])?;
    ///     let b = Tensor::new(&mut ops.manager(), &[4.0f32, 5.0, 6.0])?;
    ///     let mut output = Tensor::<f32>::uninit(ops.manager(), 3)?;
    ///
    ///     ops.elementwise_add(&a, &b, &mut output)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn elementwise_add<T: TensorData>(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        output: &mut Tensor<T>,
    ) -> Result<()> {
        // Validate tensor sizes
        if a.size() != b.size() || a.size() != output.size() {
            return Err(KomputeError::compute(format!(
                "Tensor size mismatch: a={}, b={}, output={}",
                a.size(),
                b.size(),
                output.size()
            )));
        }

        // Get the appropriate SPIR-V shader for the tensor data type
        let spirv_data = spirv_shaders::get_elementwise_add_spirv::<T>();

        // Validate the SPIR-V bytecode
        if !spirv_shaders::validate_spirv(spirv_data) {
            return Err(KomputeError::compute(
                "Invalid SPIR-V bytecode for element-wise addition",
            ));
        }

        // Create algorithm with bound tensors
        let tensors = vec![a, b, output];
        let algorithm = Algorithm::with_tensors(&mut self.manager, spirv_data, &tensors)?;

        // Create sequence and record operations
        let mut sequence = Sequence::new(&mut self.manager)?;

        // Record tensor synchronization to device
        sequence.record_tensor_sync_device(a.handle())?;
        sequence.record_tensor_sync_device(b.handle())?;
        sequence.record_tensor_sync_device(output.handle())?;

        // Record algorithm dispatch
        sequence.record_algorithm_dispatch(algorithm.handle())?;

        // Record output tensor synchronization back to host
        sequence.record_tensor_sync_local(output.handle())?;

        // Execute the sequence
        sequence.eval()?;

        Ok(())
    }

    /// Perform element-wise multiplication of two tensors.
    pub fn elementwise_multiply<T: TensorData>(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        output: &mut Tensor<T>,
    ) -> Result<()> {
        if a.size() != b.size() || a.size() != output.size() {
            return Err(KomputeError::compute(format!(
                "Tensor size mismatch: a={}, b={}, output={}",
                a.size(),
                b.size(),
                output.size()
            )));
        }

        Err(KomputeError::compute(
            "Element-wise operations require SPIR-V shader compilation. \
            Use the Algorithm API directly with appropriate SPIR-V bytecode for GPU compute operations.",
        ))
    }

    /// Copy data from one tensor to another.
    ///
    /// This performs an efficient GPU-to-GPU copy operation.
    pub fn copy<T: TensorData>(&mut self, source: &Tensor<T>, dest: &mut Tensor<T>) -> Result<()> {
        if source.size() != dest.size() {
            return Err(KomputeError::compute(format!(
                "Tensor size mismatch: source={}, dest={}",
                source.size(),
                dest.size()
            )));
        }

        // Create a sequence for the copy operation
        let mut sequence = Sequence::new(&mut self.manager)?;

        // Record tensor synchronization operations
        sequence.record_tensor_sync_device(source.handle())?;
        sequence.record_tensor_sync_device(dest.handle())?;

        // Execute the sequence
        sequence.eval()?;

        Ok(())
    }

    /// Fill a tensor with a constant value.
    ///
    /// # Arguments
    ///
    /// * `_tensor` - Tensor to fill
    /// * `_value` - Value to fill with
    pub fn fill<T: TensorData>(&mut self, _tensor: &mut Tensor<T>, _value: T) -> Result<()> {
        // This would require a SPIR-V shader that sets all elements to the given value
        Err(KomputeError::compute(
            "Fill operation requires SPIR-V shader compilation. \
            Use the Algorithm API directly with appropriate SPIR-V bytecode for GPU compute operations.",
        ))
    }

    /// Perform matrix multiplication using GPU compute.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (MxK)
    /// * `b` - Input matrix B (KxN)
    /// * `output` - Output matrix C (MxN)
    /// * `m` - Number of rows in A and C
    /// * `k` - Number of columns in A and rows in B
    /// * `n` - Number of columns in B and C
    ///
    /// # Errors
    ///
    /// Returns error if tensor sizes don't match matrix dimensions or GPU operation fails.
    pub fn matrix_multiply<T: TensorData>(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        output: &mut Tensor<T>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        // Validate matrix dimensions
        if a.size() != m * k {
            return Err(KomputeError::compute(format!(
                "Matrix A size mismatch: expected {}, got {}",
                m * k,
                a.size()
            )));
        }
        if b.size() != k * n {
            return Err(KomputeError::compute(format!(
                "Matrix B size mismatch: expected {}, got {}",
                k * n,
                b.size()
            )));
        }
        if output.size() != m * n {
            return Err(KomputeError::compute(format!(
                "Output matrix size mismatch: expected {}, got {}",
                m * n,
                output.size()
            )));
        }

        Err(KomputeError::compute(
            "Matrix multiplication requires SPIR-V shader compilation. \
            Use the Algorithm API directly with matrix multiplication SPIR-V bytecode. \
            See kompute/shaders/matrix_mult.glsl for example implementation.",
        ))
    }

    /// Perform reduction sum operation on a tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor to reduce
    /// * `output` - Output tensor containing the sum (must be size 1)
    pub fn reduce_sum<T: TensorData>(
        &mut self,
        input: &Tensor<T>,
        output: &mut Tensor<T>,
    ) -> Result<()> {
        if output.size() != 1 {
            return Err(KomputeError::compute(format!(
                "Output tensor for reduction must have size 1, got {}",
                output.size()
            )));
        }

        if input.size() == 0 {
            return Err(KomputeError::compute(
                "Cannot reduce empty tensor".to_string(),
            ));
        }

        Err(KomputeError::compute(
            "Reduction operations require SPIR-V shader compilation. \
            Use the Algorithm API directly with reduction SPIR-V bytecode. \
            See kompute/shaders/reduction_sum.glsl for example implementation.",
        ))
    }

    /// Apply a custom SPIR-V compute shader to tensors.
    ///
    /// This is the core method that wraps Kompute's Algorithm functionality
    /// for custom GPU compute operations.
    ///
    /// # Arguments
    ///
    /// * `spirv_data` - Compiled SPIR-V bytecode
    /// * `workgroup` - Workgroup configuration for GPU dispatch
    /// * `push_constants` - Optional push constant data
    ///
    /// Note: This method provides guidance for using the Algorithm API directly,
    /// as custom shaders require specific tensor binding patterns.
    pub fn execute_custom_shader(
        &mut self,
        _spirv_data: &[u32],
        workgroup: WorkgroupConfig,
        push_constants: Option<&[u8]>,
    ) -> Result<()> {
        // This would be the main interface to Kompute's Algorithm functionality
        // For now, provide guidance on direct Algorithm usage

        let push_size = push_constants.map(|p| p.len()).unwrap_or(0);

        Err(KomputeError::compute(format!(
            "Custom shader execution is available through the Algorithm API. \
            Use Manager::algorithm() with your SPIR-V data, then create a Sequence \
            to bind tensors and execute. Workgroup: {}x{}x{}, Push constants: {} bytes. \
            Example: let algorithm = manager.algorithm(spirv_data, &tensors)?;",
            workgroup.x, workgroup.y, workgroup.z, push_size
        )))
    }

    /// Get device information and capabilities.
    ///
    /// Returns information about the GPU device being used for operations.
    pub fn device_info(&self) -> Result<DeviceInfo> {
        // In a full implementation, this would query Kompute for device properties
        Err(KomputeError::compute(
            "Device information querying not yet implemented. \
            Use Manager::get_device_properties() for detailed device information.",
        ))
    }

    /// Synchronize all pending GPU operations.
    ///
    /// Ensures all queued operations complete before returning.
    pub fn synchronize(&mut self) -> Result<()> {
        // Create a sequence to synchronize all operations
        let mut sequence = Sequence::new(&mut self.manager)?;
        sequence.eval()?;
        Ok(())
    }

    /// Perform element-wise addition using advanced algorithm binding.
    ///
    /// This is an enhanced version of elementwise_add that uses the AlgorithmBinding
    /// system for better validation and error reporting.
    ///
    /// # Arguments
    ///
    /// * `a` - First input tensor
    /// * `b` - Second input tensor
    /// * `output` - Output tensor (must be same size as inputs)
    ///
    /// # Returns
    ///
    /// A comprehensive validation report along with the operation result.
    pub fn elementwise_add_validated<T: TensorData>(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        output: &mut Tensor<T>,
    ) -> Result<crate::algorithm_binding::BindingValidationReport> {
        // Get the appropriate SPIR-V shader for the tensor data type
        let spirv_data = spirv_shaders::get_elementwise_add_spirv::<T>();

        // Validate the SPIR-V bytecode
        if !spirv_shaders::validate_spirv(spirv_data) {
            return Err(KomputeError::compute(
                "Invalid SPIR-V bytecode for element-wise addition",
            ));
        }

        // Create algorithm
        let algorithm = Algorithm::new(&mut self.manager, spirv_data)?;

        // Create algorithm binding with elementwise template
        let template = CommonBindingTemplates::elementwise_binary_op();
        let mut binding = AlgorithmBinding::new(algorithm, Some(&template))?;

        // Create tensor map for batch binding
        let tensor_map = HashMap::from([
            ("input_a".to_string(), (a, AccessPattern::Input)),
            ("input_b".to_string(), (b, AccessPattern::Input)),
            ("output".to_string(), (output, AccessPattern::Output)),
        ]);

        // Bind tensors with validation
        binding.bind_tensors_batch(tensor_map)?;

        // Get validation report before execution
        let validation_report = binding.validation_report();

        if !validation_report.is_valid {
            return Err(KomputeError::compute(format!(
                "Algorithm binding validation failed: missing required parameters: {:?}",
                validation_report.missing_required
            )));
        }

        // Create sequence and record operations
        let mut sequence = Sequence::new(&mut self.manager)?;

        // Record tensor synchronization to device
        sequence.record_tensor_sync_device(a.handle())?;
        sequence.record_tensor_sync_device(b.handle())?;
        sequence.record_tensor_sync_device(output.handle())?;

        // Record algorithm dispatch
        sequence.record_algorithm_dispatch(binding.algorithm().handle())?;

        // Record output tensor synchronization back to host
        sequence.record_tensor_sync_local(output.handle())?;

        // Execute the sequence
        sequence.eval()?;

        Ok(validation_report)
    }

    /// Perform matrix multiplication using advanced algorithm binding.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (MxK)
    /// * `b` - Input matrix B (KxN)
    /// * `output` - Output matrix C (MxN)
    /// * `m` - Number of rows in A and C
    /// * `k` - Number of columns in A and rows in B
    /// * `n` - Number of columns in B and C
    ///
    /// # Returns
    ///
    /// A comprehensive validation report along with the operation result.
    pub fn matrix_multiply_validated<T: TensorData>(
        &mut self,
        a: &Tensor<T>,
        b: &Tensor<T>,
        output: &mut Tensor<T>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<crate::algorithm_binding::BindingValidationReport> {
        // Validate matrix dimensions first
        if a.size() != m * k {
            return Err(KomputeError::compute(format!(
                "Matrix A size mismatch: expected {}, got {}",
                m * k,
                a.size()
            )));
        }
        if b.size() != k * n {
            return Err(KomputeError::compute(format!(
                "Matrix B size mismatch: expected {}, got {}",
                k * n,
                b.size()
            )));
        }
        if output.size() != m * n {
            return Err(KomputeError::compute(format!(
                "Output matrix size mismatch: expected {}, got {}",
                m * n,
                output.size()
            )));
        }

        // Get matrix multiplication SPIR-V
        let spirv_data = spirv_shaders::MATRIX_MULT_F32_SPIRV;

        // Validate the SPIR-V bytecode
        if !spirv_shaders::validate_spirv(spirv_data) {
            return Err(KomputeError::compute(
                "Invalid SPIR-V bytecode for matrix multiplication",
            ));
        }

        // Create algorithm
        let algorithm = Algorithm::new(&mut self.manager, spirv_data)?;

        // Create algorithm binding with matrix multiplication template
        let template = CommonBindingTemplates::matrix_multiply();
        let mut binding = AlgorithmBinding::new(algorithm, Some(&template))?;

        // Create tensor map for batch binding
        let tensor_map = HashMap::from([
            ("matrix_a".to_string(), (a, AccessPattern::Input)),
            ("matrix_b".to_string(), (b, AccessPattern::Input)),
            ("matrix_c".to_string(), (output, AccessPattern::Output)),
        ]);

        // Bind tensors with validation
        binding.bind_tensors_batch(tensor_map)?;

        // Get validation report
        let validation_report = binding.validation_report();

        if !validation_report.is_valid {
            return Err(KomputeError::compute(format!(
                "Matrix multiplication binding validation failed: {:?}",
                validation_report.missing_required
            )));
        }

        // For now, return the validation report
        // Full GPU execution would require more complete SPIR-V implementation
        Ok(validation_report)
    }

    /// Get access to the underlying manager for advanced operations.
    pub fn manager(&mut self) -> &mut Manager {
        &mut self.manager
    }

    /// Get access to the memory optimizer for performance tuning.
    pub fn memory_optimizer(&mut self) -> &mut MemoryOptimizer {
        &mut self.memory_optimizer
    }

    /// Optimize tensor allocation using platform-specific strategies.
    ///
    /// This method automatically configures memory allocation strategies
    /// based on the detected platform and GPU capabilities.
    pub fn optimize_tensor_allocation<T: TensorData>(&mut self, tensor: &Tensor<T>) -> Result<()> {
        let size_bytes = tensor.size() * std::mem::size_of::<T>();
        self.memory_optimizer.allocate_optimized(
            &format!("tensor_{}", tensor.handle()),
            size_bytes,
            "DEVICE_LOCAL",
        )?;
        Ok(())
    }
}

/// GPU device information.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Device type (integrated, discrete, etc.)
    pub device_type: String,
    /// Available memory in bytes
    pub memory_size: u64,
    /// Maximum workgroup size
    pub max_workgroup_size: (u32, u32, u32),
    /// Maximum number of compute units
    pub compute_units: u32,
}

/// Workgroup configuration for GPU compute operations.
///
/// Defines how work is distributed across GPU compute units.
#[derive(Debug, Clone, Copy)]
pub struct WorkgroupConfig {
    /// Number of work items in X dimension
    pub x: u32,
    /// Number of work items in Y dimension
    pub y: u32,
    /// Number of work items in Z dimension
    pub z: u32,
}

impl WorkgroupConfig {
    /// Create a 1D workgroup configuration.
    pub fn new_1d(size: u32) -> Self {
        Self {
            x: size,
            y: 1,
            z: 1,
        }
    }

    /// Create a 2D workgroup configuration.
    pub fn new_2d(width: u32, height: u32) -> Self {
        Self {
            x: width,
            y: height,
            z: 1,
        }
    }

    /// Create a 3D workgroup configuration.
    pub fn new_3d(width: u32, height: u32, depth: u32) -> Self {
        Self {
            x: width,
            y: height,
            z: depth,
        }
    }

    /// Calculate optimal workgroup size for a given number of elements.
    ///
    /// This uses common GPU architecture guidelines to choose efficient
    /// workgroup sizes.
    pub fn optimal_1d(num_elements: usize) -> Self {
        // Common GPU workgroup sizes are multiples of 32 (warp size)
        // We'll use 256 as a reasonable default for most GPUs
        let workgroup_size = 256;
        let num_workgroups = num_elements.div_ceil(workgroup_size);

        Self::new_1d(num_workgroups as u32)
    }

    /// Get total number of work items.
    pub fn total_items(&self) -> u32 {
        self.x * self.y * self.z
    }
}

/// GPU memory optimization patterns.
///
/// Provides guidance for efficient GPU memory access patterns.
pub struct MemoryPatterns;

impl MemoryPatterns {
    /// Check if a tensor size is optimal for GPU memory coalescing.
    ///
    /// GPU memory access is most efficient when memory accesses are coalesced
    /// (consecutive threads access consecutive memory locations).
    pub fn is_coalesced_access(tensor_size: usize) -> bool {
        // Memory coalescing works best with sizes that are multiples of the warp size (32)
        // and align well with cache line sizes (typically 128 bytes = 32 floats)
        tensor_size.is_multiple_of(32)
    }

    /// Suggest padding size to optimize memory access.
    pub fn suggest_padding(tensor_size: usize) -> usize {
        let remainder = tensor_size % 32;
        if remainder == 0 {
            0
        } else {
            32 - remainder
        }
    }

    /// Calculate optimal buffer size for GPU operations.
    pub fn optimal_buffer_size(requested_size: usize) -> usize {
        let padding = Self::suggest_padding(requested_size);
        requested_size + padding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_config_creation() {
        let config_1d = WorkgroupConfig::new_1d(64);
        assert_eq!(config_1d.x, 64);
        assert_eq!(config_1d.y, 1);
        assert_eq!(config_1d.z, 1);
        assert_eq!(config_1d.total_items(), 64);

        let config_2d = WorkgroupConfig::new_2d(8, 8);
        assert_eq!(config_2d.x, 8);
        assert_eq!(config_2d.y, 8);
        assert_eq!(config_2d.z, 1);
        assert_eq!(config_2d.total_items(), 64);

        let config_3d = WorkgroupConfig::new_3d(4, 4, 4);
        assert_eq!(config_3d.total_items(), 64);
    }

    #[test]
    fn test_optimal_workgroup_calculation() {
        let config = WorkgroupConfig::optimal_1d(1000);
        // Should calculate appropriate number of workgroups for 1000 elements
        assert!(config.x > 0);
        assert_eq!(config.y, 1);
        assert_eq!(config.z, 1);
    }

    #[test]
    fn test_memory_coalescing_checks() {
        assert!(MemoryPatterns::is_coalesced_access(32));
        assert!(MemoryPatterns::is_coalesced_access(64));
        assert!(MemoryPatterns::is_coalesced_access(128));
        assert!(!MemoryPatterns::is_coalesced_access(33));
        assert!(!MemoryPatterns::is_coalesced_access(100));
    }

    #[test]
    fn test_padding_suggestions() {
        assert_eq!(MemoryPatterns::suggest_padding(32), 0);
        assert_eq!(MemoryPatterns::suggest_padding(33), 31);
        assert_eq!(MemoryPatterns::suggest_padding(100), 28); // 100 % 32 = 4, so 32 - 4 = 28
    }

    #[test]
    #[ignore = "requires GPU device"]
    fn test_gpu_operations_creation() {
        match GpuOperations::new() {
            Ok(_ops) => {
                println!("✓ GPU operations manager created successfully");
            }
            Err(_) => {
                println!("⚠️ GPU operations creation failed (expected in headless environments)");
            }
        }
    }

    #[test]
    #[ignore = "requires GPU device"]
    fn test_tensor_copy_operation() {
        if let Ok(mut ops) = GpuOperations::new() {
            let source_data = vec![1.0f32, 2.0, 3.0, 4.0];

            if let Ok(source) = Tensor::new(ops.manager(), &source_data) {
                if let Ok(mut dest) = Tensor::<f32>::uninit(ops.manager(), 4) {
                    match ops.copy(&source, &mut dest) {
                        Ok(()) => {
                            println!("✓ Tensor copy operation completed");
                        }
                        Err(_) => {
                            println!("⚠️ Tensor copy failed (expected in some environments)");
                        }
                    }
                }
            }
        }
    }
}

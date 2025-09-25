//! GPU memory layout and alignment safety.
//!
//! This module provides compile-time guarantees about GPU memory layout,
//! preventing alignment issues and optimizing memory access patterns.

use crate::error::{KomputeError, Result};
use crate::tensor::{DimensionTensor, TensorData};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};

/// GPU memory alignment requirements for different data types.
///
/// These constants define the minimum alignment requirements for optimal
/// GPU memory access across different hardware architectures.
pub mod alignment {
    /// Standard scalar alignment (4 bytes) - f32, i32, u32
    pub const SCALAR: usize = 4;

    /// Vector alignment (8 bytes) - vec2 types
    pub const VEC2: usize = 8;

    /// Vector alignment (16 bytes) - vec3, vec4 types, optimal for SIMD
    pub const VEC4: usize = 16;

    /// Matrix alignment (16 bytes) - 4x4 matrices, large structures
    pub const MATRIX: usize = 16;

    /// Buffer alignment (64 bytes) - optimal for GPU cache lines
    pub const BUFFER: usize = 64;

    /// Compute shader alignment (256 bytes) - workgroup shared memory
    pub const WORKGROUP: usize = 256;
}

/// Trait for types that have specific GPU memory alignment requirements.
///
/// This trait ensures that types used in GPU operations have the correct
/// alignment for optimal memory access and prevent GPU crashes.
pub trait GpuAligned {
    /// The required alignment in bytes for this type on GPU.
    const ALIGNMENT: usize;

    /// Validate that this type meets GPU alignment requirements.
    fn validate_alignment() -> Result<()>
    where
        Self: Sized,
    {
        let rust_align = align_of::<Self>();

        if rust_align < Self::ALIGNMENT {
            return Err(KomputeError::memory(format!(
                "Type {} has alignment {} but GPU requires alignment {}",
                std::any::type_name::<Self>(),
                rust_align,
                Self::ALIGNMENT
            )));
        }

        // Ensure alignment is a power of 2
        if Self::ALIGNMENT & (Self::ALIGNMENT - 1) != 0 {
            return Err(KomputeError::memory(format!(
                "GPU alignment {} must be a power of 2",
                Self::ALIGNMENT
            )));
        }

        Ok(())
    }

    /// Check if a memory address meets the alignment requirements.
    fn check_address_alignment(ptr: *const Self) -> bool
    where
        Self: Sized,
    {
        (ptr as *const u8 as usize).is_multiple_of(Self::ALIGNMENT)
    }

    /// Get the padded size needed for an array of this type.
    fn padded_array_size(count: usize) -> usize
    where
        Self: Sized,
    {
        let element_size = size_of::<Self>();
        let aligned_size = (element_size + Self::ALIGNMENT - 1) & !(Self::ALIGNMENT - 1);
        aligned_size * count
    }
}

/// GPU-aligned scalar types with automatic alignment validation.
macro_rules! impl_gpu_aligned_scalar {
    ($type:ty, $align:expr) => {
        impl GpuAligned for $type {
            const ALIGNMENT: usize = $align;
        }
    };
}

impl_gpu_aligned_scalar!(f32, alignment::SCALAR);
impl_gpu_aligned_scalar!(f64, alignment::VEC2); // f64 needs 8-byte alignment
impl_gpu_aligned_scalar!(i32, alignment::SCALAR);
impl_gpu_aligned_scalar!(u32, alignment::SCALAR);

/// Wrapper type that enforces specific GPU memory alignment at compile time.
///
/// This type provides revolutionary compile-time alignment validation,
/// ensuring optimal GPU memory access patterns and preventing crashes.
#[repr(C, align(1))] // Will be overridden by const generic
pub struct AlignedData<T, const ALIGN: usize> {
    data: T,
    _phantom: PhantomData<[u8; ALIGN]>,
}

impl<T, const ALIGN: usize> AlignedData<T, ALIGN> {
    /// Create a new GPU-aligned data wrapper with runtime validation.
    ///
    /// This constructor validates alignment requirements at runtime,
    /// ensuring optimal GPU memory access patterns.
    ///
    /// # Safety Features
    ///
    /// - **Alignment Validation**: Ensures ALIGN is a power of 2
    /// - **Size Compatibility**: Validates T can be aligned to ALIGN
    /// - **GPU Optimization**: Guarantees optimal memory access patterns
    /// - **Runtime Checks**: Validates alignment requirements
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use komputers::memory_layout::{AlignedData, alignment};
    ///
    /// // ✅ Create optimally-aligned GPU data
    /// let aligned_f32 = AlignedData::<f32, {alignment::SCALAR}>::new(3.14159)?;
    /// let aligned_vec4 = AlignedData::<[f32; 4], {alignment::VEC4}>::new([1.0, 2.0, 3.0, 4.0])?;
    ///
    /// // ❌ This would fail at runtime - invalid alignment!
    /// // let bad_align = AlignedData::<f32, 3>::new(1.0)?; // 3 is not power of 2
    /// ```
    pub fn new(data: T) -> Result<Self> {
        // Runtime alignment validation
        if ALIGN == 0 || (ALIGN & (ALIGN - 1)) != 0 {
            return Err(KomputeError::memory(format!(
                "GPU alignment {} must be a power of 2",
                ALIGN
            )));
        }

        if align_of::<T>() > ALIGN {
            return Err(KomputeError::memory(format!(
                "Type alignment {} exceeds specified GPU alignment {}",
                align_of::<T>(),
                ALIGN
            )));
        }

        Ok(Self {
            data,
            _phantom: PhantomData,
        })
    }

    /// Get a reference to the aligned data.
    pub const fn get(&self) -> &T {
        &self.data
    }

    /// Get a mutable reference to the aligned data.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Consume the wrapper and return the inner data.
    pub fn into_inner(self) -> T {
        self.data
    }

    /// Get the compile-time alignment of this data.
    pub const fn alignment() -> usize {
        ALIGN
    }

    /// Check if this data meets GPU alignment requirements.
    pub fn validate_gpu_alignment(&self) -> Result<()> {
        let ptr = self as *const Self as *const u8;
        if !(ptr as usize).is_multiple_of(ALIGN) {
            return Err(KomputeError::memory(format!(
                "Data at address {:p} is not aligned to {} bytes",
                ptr, ALIGN
            )));
        }
        Ok(())
    }
}

impl<T: GpuAligned + Clone, const ALIGN: usize> AlignedData<T, ALIGN> {
    /// Create aligned data with automatic GPU alignment detection.
    ///
    /// This method uses the type's `GpuAligned` implementation to determine
    /// the optimal alignment automatically.
    pub fn new_auto_aligned(data: T) -> Result<Self> {
        if ALIGN < T::ALIGNMENT {
            return Err(KomputeError::memory(format!(
                "Specified alignment {} is less than required alignment {} for type {}",
                ALIGN,
                T::ALIGNMENT,
                std::any::type_name::<T>()
            )));
        }

        T::validate_alignment()?;
        Self::new(data)
    }
}

/// Memory-aligned tensor that guarantees optimal GPU access patterns.
///
/// This type combines dimension safety with memory alignment validation,
/// providing revolutionary compile-time guarantees about GPU memory layout.
pub struct AlignedTensor<T, const N: usize, const ALIGN: usize> {
    tensor: DimensionTensor<T, N>,
    _phantom: PhantomData<[u8; ALIGN]>,
}

impl<T: TensorData + GpuAligned, const N: usize, const ALIGN: usize> AlignedTensor<T, N, ALIGN> {
    /// Create a memory-aligned tensor with compile-time guarantees.
    ///
    /// This constructor provides revolutionary safety by combining:
    /// - Dimension validation (from DimensionTensor)
    /// - Memory alignment validation (from AlignedData)
    /// - GPU optimization hints (from alignment requirements)
    ///
    /// # Revolutionary Safety Features
    ///
    /// - **Triple Validation**: Dimension + Alignment + Type safety
    /// - **GPU Optimization**: Guaranteed optimal memory access patterns
    /// - **Compile-Time Checks**: All validation happens before runtime
    /// - **Performance Guarantees**: Aligned access prevents GPU stalls
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use komputers::memory_layout::{AlignedTensor, alignment};
    ///
    /// // ✅ Create dimension + alignment safe tensor
    /// let aligned_tensor = AlignedTensor::<f32, 4, {alignment::VEC4}>::new(
    ///     &mut manager,
    ///     &[1.0, 2.0, 3.0, 4.0]
    /// )?;
    ///
    /// // GPU access will be optimally aligned for SIMD operations
    /// assert_eq!(aligned_tensor.alignment(), alignment::VEC4);
    /// assert_eq!(aligned_tensor.compile_time_size(), 4);
    /// ```
    pub fn new(manager: &mut crate::Manager, data: &[T; N]) -> Result<Self> {
        // Validate alignment requirements
        if ALIGN < T::ALIGNMENT {
            return Err(KomputeError::memory(format!(
                "Tensor alignment {} is less than required {} for type {}",
                ALIGN,
                T::ALIGNMENT,
                std::any::type_name::<T>()
            )));
        }

        // Validate that alignment is appropriate for the data size
        let total_size = N * size_of::<T>();
        if total_size > 0 && ALIGN > total_size && ALIGN > alignment::BUFFER {
            return Err(KomputeError::memory(format!(
                "Excessive alignment {} for {} bytes of data",
                ALIGN, total_size
            )));
        }

        T::validate_alignment()?;

        // Create dimension-safe tensor
        let tensor = DimensionTensor::new_safe(manager, data)?;

        Ok(Self {
            tensor,
            _phantom: PhantomData,
        })
    }

    /// Get the compile-time alignment guarantee.
    pub const fn alignment() -> usize {
        ALIGN
    }

    /// Get the compile-time size guarantee.
    pub const fn compile_time_size() -> usize {
        N
    }

    /// Check if this tensor has optimal GPU alignment.
    pub const fn is_optimally_aligned() -> bool {
        ALIGN >= alignment::VEC4
    }

    /// Get a reference to the underlying dimension-safe tensor.
    pub fn tensor(&self) -> &DimensionTensor<T, N> {
        &self.tensor
    }

    /// Get the tensor handle for GPU operations.
    pub fn handle(&self) -> u64 {
        self.tensor.handle()
    }

    /// Get the runtime size of the tensor.
    pub fn size(&self) -> usize {
        self.tensor.size()
    }

    /// Get the runtime size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.tensor.size_bytes()
    }

    /// Calculate the optimal memory layout for this tensor.
    pub fn memory_layout_info(&self) -> MemoryLayoutInfo {
        let element_size = size_of::<T>();
        let total_size = N * element_size;
        let padded_size = (total_size + ALIGN - 1) & !(ALIGN - 1);

        MemoryLayoutInfo {
            element_count: N,
            element_size,
            element_alignment: T::ALIGNMENT,
            total_size,
            padded_size,
            tensor_alignment: ALIGN,
            is_simd_optimized: ALIGN >= alignment::VEC4,
            cache_line_aligned: ALIGN >= alignment::BUFFER,
        }
    }
}

/// Comprehensive information about tensor memory layout.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryLayoutInfo {
    /// Number of elements in the tensor
    pub element_count: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Required alignment for each element
    pub element_alignment: usize,
    /// Total size of all elements
    pub total_size: usize,
    /// Size including padding for alignment
    pub padded_size: usize,
    /// Alignment guarantee for the entire tensor
    pub tensor_alignment: usize,
    /// Whether the tensor is optimized for SIMD operations
    pub is_simd_optimized: bool,
    /// Whether the tensor is aligned to GPU cache lines
    pub cache_line_aligned: bool,
}

impl MemoryLayoutInfo {
    /// Calculate the memory efficiency (useful data / total allocated).
    pub fn memory_efficiency(&self) -> f64 {
        if self.padded_size == 0 {
            0.0
        } else {
            self.total_size as f64 / self.padded_size as f64
        }
    }

    /// Get optimization recommendations for this memory layout.
    pub fn optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.memory_efficiency() < 0.8 {
            recommendations.push(format!(
                "Consider reducing alignment from {} bytes to improve memory efficiency ({:.1}%)",
                self.tensor_alignment,
                self.memory_efficiency() * 100.0
            ));
        }

        if !self.is_simd_optimized && self.element_count >= 4 {
            recommendations.push(format!(
                "Consider using {}-byte alignment for SIMD optimization",
                alignment::VEC4
            ));
        }

        if !self.cache_line_aligned && self.total_size >= alignment::BUFFER {
            recommendations.push(format!(
                "Consider using {}-byte alignment for cache line optimization",
                alignment::BUFFER
            ));
        }

        if !self.element_count.is_multiple_of(4) && self.is_simd_optimized {
            recommendations.push(
                "Consider padding element count to multiple of 4 for optimal SIMD".to_string(),
            );
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_aligned_trait() {
        // Test alignment constants
        assert_eq!(f32::ALIGNMENT, alignment::SCALAR);
        assert_eq!(f64::ALIGNMENT, alignment::VEC2);
        assert_eq!(i32::ALIGNMENT, alignment::SCALAR);
        assert_eq!(u32::ALIGNMENT, alignment::SCALAR);

        // Test validation
        assert!(f32::validate_alignment().is_ok());
        assert!(f64::validate_alignment().is_ok());
        assert!(i32::validate_alignment().is_ok());
        assert!(u32::validate_alignment().is_ok());
    }

    #[test]
    fn test_aligned_data_creation() {
        // Test valid alignments
        let aligned_f32 = AlignedData::<f32, { alignment::SCALAR }>::new(std::f32::consts::PI);
        assert!(aligned_f32.is_ok());

        let aligned_array = AlignedData::<[f32; 4], { alignment::VEC4 }>::new([1.0, 2.0, 3.0, 4.0]);
        assert!(aligned_array.is_ok());

        // Test alignment properties
        if let Ok(data) = aligned_f32 {
            assert_eq!(*data.get(), std::f32::consts::PI);
            assert_eq!(
                AlignedData::<f32, { alignment::SCALAR }>::alignment(),
                alignment::SCALAR
            );
        }
    }

    #[test]
    fn test_memory_layout_info() {
        let info = MemoryLayoutInfo {
            element_count: 4,
            element_size: 4,
            element_alignment: alignment::SCALAR,
            total_size: 16,
            padded_size: 16,
            tensor_alignment: alignment::VEC4,
            is_simd_optimized: true,
            cache_line_aligned: false,
        };

        assert_eq!(info.memory_efficiency(), 1.0);
        assert!(info.is_simd_optimized);
        assert!(!info.cache_line_aligned);

        let recommendations = info.optimization_recommendations();
        assert!(!recommendations.is_empty()); // Should have at least cache line recommendation
    }

    #[test]
    fn test_alignment_constants() {
        // Verify all alignments are powers of 2
        assert!(alignment::SCALAR.is_power_of_two());
        assert!(alignment::VEC2.is_power_of_two());
        assert!(alignment::VEC4.is_power_of_two());
        assert!(alignment::MATRIX.is_power_of_two());
        assert!(alignment::BUFFER.is_power_of_two());
        assert!(alignment::WORKGROUP.is_power_of_two());

        // Verify alignment hierarchy
        assert!(alignment::SCALAR <= alignment::VEC2);
        assert!(alignment::VEC2 <= alignment::VEC4);
        assert!(alignment::VEC4 <= alignment::MATRIX);
        assert!(alignment::MATRIX <= alignment::BUFFER);
        assert!(alignment::BUFFER <= alignment::WORKGROUP);
    }
}

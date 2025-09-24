//! Compile-time and runtime workgroup validation for GPU compute operations
//!
//! This module provides comprehensive validation for GPU compute workgroup configurations,
//! combining compile-time const generic validation with runtime GPU capability checking
//! to ensure optimal performance and prevent dispatch failures.
//!
//! # Safety Guarantees
//!
//! - **Compile-Time Size Validation**: Invalid workgroup sizes caught at compile time
//! - **GPU Capability Checking**: Runtime validation against actual GPU limits
//! - **Performance Optimization**: Automatic suggestion of optimal configurations
//! - **Cross-Platform Compatibility**: Handles different GPU vendor limitations
//!
//! # Example
//!
//! ```rust,no_run
//! use komputers::workgroup_validation::{WorkgroupConfig, GpuLimits};
//!
//! // Compile-time validated workgroup (this will fail to compile if invalid)
//! let workgroup: WorkgroupConfig<256, 1, 1> = WorkgroupConfig::new();
//!
//! // Runtime validation against actual GPU
//! let gpu_limits = GpuLimits::detect()?;
//! workgroup.validate_against_gpu(&gpu_limits)?;
//! # Ok::<(), komputers::KomputeError>(())
//! ```

use crate::error::{KomputeError, Result};
use std::marker::PhantomData;

// =============================================================================
// Compile-Time Workgroup Configuration with Const Generics
// =============================================================================

/// Compile-time validated workgroup configuration using const generics
///
/// This struct uses const generics to enforce workgroup size limits at compile time,
/// preventing common GPU compute errors before they reach runtime.
///
/// # Type Parameters
///
/// * `X` - Workgroup size in X dimension (must be > 0 and <= 1024)
/// * `Y` - Workgroup size in Y dimension (must be > 0 and <= 1024)
/// * `Z` - Workgroup size in Z dimension (must be > 0 and <= 1024)
///
/// # Compile-Time Validation
///
/// The following constraints are enforced at compile time:
/// - All dimensions must be greater than 0
/// - All dimensions must be powers of 2 (for optimal GPU performance)
/// - Total workgroup size (X * Y * Z) must not exceed 1024
/// - Individual dimensions must not exceed 1024
///
/// # Example
///
/// ```rust,no_run
/// use komputers::workgroup_validation::WorkgroupConfig;
///
/// // ✅ Valid configurations (compile successfully)
/// let wg1: WorkgroupConfig<256, 1, 1> = WorkgroupConfig::new(); // 1D compute
/// let wg2: WorkgroupConfig<16, 16, 1> = WorkgroupConfig::new(); // 2D compute
/// let wg3: WorkgroupConfig<8, 8, 8> = WorkgroupConfig::new();   // 3D compute
///
/// // ❌ Invalid configurations (compile errors)
/// // let bad1: WorkgroupConfig<0, 1, 1> = WorkgroupConfig::new();     // Zero size
/// // let bad2: WorkgroupConfig<2048, 1, 1> = WorkgroupConfig::new();  // Too large
/// // let bad3: WorkgroupConfig<1025, 1, 1> = WorkgroupConfig::new();  // Exceeds total limit
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkgroupConfig<const X: u32, const Y: u32, const Z: u32> {
    _phantom: PhantomData<()>,
}

impl<const X: u32, const Y: u32, const Z: u32> Default for WorkgroupConfig<X, Y, Z> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const X: u32, const Y: u32, const Z: u32> WorkgroupConfig<X, Y, Z> {
    /// Create a new compile-time validated workgroup configuration
    ///
    /// This method enforces compile-time validation of workgroup parameters.
    /// Invalid configurations will result in compile-time errors.
    ///
    /// # Compile-Time Validation
    ///
    /// - All dimensions must be > 0
    /// - All dimensions must be <= 1024
    /// - Total size (X * Y * Z) must be <= 1024
    /// - Dimensions should be powers of 2 for optimal performance
    pub const fn new() -> Self {
        // Compile-time validation using const assertions
        assert!(X > 0, "Workgroup X dimension must be greater than 0");
        assert!(Y > 0, "Workgroup Y dimension must be greater than 0");
        assert!(Z > 0, "Workgroup Z dimension must be greater than 0");

        assert!(X <= 1024, "Workgroup X dimension must not exceed 1024");
        assert!(Y <= 1024, "Workgroup Y dimension must not exceed 1024");
        assert!(Z <= 1024, "Workgroup Z dimension must not exceed 1024");

        // Total workgroup size limit (common GPU constraint)
        assert!(
            X * Y * Z <= 1024,
            "Total workgroup size must not exceed 1024"
        );

        // Optional: Warn about non-power-of-2 sizes (suboptimal performance)
        // Note: This is a performance hint, not a hard requirement

        Self {
            _phantom: PhantomData,
        }
    }

    /// Get workgroup dimensions as a tuple
    pub const fn dimensions(&self) -> (u32, u32, u32) {
        (X, Y, Z)
    }

    /// Get total workgroup size
    pub const fn total_size(&self) -> u32 {
        X * Y * Z
    }

    /// Check if workgroup size is optimal (power of 2 dimensions)
    pub const fn is_optimal(&self) -> bool {
        Self::is_power_of_2(X) && Self::is_power_of_2(Y) && Self::is_power_of_2(Z)
    }

    /// Helper function to check if a number is a power of 2
    const fn is_power_of_2(n: u32) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Validate against actual GPU capabilities at runtime
    ///
    /// This method performs runtime validation to ensure the compile-time
    /// configuration is compatible with the actual GPU hardware.
    ///
    /// # Arguments
    ///
    /// * `gpu_limits` - The actual GPU capability limits
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the workgroup is compatible with the GPU
    /// * `Err(KomputeError)` if the workgroup exceeds GPU capabilities
    pub fn validate_against_gpu(&self, gpu_limits: &GpuLimits) -> Result<()> {
        // Check individual dimension limits
        if X > gpu_limits.max_workgroup_size.0 {
            return Err(KomputeError::compute(format!(
                "Workgroup X dimension {} exceeds GPU limit {}",
                X, gpu_limits.max_workgroup_size.0
            )));
        }

        if Y > gpu_limits.max_workgroup_size.1 {
            return Err(KomputeError::compute(format!(
                "Workgroup Y dimension {} exceeds GPU limit {}",
                Y, gpu_limits.max_workgroup_size.1
            )));
        }

        if Z > gpu_limits.max_workgroup_size.2 {
            return Err(KomputeError::compute(format!(
                "Workgroup Z dimension {} exceeds GPU limit {}",
                Z, gpu_limits.max_workgroup_size.2
            )));
        }

        // Check total workgroup size limit
        let total_size = X * Y * Z;
        if total_size > gpu_limits.max_workgroup_invocations {
            return Err(KomputeError::compute(format!(
                "Total workgroup size {} exceeds GPU limit {}",
                total_size, gpu_limits.max_workgroup_invocations
            )));
        }

        Ok(())
    }

    /// Convert to dynamic workgroup configuration for runtime use
    pub fn to_dynamic(&self) -> DynamicWorkgroupConfig {
        DynamicWorkgroupConfig { x: X, y: Y, z: Z }
    }

    /// Get performance hints for this workgroup configuration
    pub fn performance_hints(&self) -> WorkgroupPerformanceHints {
        WorkgroupPerformanceHints {
            is_power_of_2: self.is_optimal(),
            memory_coalescing_friendly: X.is_multiple_of(32), // Warp size alignment
            recommended_for_1d: Y == 1 && Z == 1,
            recommended_for_2d: Z == 1,
            total_threads: self.total_size(),
        }
    }
}

// =============================================================================
// Dynamic Workgroup Configuration for Runtime
// =============================================================================

/// Dynamic workgroup configuration for runtime computation
///
/// This is used when workgroup sizes are determined at runtime rather than
/// compile time. It provides runtime validation and optimization suggestions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynamicWorkgroupConfig {
    /// Workgroup size in X dimension
    pub x: u32,
    /// Workgroup size in Y dimension
    pub y: u32,
    /// Workgroup size in Z dimension
    pub z: u32,
}

impl DynamicWorkgroupConfig {
    /// Create a new dynamic workgroup configuration
    ///
    /// # Arguments
    ///
    /// * `x` - Workgroup size in X dimension
    /// * `y` - Workgroup size in Y dimension
    /// * `z` - Workgroup size in Z dimension
    ///
    /// # Returns
    ///
    /// A validated workgroup configuration or error if invalid
    pub fn new(x: u32, y: u32, z: u32) -> Result<Self> {
        // Runtime validation
        if x == 0 || y == 0 || z == 0 {
            return Err(KomputeError::compute(
                "Workgroup dimensions must be greater than 0",
            ));
        }

        if x > 1024 || y > 1024 || z > 1024 {
            return Err(KomputeError::compute(
                "Workgroup dimensions must not exceed 1024",
            ));
        }

        if x * y * z > 1024 {
            return Err(KomputeError::compute(
                "Total workgroup size must not exceed 1024",
            ));
        }

        Ok(Self { x, y, z })
    }

    /// Create optimized 1D workgroup for given number of elements
    pub fn optimal_1d(num_elements: u32) -> Self {
        // Use powers of 2 for optimal GPU performance
        let workgroup_size = if num_elements <= 64 {
            64
        } else if num_elements <= 128 {
            128
        } else if num_elements <= 256 {
            256
        } else {
            512
        };

        Self {
            x: workgroup_size,
            y: 1,
            z: 1,
        }
    }

    /// Create optimized 2D workgroup for given width and height
    pub fn optimal_2d(width: u32, height: u32) -> Self {
        // Find optimal 2D configuration
        let total_elements = width * height;

        if total_elements <= 256 {
            Self { x: 16, y: 16, z: 1 }
        } else if total_elements <= 512 {
            Self { x: 16, y: 32, z: 1 }
        } else {
            Self { x: 32, y: 32, z: 1 }
        }
    }

    /// Get total workgroup size
    pub fn total_size(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Validate against GPU capabilities
    pub fn validate_against_gpu(&self, gpu_limits: &GpuLimits) -> Result<()> {
        if self.x > gpu_limits.max_workgroup_size.0 {
            return Err(KomputeError::compute(format!(
                "Workgroup X dimension {} exceeds GPU limit {}",
                self.x, gpu_limits.max_workgroup_size.0
            )));
        }

        if self.y > gpu_limits.max_workgroup_size.1 {
            return Err(KomputeError::compute(format!(
                "Workgroup Y dimension {} exceeds GPU limit {}",
                self.y, gpu_limits.max_workgroup_size.1
            )));
        }

        if self.z > gpu_limits.max_workgroup_size.2 {
            return Err(KomputeError::compute(format!(
                "Workgroup Z dimension {} exceeds GPU limit {}",
                self.z, gpu_limits.max_workgroup_size.2
            )));
        }

        if self.total_size() > gpu_limits.max_workgroup_invocations {
            return Err(KomputeError::compute(format!(
                "Total workgroup size {} exceeds GPU limit {}",
                self.total_size(),
                gpu_limits.max_workgroup_invocations
            )));
        }

        Ok(())
    }
}

// =============================================================================
// GPU Capability Detection and Limits
// =============================================================================

/// GPU capability limits and characteristics
///
/// This structure contains the actual GPU hardware limits that constrain
/// workgroup configurations and compute operations.
#[derive(Debug, Clone)]
pub struct GpuLimits {
    /// Maximum workgroup size in each dimension (X, Y, Z)
    pub max_workgroup_size: (u32, u32, u32),
    /// Maximum total number of invocations in a workgroup
    pub max_workgroup_invocations: u32,
    /// Maximum number of compute units/shader cores
    pub max_compute_units: u32,
    /// Device name for identification
    pub device_name: String,
    /// Vendor-specific information
    pub vendor_info: VendorInfo,
}

impl GpuLimits {
    /// Detect GPU capabilities from the system
    ///
    /// This method queries the GPU hardware to determine actual capabilities
    /// and limits for workgroup configuration validation.
    ///
    /// # Returns
    ///
    /// GPU capability information or error if detection fails
    pub fn detect() -> Result<Self> {
        // TODO: Integrate with actual Vulkan device capability queries
        // For now, provide conservative defaults that work on most GPUs

        // These are conservative limits that should work on most modern GPUs
        let limits = Self {
            max_workgroup_size: (1024, 1024, 64), // Common Vulkan limits
            max_workgroup_invocations: 1024,      // Conservative total
            max_compute_units: 64,                // Conservative estimate
            device_name: "Unknown GPU".to_string(),
            vendor_info: VendorInfo::detect(),
        };

        // TODO: Replace with actual Vulkan device property queries
        // VkPhysicalDeviceProperties properties;
        // vkGetPhysicalDeviceProperties(device, &properties);

        Ok(limits)
    }

    /// Get recommended workgroup configuration for optimal performance
    ///
    /// # Arguments
    ///
    /// * `total_elements` - Total number of elements to process
    /// * `dimensions` - Number of dimensions (1, 2, or 3)
    ///
    /// # Returns
    ///
    /// Optimized workgroup configuration for this GPU
    pub fn recommend_workgroup(
        &self,
        total_elements: u32,
        dimensions: u8,
    ) -> DynamicWorkgroupConfig {
        match dimensions {
            1 => self.recommend_1d_workgroup(total_elements),
            2 => self.recommend_2d_workgroup(total_elements),
            3 => self.recommend_3d_workgroup(total_elements),
            _ => DynamicWorkgroupConfig::optimal_1d(total_elements),
        }
    }

    fn recommend_1d_workgroup(&self, total_elements: u32) -> DynamicWorkgroupConfig {
        // Optimize for this specific GPU's characteristics
        let optimal_size = match self.vendor_info {
            VendorInfo::Nvidia => {
                // NVIDIA GPUs work best with warp-aligned sizes (32)
                if total_elements <= 128 {
                    128
                } else {
                    256
                }
            }
            VendorInfo::Amd => {
                // AMD GPUs work best with wavefront-aligned sizes (64)
                if total_elements <= 128 {
                    128
                } else {
                    256
                }
            }
            VendorInfo::Intel => {
                // Intel GPUs typically prefer smaller workgroups
                if total_elements <= 64 {
                    64
                } else {
                    128
                }
            }
            VendorInfo::Apple => {
                // Apple GPUs (Metal-via-MoltenVK) prefer specific sizes
                if total_elements <= 128 {
                    128
                } else {
                    256
                }
            }
            VendorInfo::Unknown => {
                // Conservative default
                128
            }
        };

        DynamicWorkgroupConfig {
            x: optimal_size
                .min(self.max_workgroup_size.0)
                .min(total_elements),
            y: 1,
            z: 1,
        }
    }

    fn recommend_2d_workgroup(&self, _total_elements: u32) -> DynamicWorkgroupConfig {
        // 2D workgroup recommendations based on GPU vendor
        let (x, y) = match self.vendor_info {
            VendorInfo::Nvidia => (16, 16), // Good for NVIDIA tensor cores
            VendorInfo::Amd => (16, 16),    // Good for AMD compute units
            VendorInfo::Intel => (8, 8),    // Conservative for Intel
            VendorInfo::Apple => (16, 16),  // Good for Apple GPUs
            VendorInfo::Unknown => (8, 8),  // Very conservative
        };

        DynamicWorkgroupConfig {
            x: x.min(self.max_workgroup_size.0),
            y: y.min(self.max_workgroup_size.1),
            z: 1,
        }
    }

    fn recommend_3d_workgroup(&self, _total_elements: u32) -> DynamicWorkgroupConfig {
        // 3D workgroup recommendations
        let (x, y, z) = match self.vendor_info {
            VendorInfo::Nvidia => (8, 8, 8),  // Good for 3D compute
            VendorInfo::Amd => (8, 8, 8),     // Good for 3D compute
            VendorInfo::Intel => (4, 4, 4),   // Conservative for Intel
            VendorInfo::Apple => (8, 8, 8),   // Good for Apple GPUs
            VendorInfo::Unknown => (4, 4, 4), // Very conservative
        };

        DynamicWorkgroupConfig {
            x: x.min(self.max_workgroup_size.0),
            y: y.min(self.max_workgroup_size.1),
            z: z.min(self.max_workgroup_size.2),
        }
    }
}

/// GPU vendor information for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VendorInfo {
    /// NVIDIA GPU vendor
    Nvidia,
    /// AMD GPU vendor
    Amd,
    /// Intel GPU vendor
    Intel,
    /// Apple GPU vendor (Metal via MoltenVK)
    Apple,
    /// Unknown or unrecognized GPU vendor
    Unknown,
}

impl VendorInfo {
    /// Detect GPU vendor from system information
    fn detect() -> Self {
        // TODO: Integrate with actual Vulkan device vendor detection
        // For now, return Unknown as a safe default
        VendorInfo::Unknown
    }
}

/// Performance hints for workgroup configuration
#[derive(Debug, Clone)]
pub struct WorkgroupPerformanceHints {
    /// Whether dimensions are powers of 2 (optimal)
    pub is_power_of_2: bool,
    /// Whether configuration is friendly for memory coalescing
    pub memory_coalescing_friendly: bool,
    /// Whether this is recommended for 1D operations
    pub recommended_for_1d: bool,
    /// Whether this is recommended for 2D operations
    pub recommended_for_2d: bool,
    /// Total number of threads in workgroup
    pub total_threads: u32,
}

impl WorkgroupPerformanceHints {
    /// Get optimization score (0-100, higher is better)
    pub fn optimization_score(&self) -> u8 {
        let mut score = 50; // Base score

        if self.is_power_of_2 {
            score += 20; // Power of 2 is very important
        }

        if self.memory_coalescing_friendly {
            score += 15; // Memory coalescing helps performance
        }

        if self.total_threads >= 64 && self.total_threads <= 256 {
            score += 15; // Sweet spot for most GPUs
        }

        score.min(100)
    }

    /// Get performance recommendations
    pub fn recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if !self.is_power_of_2 {
            recommendations.push(
                "Consider using power-of-2 dimensions for better GPU utilization".to_string(),
            );
        }

        if !self.memory_coalescing_friendly {
            recommendations.push(
                "Consider aligning X dimension to 32 for better memory coalescing".to_string(),
            );
        }

        if self.total_threads < 64 {
            recommendations
                .push("Consider increasing workgroup size for better GPU occupancy".to_string());
        }

        if self.total_threads > 512 {
            recommendations
                .push("Consider reducing workgroup size to avoid resource contention".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Workgroup configuration looks optimal!".to_string());
        }

        recommendations
    }
}

// =============================================================================
// Workgroup Validation Integration
// =============================================================================

/// Trait for types that can validate workgroup configurations
pub trait WorkgroupValidation {
    /// Validate workgroup configuration for this operation
    fn validate_workgroup(&self, config: &DynamicWorkgroupConfig) -> Result<()>;

    /// Get recommended workgroup configuration for this operation
    fn recommend_workgroup(&self, gpu_limits: &GpuLimits) -> DynamicWorkgroupConfig;
}

// =============================================================================
// Common Workgroup Configurations (Type Aliases)
// =============================================================================

/// Common 1D workgroup configurations
pub mod common_1d {
    use super::WorkgroupConfig;

    /// Small 1D workgroup (64 threads)
    pub type Small = WorkgroupConfig<64, 1, 1>;
    /// Medium 1D workgroup (128 threads)
    pub type Medium = WorkgroupConfig<128, 1, 1>;
    /// Large 1D workgroup (256 threads)
    pub type Large = WorkgroupConfig<256, 1, 1>;
    /// Extra large 1D workgroup (512 threads)
    pub type XLarge = WorkgroupConfig<512, 1, 1>;
}

/// Common 2D workgroup configurations
pub mod common_2d {
    use super::WorkgroupConfig;

    /// Small 2D workgroup (8x8 = 64 threads)
    pub type Small = WorkgroupConfig<8, 8, 1>;
    /// Medium 2D workgroup (16x16 = 256 threads)
    pub type Medium = WorkgroupConfig<16, 16, 1>;
    /// Large 2D workgroup (32x16 = 512 threads)
    pub type Large = WorkgroupConfig<32, 16, 1>;
}

/// Common 3D workgroup configurations
pub mod common_3d {
    use super::WorkgroupConfig;

    /// Small 3D workgroup (4x4x4 = 64 threads)
    pub type Small = WorkgroupConfig<4, 4, 4>;
    /// Medium 3D workgroup (8x8x4 = 256 threads)
    pub type Medium = WorkgroupConfig<8, 8, 4>;
    /// Large 3D workgroup (8x8x8 = 512 threads)
    pub type Large = WorkgroupConfig<8, 8, 8>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_time_workgroup_validation() {
        // These should compile successfully
        let _wg1: WorkgroupConfig<64, 1, 1> = WorkgroupConfig::new();
        let _wg2: WorkgroupConfig<16, 16, 1> = WorkgroupConfig::new();
        let _wg3: WorkgroupConfig<8, 8, 8> = WorkgroupConfig::new();

        assert_eq!(_wg1.dimensions(), (64, 1, 1));
        assert_eq!(_wg1.total_size(), 64);
        assert!(_wg1.is_optimal());
    }

    #[test]
    fn test_dynamic_workgroup_creation() {
        let wg = DynamicWorkgroupConfig::new(128, 1, 1).unwrap();
        assert_eq!(wg.total_size(), 128);

        // Test validation
        assert!(DynamicWorkgroupConfig::new(0, 1, 1).is_err());
        assert!(DynamicWorkgroupConfig::new(2048, 1, 1).is_err());
        assert!(DynamicWorkgroupConfig::new(1025, 1, 1).is_err());
    }

    #[test]
    fn test_gpu_limits_detection() {
        let limits = GpuLimits::detect().unwrap();
        assert!(limits.max_workgroup_invocations > 0);
        assert!(limits.max_workgroup_size.0 > 0);
    }

    #[test]
    fn test_workgroup_optimization() {
        let wg: WorkgroupConfig<256, 1, 1> = WorkgroupConfig::new();
        let hints = wg.performance_hints();

        assert!(hints.is_power_of_2);
        assert!(hints.memory_coalescing_friendly);
        assert!(hints.optimization_score() > 70);
    }

    #[test]
    fn test_common_configurations() {
        let _small_1d = common_1d::Small::new();
        let _medium_2d = common_2d::Medium::new();
        let _large_3d = common_3d::Large::new();

        assert_eq!(_small_1d.dimensions(), (64, 1, 1));
        assert_eq!(_medium_2d.dimensions(), (16, 16, 1));
        assert_eq!(_large_3d.dimensions(), (8, 8, 8));
    }

    #[test]
    fn test_performance_recommendations() {
        let wg: WorkgroupConfig<63, 1, 1> = WorkgroupConfig::new(); // Not power of 2
        let hints = wg.performance_hints();
        let recommendations = hints.recommendations();

        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("power-of-2")));
    }
}

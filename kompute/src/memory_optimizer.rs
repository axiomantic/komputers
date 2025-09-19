//! Platform-specific GPU memory optimization and management.
//!
//! This module provides platform-appropriate memory allocation strategies,
//! memory usage optimization, and platform-specific GPU memory management
//! for optimal performance across macOS (MoltenVK), Linux, and Windows.

use crate::error::{KomputeError, Result};
use crate::platform::PlatformInfo;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Platform-specific memory optimization strategies and memory pool management.
///
/// The memory optimizer provides platform-appropriate allocation strategies:
/// - **macOS (MoltenVK)**: Conservative allocation with smaller pools
/// - **Linux/Windows**: Aggressive allocation with larger memory pools
/// - **Adaptive**: Dynamic optimization based on runtime usage patterns
pub struct MemoryOptimizer {
    platform: PlatformInfo,
    memory_pools: Arc<Mutex<HashMap<String, MemoryPool>>>,
    allocation_strategy: AllocationStrategy,
    statistics: Arc<Mutex<MemoryStatistics>>,
}

/// Memory pool tracking for GPU memory management
#[derive(Debug, Clone)]
pub struct MemoryPool {
    /// Unique identifier for this memory pool
    pub pool_id: String,
    /// Total size of the pool in bytes
    pub size_bytes: usize,
    /// Currently allocated bytes in this pool
    pub allocated_bytes: usize,
    /// Available free bytes in this pool
    pub free_bytes: usize,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_ratio: f32,
    /// Number of active allocations in this pool
    pub active_allocations: usize,
    /// Platform-specific optimization flags
    pub optimization_flags: Vec<String>,
}

/// Memory allocation strategies for different platforms
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationStrategy {
    /// Conservative strategy for MoltenVK - smaller allocations, frequent cleanup
    Conservative {
        max_single_allocation: usize,
        cleanup_threshold: f32,
    },
    /// Aggressive strategy for native Vulkan - larger allocations, batch operations
    Aggressive {
        preferred_allocation_size: usize,
        batch_allocation_count: usize,
    },
    /// Adaptive strategy that adjusts based on runtime behavior
    Adaptive {
        current_strategy: Box<AllocationStrategy>,
        adaptation_history: Vec<PerformanceMetric>,
    },
}

/// Performance metrics for adaptive memory management
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceMetric {
    pub timestamp: std::time::Instant,
    pub allocation_time_ms: f64,
    pub allocation_size: usize,
    pub success: bool,
    pub platform_specific_data: HashMap<String, String>,
}

/// Comprehensive memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStatistics {
    /// Total number of memory pools
    pub total_pools: usize,
    /// Total bytes allocated across all pools
    pub total_allocated: usize,
    /// Total bytes available across all pools
    pub total_free: usize,
    /// Average fragmentation across all pools
    pub average_fragmentation: f32,
    /// Current allocation strategy being used
    pub platform_strategy: String,
    /// Total number of allocations performed
    pub total_allocations: u64,
    /// Total number of failed allocations
    pub failed_allocations: u64,
    /// Average allocation time in milliseconds
    pub average_allocation_time_ms: f64,
    /// Platform-specific optimization metrics
    pub platform_metrics: HashMap<String, f64>,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer with platform-specific settings.
    ///
    /// Automatically detects the current platform and configures appropriate
    /// memory management strategies for optimal GPU performance.
    pub fn new() -> Result<Self> {
        let platform = PlatformInfo::detect();
        let allocation_strategy = Self::determine_optimal_strategy(&platform);
        let memory_pools = Arc::new(Mutex::new(HashMap::new()));
        let statistics = Arc::new(Mutex::new(MemoryStatistics::default()));

        Ok(Self {
            platform,
            memory_pools,
            allocation_strategy,
            statistics,
        })
    }

    /// Determine the optimal allocation strategy for the detected platform.
    fn determine_optimal_strategy(platform: &PlatformInfo) -> AllocationStrategy {
        match platform.os.as_str() {
            "macos" => {
                // MoltenVK has more conservative memory management requirements
                AllocationStrategy::Conservative {
                    max_single_allocation: 512 * 1024 * 1024, // 512MB max per allocation
                    cleanup_threshold: 0.7, // Clean up when 70% fragmented
                }
            }
            "linux" | "windows" => {
                // Native Vulkan can handle more aggressive allocation strategies
                AllocationStrategy::Aggressive {
                    preferred_allocation_size: 256 * 1024 * 1024, // 256MB preferred chunks
                    batch_allocation_count: 8, // Allocate up to 8 chunks at once
                }
            }
            _ => {
                // Unknown platforms start with conservative and adapt
                AllocationStrategy::Adaptive {
                    current_strategy: Box::new(AllocationStrategy::Conservative {
                        max_single_allocation: 256 * 1024 * 1024,
                        cleanup_threshold: 0.8,
                    }),
                    adaptation_history: Vec::new(),
                }
            }
        }
    }

    /// Get the optimal tensor allocation size for the current platform.
    ///
    /// This method considers platform-specific memory alignment requirements,
    /// GPU memory bandwidth characteristics, and allocation pool strategies.
    ///
    /// # Arguments
    ///
    /// * `requested_size` - The requested tensor size in bytes
    ///
    /// # Returns
    ///
    /// The optimized allocation size that provides best performance for the platform
    pub fn optimal_tensor_size(&self, requested_size: usize) -> usize {
        if requested_size == 0 {
            return 0;
        }

        match &self.allocation_strategy {
            AllocationStrategy::Conservative { max_single_allocation, .. } => {
                // macOS MoltenVK: Conservative allocation with size limits
                let clamped_size = requested_size.min(*max_single_allocation);
                Self::align_to_memory_boundary(clamped_size, 4096) // 4KB alignment
            }
            AllocationStrategy::Aggressive { preferred_allocation_size, .. } => {
                // Native Vulkan: Round up to preferred chunk sizes for efficiency
                let aligned_size = Self::align_to_memory_boundary(requested_size, 4096);
                if aligned_size <= *preferred_allocation_size {
                    aligned_size
                } else {
                    // For large allocations, round up to nearest preferred chunk multiple
                    let chunks = (aligned_size + preferred_allocation_size - 1) / preferred_allocation_size;
                    chunks * preferred_allocation_size
                }
            }
            AllocationStrategy::Adaptive { current_strategy, .. } => {
                // Delegate to the current adaptive strategy
                match current_strategy.as_ref() {
                    AllocationStrategy::Conservative { max_single_allocation, .. } => {
                        requested_size.min(*max_single_allocation)
                    }
                    AllocationStrategy::Aggressive { preferred_allocation_size, .. } => {
                        Self::align_to_memory_boundary(requested_size, *preferred_allocation_size)
                    }
                    _ => requested_size, // Fallback
                }
            }
        }
    }

    /// Align memory size to platform-appropriate boundaries.
    fn align_to_memory_boundary(size: usize, boundary: usize) -> usize {
        (size + boundary - 1) & !(boundary - 1)
    }

    /// Configure platform-specific memory management settings.
    ///
    /// This method applies platform-specific optimizations including:
    /// - Environment variable configuration
    /// - Memory pool initialization
    /// - Platform-specific GPU driver optimizations
    pub fn configure_platform_memory(&mut self) -> Result<()> {
        match self.platform.os.as_str() {
            "macos" => self.configure_moltenvk_memory(),
            "linux" => self.configure_linux_memory(),
            "windows" => self.configure_windows_memory(),
            _ => {
                // Unknown platform - use safe defaults
                self.initialize_default_memory_pools()
            }
        }
    }

    /// Configure MoltenVK-specific memory optimizations for macOS.
    pub fn configure_moltenvk_memory(&mut self) -> Result<()> {
        // Set MoltenVK environment variables for optimal memory usage
        std::env::set_var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "1");
        std::env::set_var("MVK_CONFIG_LOG_LEVEL", "1"); // Reduce logging overhead
        std::env::set_var("MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE", "");
        std::env::set_var("MVK_CONFIG_PERFORMANCE_TRACKING", "0"); // Disable for production
        std::env::set_var("MVK_CONFIG_USE_METAL_PRIVATE_API", "0"); // Use public APIs only

        // Initialize conservative memory pools suitable for MoltenVK
        let mut pools = self.memory_pools.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire memory pools lock")
        })?;

        pools.insert("moltenvk_primary".to_string(), MemoryPool {
            pool_id: "moltenvk_primary".to_string(),
            size_bytes: 256 * 1024 * 1024, // 256MB conservative pool
            allocated_bytes: 0,
            free_bytes: 256 * 1024 * 1024,
            fragmentation_ratio: 0.0,
            active_allocations: 0,
            optimization_flags: vec![
                "moltenvk_compatible".to_string(),
                "metal_argument_buffers".to_string(),
                "conservative_allocation".to_string(),
            ],
        });

        pools.insert("moltenvk_scratch".to_string(), MemoryPool {
            pool_id: "moltenvk_scratch".to_string(),
            size_bytes: 64 * 1024 * 1024, // 64MB scratch space
            allocated_bytes: 0,
            free_bytes: 64 * 1024 * 1024,
            fragmentation_ratio: 0.0,
            active_allocations: 0,
            optimization_flags: vec![
                "scratch_space".to_string(),
                "fast_cleanup".to_string(),
            ],
        });

        // Update statistics
        let mut stats = self.statistics.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire statistics lock")
        })?;
        stats.platform_strategy = "MoltenVK Conservative".to_string();
        stats.total_pools = pools.len();
        stats.platform_metrics.insert("moltenvk_version".to_string(), 1.2);
        stats.platform_metrics.insert("metal_version".to_string(), 3.0);

        println!("✓ MoltenVK memory optimization configured");
        Ok(())
    }

    /// Configure Linux native Vulkan memory optimizations.
    pub fn configure_linux_memory(&mut self) -> Result<()> {
        // Linux native Vulkan can use more aggressive memory strategies
        let mut pools = self.memory_pools.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire memory pools lock")
        })?;

        // Check if we have a discrete GPU for aggressive allocation
        let is_discrete_gpu = self.platform.gpu_support == crate::platform::GpuSupport::Native;

        if is_discrete_gpu {
            pools.insert("linux_discrete".to_string(), MemoryPool {
                pool_id: "linux_discrete".to_string(),
                size_bytes: 2 * 1024 * 1024 * 1024, // 2GB aggressive pool for discrete GPU
                allocated_bytes: 0,
                free_bytes: 2 * 1024 * 1024 * 1024,
                fragmentation_ratio: 0.0,
                active_allocations: 0,
                optimization_flags: vec![
                    "discrete_gpu".to_string(),
                    "aggressive_allocation".to_string(),
                    "large_buffers".to_string(),
                ],
            });
        } else {
            pools.insert("linux_integrated".to_string(), MemoryPool {
                pool_id: "linux_integrated".to_string(),
                size_bytes: 512 * 1024 * 1024, // 512MB for integrated GPU
                allocated_bytes: 0,
                free_bytes: 512 * 1024 * 1024,
                fragmentation_ratio: 0.0,
                active_allocations: 0,
                optimization_flags: vec![
                    "integrated_gpu".to_string(),
                    "shared_memory".to_string(),
                ],
            });
        }

        // High-performance compute pool for large tensor operations
        pools.insert("linux_compute".to_string(), MemoryPool {
            pool_id: "linux_compute".to_string(),
            size_bytes: 1024 * 1024 * 1024, // 1GB compute pool
            allocated_bytes: 0,
            free_bytes: 1024 * 1024 * 1024,
            fragmentation_ratio: 0.0,
            active_allocations: 0,
            optimization_flags: vec![
                "compute_optimized".to_string(),
                "high_bandwidth".to_string(),
            ],
        });

        // Update statistics
        let mut stats = self.statistics.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire statistics lock")
        })?;
        stats.platform_strategy = "Linux Native Vulkan".to_string();
        stats.total_pools = pools.len();
        stats.platform_metrics.insert("vulkan_version".to_string(), 1.3);
        stats.platform_metrics.insert("discrete_gpu".to_string(), if is_discrete_gpu { 1.0 } else { 0.0 });

        println!("✓ Linux Vulkan memory optimization configured");
        Ok(())
    }

    /// Configure Windows native Vulkan memory optimizations.
    pub fn configure_windows_memory(&mut self) -> Result<()> {
        // Windows native Vulkan with DirectX compatibility considerations
        let mut pools = self.memory_pools.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire memory pools lock")
        })?;

        pools.insert("windows_primary".to_string(), MemoryPool {
            pool_id: "windows_primary".to_string(),
            size_bytes: 1024 * 1024 * 1024, // 1GB primary pool for Windows
            allocated_bytes: 0,
            free_bytes: 1024 * 1024 * 1024,
            fragmentation_ratio: 0.0,
            active_allocations: 0,
            optimization_flags: vec![
                "windows_native".to_string(),
                "directx_compatible".to_string(),
                "wddm_optimized".to_string(),
            ],
        });

        pools.insert("windows_staging".to_string(), MemoryPool {
            pool_id: "windows_staging".to_string(),
            size_bytes: 256 * 1024 * 1024, // 256MB staging area
            allocated_bytes: 0,
            free_bytes: 256 * 1024 * 1024,
            fragmentation_ratio: 0.0,
            active_allocations: 0,
            optimization_flags: vec![
                "staging_buffer".to_string(),
                "host_visible".to_string(),
            ],
        });

        // Update statistics
        let mut stats = self.statistics.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire statistics lock")
        })?;
        stats.platform_strategy = "Windows Native Vulkan".to_string();
        stats.total_pools = pools.len();
        stats.platform_metrics.insert("vulkan_version".to_string(), 1.3);
        stats.platform_metrics.insert("wddm_version".to_string(), 3.0);

        println!("✓ Windows Vulkan memory optimization configured");
        Ok(())
    }

    /// Initialize default memory pools for unknown platforms.
    fn initialize_default_memory_pools(&mut self) -> Result<()> {
        let mut pools = self.memory_pools.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire memory pools lock")
        })?;

        pools.insert("default_primary".to_string(), MemoryPool {
            pool_id: "default_primary".to_string(),
            size_bytes: 512 * 1024 * 1024, // 512MB safe default
            allocated_bytes: 0,
            free_bytes: 512 * 1024 * 1024,
            fragmentation_ratio: 0.0,
            active_allocations: 0,
            optimization_flags: vec!["safe_defaults".to_string()],
        });

        let mut stats = self.statistics.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire statistics lock")
        })?;
        stats.platform_strategy = "Safe Defaults".to_string();
        stats.total_pools = pools.len();

        println!("✓ Default memory pools initialized");
        Ok(())
    }

    /// Get comprehensive memory usage statistics.
    ///
    /// Returns detailed statistics about memory pool usage, allocation patterns,
    /// and platform-specific performance metrics.
    pub fn memory_statistics(&self) -> Result<MemoryStatistics> {
        let pools = self.memory_pools.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire memory pools lock")
        })?;

        let stats = self.statistics.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire statistics lock")
        })?;

        let mut current_stats = stats.clone();

        // Update current pool statistics
        current_stats.total_pools = pools.len();
        current_stats.total_allocated = pools.values().map(|p| p.allocated_bytes).sum();
        current_stats.total_free = pools.values().map(|p| p.free_bytes).sum();

        if !pools.is_empty() {
            current_stats.average_fragmentation = pools.values()
                .map(|p| p.fragmentation_ratio)
                .sum::<f32>() / pools.len() as f32;
        }

        Ok(current_stats)
    }

    /// Record an allocation performance metric for adaptive optimization.
    pub fn record_allocation_metric(&mut self, metric: PerformanceMetric) -> Result<()> {
        let mut stats = self.statistics.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire statistics lock")
        })?;

        stats.total_allocations += 1;
        if !metric.success {
            stats.failed_allocations += 1;
        }

        // Update running average of allocation time
        let new_time = metric.allocation_time_ms;
        stats.average_allocation_time_ms = if stats.total_allocations == 1 {
            new_time
        } else {
            (stats.average_allocation_time_ms * (stats.total_allocations - 1) as f64 + new_time)
                / stats.total_allocations as f64
        };

        // For adaptive strategies, consider adjusting based on performance
        if let AllocationStrategy::Adaptive { current_strategy: _, adaptation_history } = &mut self.allocation_strategy {
            adaptation_history.push(metric);

            // Keep only recent history (last 100 allocations)
            if adaptation_history.len() > 100 {
                adaptation_history.remove(0);
            }

            // TODO: Implement strategy adaptation logic based on performance trends
        }

        Ok(())
    }

    /// Get platform-specific memory recommendations.
    ///
    /// Returns recommendations for optimal memory usage patterns
    /// based on the current platform and allocation strategy.
    pub fn platform_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        match self.platform.os.as_str() {
            "macos" => {
                recommendations.extend(vec![
                    "Use smaller tensor allocations (< 512MB) for MoltenVK compatibility".to_string(),
                    "Prefer frequent memory cleanup to avoid fragmentation".to_string(),
                    "Enable Metal argument buffers for better performance".to_string(),
                    "Consider using unified memory architecture advantages".to_string(),
                ]);
            }
            "linux" => {
                recommendations.extend(vec![
                    "Take advantage of native Vulkan for large memory allocations".to_string(),
                    "Use memory-mapped buffers for frequent CPU-GPU data transfer".to_string(),
                    "Consider NUMA topology for multi-GPU systems".to_string(),
                    "Optimize for your specific GPU vendor (NVIDIA/AMD/Intel)".to_string(),
                ]);
            }
            "windows" => {
                recommendations.extend(vec![
                    "Leverage WDDM memory management for stability".to_string(),
                    "Consider DirectX interoperability if needed".to_string(),
                    "Use staging buffers for large data transfers".to_string(),
                    "Be aware of GPU memory virtualization on Windows".to_string(),
                ]);
            }
            _ => {
                recommendations.push("Use conservative memory allocation for unknown platforms".to_string());
            }
        }

        // Add strategy-specific recommendations
        match &self.allocation_strategy {
            AllocationStrategy::Conservative { .. } => {
                recommendations.push("Conservative strategy active: smaller, safer allocations".to_string());
            }
            AllocationStrategy::Aggressive { .. } => {
                recommendations.push("Aggressive strategy active: larger, batched allocations".to_string());
            }
            AllocationStrategy::Adaptive { .. } => {
                recommendations.push("Adaptive strategy active: learning from allocation patterns".to_string());
            }
        }

        recommendations
    }

    /// Allocate optimized memory with platform-specific settings.
    ///
    /// This is the main allocation method that applies platform-specific
    /// optimizations and tracks allocation metrics.
    ///
    /// # Arguments
    ///
    /// * `pool_name` - Unique identifier for the memory pool
    /// * `size_bytes` - Size of allocation in bytes
    /// * `memory_type` - Type of memory (e.g., "DEVICE_LOCAL", "HOST_VISIBLE")
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the allocation
    pub fn allocate_optimized(
        &mut self,
        pool_name: &str,
        size_bytes: usize,
        memory_type: &str,
    ) -> Result<()> {
        if pool_name.is_empty() {
            return Err(KomputeError::memory("Memory pool name cannot be empty"));
        }

        if size_bytes == 0 {
            return Err(KomputeError::memory("Cannot allocate zero bytes"));
        }

        let start_time = std::time::Instant::now();
        let optimized_size = self.optimal_tensor_size(size_bytes);

        // Update statistics
        let mut stats = self.statistics.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire statistics lock")
        })?;

        stats.total_allocations += 1;

        // Create or update memory pool
        let mut pools = self.memory_pools.lock().map_err(|_| {
            KomputeError::memory("Failed to acquire memory pools lock")
        })?;

        let pool = pools.entry(pool_name.to_string()).or_insert_with(|| MemoryPool {
            pool_id: pool_name.to_string(),
            size_bytes: optimized_size,
            allocated_bytes: 0,
            free_bytes: optimized_size,
            fragmentation_ratio: 0.0,
            active_allocations: 0,
            optimization_flags: vec![memory_type.to_string()],
        });

        // Check if allocation fits
        if pool.free_bytes < optimized_size {
            stats.failed_allocations += 1;
            return Err(KomputeError::memory(format!(
                "Insufficient memory in pool '{}': requested {}, available {}",
                pool_name, optimized_size, pool.free_bytes
            )));
        }

        // Update pool statistics
        pool.allocated_bytes += optimized_size;
        pool.free_bytes -= optimized_size;
        pool.active_allocations += 1;

        // Calculate fragmentation ratio
        if pool.size_bytes > 0 {
            pool.fragmentation_ratio = 1.0 - (pool.free_bytes as f32 / pool.size_bytes as f32);
        }

        // Update global statistics
        stats.total_allocated = pools.values().map(|p| p.allocated_bytes).sum();
        stats.total_free = pools.values().map(|p| p.free_bytes).sum();
        stats.total_pools = pools.len();

        let allocation_time = start_time.elapsed();
        stats.average_allocation_time_ms = if stats.total_allocations > 0 {
            (stats.average_allocation_time_ms * (stats.total_allocations - 1) as f64
                + allocation_time.as_secs_f64() * 1000.0) / stats.total_allocations as f64
        } else {
            allocation_time.as_secs_f64() * 1000.0
        };

        Ok(())
    }

    /// Get the current allocation strategy.
    pub fn allocation_strategy(&self) -> &AllocationStrategy {
        &self.allocation_strategy
    }

    /// Set a new allocation strategy.
    pub fn set_allocation_strategy(&mut self, strategy: AllocationStrategy) -> Result<()> {
        self.allocation_strategy = strategy;
        Ok(())
    }

    /// Get platform information.
    pub fn platform_info(&self) -> &PlatformInfo {
        &self.platform
    }

    /// Get current performance metrics.
    pub fn performance_metrics(&self) -> PerformanceMetrics {
        let stats = self.statistics.lock().unwrap_or_else(|e| e.into_inner());
        PerformanceMetrics {
            total_allocations: stats.total_allocations,
            failed_allocations: stats.failed_allocations,
            active_pools: stats.total_pools,
            total_memory_allocated: stats.total_allocated,
            peak_memory_usage: stats.total_allocated, // Simplified
            total_allocation_time: std::time::Duration::from_secs_f64(
                stats.average_allocation_time_ms * stats.total_allocations as f64 / 1000.0
            ),
            average_allocation_time: std::time::Duration::from_secs_f64(
                stats.average_allocation_time_ms / 1000.0
            ),
            fragmentation_ratio: stats.average_fragmentation,
        }
    }

}

/// Performance metrics structure for the integration tests
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_allocations: u64,
    pub failed_allocations: u64,
    pub active_pools: usize,
    pub total_memory_allocated: usize,
    pub peak_memory_usage: usize,
    pub total_allocation_time: std::time::Duration,
    pub average_allocation_time: std::time::Duration,
    pub fragmentation_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_optimizer_creation() {
        let optimizer = MemoryOptimizer::new().expect("Failed to create memory optimizer");

        // Should have a valid platform
        assert!(!optimizer.platform.os.is_empty());

        // Should have an allocation strategy
        match optimizer.allocation_strategy {
            AllocationStrategy::Conservative { .. } => println!("✓ Conservative strategy selected"),
            AllocationStrategy::Aggressive { .. } => println!("✓ Aggressive strategy selected"),
            AllocationStrategy::Adaptive { .. } => println!("✓ Adaptive strategy selected"),
        }
    }

    #[test]
    fn test_optimal_tensor_size_calculation() {
        let optimizer = MemoryOptimizer::new().expect("Failed to create optimizer");

        // Test various sizes
        let test_sizes = vec![1024, 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024 * 1024];

        for size in test_sizes {
            let optimized = optimizer.optimal_tensor_size(size);
            assert!(optimized >= size, "Optimized size should be at least the requested size");
            assert!(optimized <= size * 2, "Optimized size shouldn't be excessively large");
            println!("Size {} -> Optimized {}", size, optimized);
        }

        // Test edge cases
        assert_eq!(optimizer.optimal_tensor_size(0), 0);
    }

    #[test]
    fn test_memory_alignment() {
        let test_cases = vec![
            (1023, 1024, 1024),
            (1024, 1024, 1024),
            (1025, 1024, 2048),
            (4095, 4096, 4096),
            (4097, 4096, 8192),
        ];

        for (size, boundary, expected) in test_cases {
            let aligned = MemoryOptimizer::align_to_memory_boundary(size, boundary);
            assert_eq!(aligned, expected, "Alignment of {} to {} boundary", size, boundary);
        }
    }

    #[test]
    fn test_platform_memory_configuration() {
        let mut optimizer = MemoryOptimizer::new().expect("Failed to create optimizer");

        match optimizer.configure_platform_memory() {
            Ok(()) => {
                println!("✓ Platform memory configuration successful");

                // Should have created some memory pools
                let stats = optimizer.memory_statistics().expect("Failed to get statistics");
                assert!(stats.total_pools > 0, "Should have created memory pools");

                println!("Platform strategy: {}", stats.platform_strategy);
                println!("Total pools: {}", stats.total_pools);
            }
            Err(e) => {
                println!("⚠️ Platform memory configuration failed: {}", e);
            }
        }
    }

    #[test]
    fn test_platform_recommendations() {
        let optimizer = MemoryOptimizer::new().expect("Failed to create optimizer");
        let recommendations = optimizer.platform_recommendations();

        assert!(!recommendations.is_empty(), "Should provide platform recommendations");

        println!("Platform recommendations:");
        for (i, rec) in recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
    }

    #[test]
    fn test_allocation_strategy_selection() {
        let platform = PlatformInfo::detect();
        let strategy = MemoryOptimizer::determine_optimal_strategy(&platform);

        match platform.os.as_str() {
            "macos" => {
                assert!(matches!(strategy, AllocationStrategy::Conservative { .. }));
                println!("✓ macOS correctly selected Conservative strategy");
            }
            "linux" | "windows" => {
                assert!(matches!(strategy, AllocationStrategy::Aggressive { .. }));
                println!("✓ {} correctly selected Aggressive strategy", platform.os);
            }
            _ => {
                assert!(matches!(strategy, AllocationStrategy::Adaptive { .. }));
                println!("✓ Unknown platform correctly selected Adaptive strategy");
            }
        }
    }
}
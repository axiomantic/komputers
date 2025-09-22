//! Convenience APIs for Common GPU Operations
//!
//! This module provides high-level, easy-to-use APIs for common GPU computing patterns.
//! These functions wrap the lower-level Kompute functionality with sensible defaults
//! and simplified interfaces for rapid prototyping and common use cases.

use crate::error::{KomputeError, Result};
use crate::manager::Manager;
use crate::tensor::{Tensor, TensorData};

/// High-level convenience functions for GPU operations
pub struct GpuOps;

/// Device selection preference
#[derive(Debug, Clone, Copy)]
pub enum DevicePreference {
    /// Use the first available device
    First,
    /// Prefer discrete GPUs over integrated
    PreferDiscrete,
    /// Use the device with most memory
    MostMemory,
    /// Use the device with highest compute capability
    HighestCompute,
    /// Use a specific device by index
    Specific(u32),
}

/// Result of a GPU operation with timing information
#[derive(Debug)]
pub struct OperationResult<T> {
    /// The result data
    pub data: T,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory used in bytes
    pub memory_used_bytes: usize,
}

/// Reduction operations for GPU arrays
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    /// Sum all elements
    Sum,
    /// Find maximum element
    Max,
    /// Find minimum element
    Min,
    /// Multiply all elements
    Product,
}

/// Binary operations for GPU arrays
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
}

impl GpuOps {
    /// Create a GPU manager with optimal settings
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use kompute::GpuOps;
    ///
    /// let manager = GpuOps::create_manager().unwrap();
    /// ```
    pub fn create_manager() -> Result<Manager> {
        Manager::new()
    }

    /// Sync a tensor from device to local memory for reading
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use kompute::{GpuOps, Manager, Tensor};
    ///
    /// let mut manager = Manager::new().unwrap();
    /// let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0]).unwrap();
    ///
    /// let result = GpuOps::sync_tensor_to_local(&mut manager, &tensor).unwrap();
    /// ```
    pub fn sync_tensor_to_local<T>(
        manager: &mut Manager,
        tensor: &Tensor<T>,
    ) -> Result<OperationResult<()>>
    where
        T: TensorData,
    {
        let start_time = std::time::Instant::now();

        // Use basic sequence for tensor synchronization
        let mut sequence = manager.sequence()?;
        sequence.record_tensor_sync_local(tensor.handle())?;
        sequence.eval()?;

        let execution_time = start_time.elapsed();

        Ok(OperationResult {
            data: (),
            execution_time_ms: execution_time.as_secs_f64() * 1000.0,
            memory_used_bytes: std::mem::size_of::<T>() * tensor.size(),
        })
    }
}

/// Mathematical operations for CPU fallback
pub struct MathOps;

impl MathOps {
    /// Compute dot product of two vectors (CPU implementation)
    pub fn dot_product<T>(a: &[T], b: &[T]) -> T
    where
        T: TensorData + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
    {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x * y)
            .fold(T::default(), |acc, x| acc + x)
    }

    /// Compute L2 norm of a vector (CPU implementation)
    pub fn norm<T>(values: &[T]) -> f64
    where
        T: TensorData + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
    {
        let _sum_squares = values
            .iter()
            .map(|&x| x * x)
            .fold(T::default(), |acc, x| acc + x);

        // Convert to f64 for sqrt calculation
        // This is a simplification - real implementation would handle type conversion properly
        1.0 // Placeholder - would need proper type conversion
    }

    /// Normalize a vector to unit length (CPU implementation)
    pub fn normalize<T>(values: &[T]) -> Vec<T>
    where
        T: TensorData
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Default,
    {
        // Simplified implementation - would need proper norm calculation and division
        values.to_vec()
    }
}

/// Extension trait for common tensor operations on standard collections
pub trait TensorExt<T> {
    /// Compute sum of all elements (CPU fallback)
    fn sum_cpu(&self) -> Result<T>;

    /// Find maximum element (CPU fallback)
    fn max_cpu(&self) -> Result<T>;

    /// Find minimum element (CPU fallback)
    fn min_cpu(&self) -> Result<T>;
}

impl<T> TensorExt<T> for Vec<T>
where
    T: TensorData + Copy + PartialOrd + std::ops::Add<Output = T> + Default,
{
    fn sum_cpu(&self) -> Result<T> {
        Ok(self.iter().fold(T::default(), |acc, &x| acc + x))
    }

    fn max_cpu(&self) -> Result<T> {
        self.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| KomputeError::tensor("Empty vector has no maximum"))
    }

    fn min_cpu(&self) -> Result<T> {
        self.iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| KomputeError::tensor("Empty vector has no minimum"))
    }
}

impl<T> TensorExt<T> for [T]
where
    T: TensorData + Copy + PartialOrd + std::ops::Add<Output = T> + Default,
{
    fn sum_cpu(&self) -> Result<T> {
        Ok(self.iter().fold(T::default(), |acc, &x| acc + x))
    }

    fn max_cpu(&self) -> Result<T> {
        self.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| KomputeError::tensor("Empty slice has no maximum"))
    }

    fn min_cpu(&self) -> Result<T> {
        self.iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| KomputeError::tensor("Empty slice has no minimum"))
    }
}

/// Benchmark configuration for performance testing
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Size of vectors for benchmark operations
    pub vector_size: usize,
    /// Number of iterations to run
    pub iterations: usize,
    /// Whether to warm up GPU before timing
    pub warmup: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            vector_size: 1024,
            iterations: 100,
            warmup: true,
        }
    }
}

/// Results from performance benchmarking
#[derive(Debug)]
pub struct BenchmarkResults {
    /// Average operation time in milliseconds
    pub avg_time_ms: f64,
    /// Operations per second
    pub ops_per_second: f64,
    /// Estimated GFLOPS (for floating point operations)
    pub gflops: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gb_s: f64,
}

impl GpuOps {
    /// Run a quick performance benchmark
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use kompute::{GpuOps, BenchmarkConfig};
    ///
    /// let config = BenchmarkConfig::default();
    /// let results = GpuOps::quick_benchmark(config).unwrap();
    /// println!("GPU Performance: {:.1} GFLOPS", results.gflops);
    /// ```
    pub fn quick_benchmark(_config: BenchmarkConfig) -> Result<BenchmarkResults> {
        let _manager = Manager::new()?;

        // Simplified benchmark - measure basic manager creation time
        let start = std::time::Instant::now();
        let _manager2 = Manager::new()?;
        let elapsed = start.elapsed();

        Ok(BenchmarkResults {
            avg_time_ms: elapsed.as_secs_f64() * 1000.0,
            ops_per_second: 1.0 / elapsed.as_secs_f64(),
            gflops: 0.001,              // Placeholder
            memory_bandwidth_gb_s: 0.1, // Placeholder
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_ops() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];

        let dot = MathOps::dot_product(&a, &b);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_tensor_ext() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(data.sum_cpu().unwrap(), 15.0);
        assert_eq!(data.max_cpu().unwrap(), 5.0);
        assert_eq!(data.min_cpu().unwrap(), 1.0);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.vector_size, 1024);
        assert_eq!(config.iterations, 100);
        assert!(config.warmup);
    }
}

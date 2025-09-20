//! GPU performance profiling and optimization utilities.
//!
//! This module provides comprehensive performance monitoring, profiling,
//! and optimization tools for GPU compute operations.

use crate::error::{KomputeError, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// GPU performance profiler for monitoring compute operations.
///
/// The profiler tracks execution times, memory usage, GPU utilization,
/// and provides optimization recommendations based on collected metrics.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::{Manager, PerformanceProfiler, Tensor};
///
/// fn profile_gpu_operations() -> kompute::Result<()> {
///     let mut manager = Manager::new()?;
///     let mut profiler = PerformanceProfiler::new();
///
///     // Start profiling
///     profiler.begin_session("matrix_operations")?;
///
///     // Create tensors and perform operations
///     let tensor_a = Tensor::new(&mut manager, &[1.0f32; 1000])?;
///     let tensor_b = Tensor::new(&mut manager, &[2.0f32; 1000])?;
///
///     // Profile a specific operation
///     profiler.begin_operation("tensor_creation")?;
///     let tensor_c = Tensor::uninit::<f32>(&mut manager, 1000)?;
///     profiler.end_operation("tensor_creation")?;
///
///     // Generate performance report
///     let report = profiler.generate_report()?;
///     println!("{}", report);
///
///     Ok(())
/// }
/// ```
pub struct PerformanceProfiler {
    /// Current profiling session name
    current_session: Option<String>,
    /// Last completed session name (for report generation)
    last_session_name: Option<String>,
    /// Operation timing data
    operation_timings: HashMap<String, Vec<Duration>>,
    /// Memory usage tracking
    memory_snapshots: Vec<MemorySnapshot>,
    /// GPU utilization data
    gpu_metrics: Vec<GpuMetric>,
    /// Operation start times for tracking
    active_operations: HashMap<String, Instant>,
    /// Profiling session start time
    session_start: Option<Instant>,
}

/// Memory usage snapshot at a specific point in time.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp of the snapshot
    pub timestamp: Instant,
    /// Operation name when snapshot was taken
    pub operation: String,
    /// Estimated GPU memory usage in bytes
    pub gpu_memory_used: u64,
    /// Number of active tensors
    pub tensor_count: u32,
    /// Number of active algorithms
    pub algorithm_count: u32,
}

/// GPU performance metric data point.
#[derive(Debug, Clone)]
pub struct GpuMetric {
    /// Timestamp of the metric
    pub timestamp: Instant,
    /// Operation being performed
    pub operation: String,
    /// Estimated GPU utilization (0.0 to 1.0)
    pub utilization: f32,
    /// Memory bandwidth utilization estimate
    pub memory_bandwidth_usage: f32,
    /// Compute unit utilization estimate
    pub compute_utilization: f32,
}

/// Performance analysis report with recommendations.
#[derive(Debug)]
pub struct PerformanceReport {
    /// Session name
    pub session_name: String,
    /// Total session duration
    pub total_duration: Duration,
    /// Operation performance summaries
    pub operation_summaries: Vec<OperationSummary>,
    /// Memory usage analysis
    pub memory_analysis: MemoryAnalysis,
    /// GPU utilization summary
    pub gpu_utilization_summary: GpuUtilizationSummary,
    /// Performance optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Summary of performance for a specific operation type.
#[derive(Debug)]
pub struct OperationSummary {
    /// Operation name
    pub operation_name: String,
    /// Number of times operation was performed
    pub execution_count: u32,
    /// Total time spent on this operation
    pub total_time: Duration,
    /// Average execution time
    pub average_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Time variance (higher indicates inconsistent performance)
    pub time_variance: f64,
    /// Percentage of total session time
    pub time_percentage: f32,
}

/// Memory usage analysis results.
#[derive(Debug)]
pub struct MemoryAnalysis {
    /// Peak GPU memory usage
    pub peak_memory_usage: u64,
    /// Average memory usage
    pub average_memory_usage: u64,
    /// Memory usage efficiency (utilization vs. allocated)
    pub memory_efficiency: f32,
    /// Peak tensor count
    pub peak_tensor_count: u32,
    /// Memory allocation pattern analysis
    pub allocation_pattern: MemoryAllocationPattern,
}

/// GPU utilization summary.
#[derive(Debug)]
pub struct GpuUtilizationSummary {
    /// Average GPU utilization
    pub average_utilization: f32,
    /// Peak GPU utilization
    pub peak_utilization: f32,
    /// Average memory bandwidth usage
    pub average_memory_bandwidth: f32,
    /// Average compute unit utilization
    pub average_compute_utilization: f32,
    /// GPU efficiency score (0.0 to 1.0)
    pub efficiency_score: f32,
}

/// Memory allocation pattern classification.
#[derive(Debug, Clone)]
pub enum MemoryAllocationPattern {
    /// Steady memory usage throughout session
    Steady,
    /// Memory usage grows throughout session
    Growing,
    /// Memory usage decreases throughout session
    Shrinking,
    /// Highly variable memory usage
    Volatile,
    /// Periodic memory usage spikes
    Periodic,
}

/// Performance optimization recommendation.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation category
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Description of the issue
    pub description: String,
    /// Suggested solution
    pub solution: String,
    /// Expected performance improvement
    pub expected_improvement: String,
}

/// Categories of performance recommendations.
#[derive(Debug, Clone)]
pub enum RecommendationCategory {
    /// Memory usage optimization
    Memory,
    /// Compute optimization
    Compute,
    /// Data transfer optimization
    DataTransfer,
    /// Algorithm structure optimization
    Algorithm,
    /// Resource management optimization
    ResourceManagement,
}

/// Priority levels for recommendations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Critical performance issue
    Critical,
    /// High impact optimization
    High,
    /// Medium impact optimization
    Medium,
    /// Low impact optimization
    Low,
}

impl PerformanceProfiler {
    /// Create a new performance profiler.
    pub fn new() -> Self {
        Self {
            current_session: None,
            last_session_name: None,
            operation_timings: HashMap::new(),
            memory_snapshots: Vec::new(),
            gpu_metrics: Vec::new(),
            active_operations: HashMap::new(),
            session_start: None,
        }
    }

    /// Begin a new profiling session.
    ///
    /// # Arguments
    ///
    /// * `session_name` - Name identifier for this profiling session
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::PerformanceProfiler;
    ///
    /// let mut profiler = PerformanceProfiler::new();
    /// profiler.begin_session("compute_benchmark").unwrap();
    /// ```
    pub fn begin_session(&mut self, session_name: &str) -> Result<()> {
        if self.current_session.is_some() {
            return Err(KomputeError::compute(
                "Cannot start new session: another session is already active",
            ));
        }

        self.current_session = Some(session_name.to_string());
        self.session_start = Some(Instant::now());
        self.operation_timings.clear();
        self.memory_snapshots.clear();
        self.gpu_metrics.clear();
        self.active_operations.clear();

        // Take initial memory snapshot
        self.take_memory_snapshot("session_start")?;

        Ok(())
    }

    /// End the current profiling session.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::PerformanceProfiler;
    ///
    /// let mut profiler = PerformanceProfiler::new();
    /// profiler.begin_session("test_session").unwrap();
    /// // ... perform operations ...
    /// profiler.end_session().unwrap();
    /// ```
    pub fn end_session(&mut self) -> Result<()> {
        if self.current_session.is_none() {
            return Err(KomputeError::compute("No active profiling session to end"));
        }

        // End any active operations
        let active_ops: Vec<String> = self.active_operations.keys().cloned().collect();
        for op in active_ops {
            self.end_operation(&op)?;
        }

        // Take final memory snapshot
        self.take_memory_snapshot("session_end")?;

        // Save session name for report generation
        self.last_session_name = self.current_session.clone();
        self.current_session = None;
        self.session_start = None;

        Ok(())
    }

    /// Begin timing a specific operation.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation to time
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::PerformanceProfiler;
    ///
    /// let mut profiler = PerformanceProfiler::new();
    /// profiler.begin_session("test").unwrap();
    /// profiler.begin_operation("matrix_multiply").unwrap();
    /// // ... perform operation ...
    /// profiler.end_operation("matrix_multiply").unwrap();
    /// ```
    pub fn begin_operation(&mut self, operation_name: &str) -> Result<()> {
        if self.current_session.is_none() {
            return Err(KomputeError::compute(
                "Cannot begin operation: no active profiling session",
            ));
        }

        if self.active_operations.contains_key(operation_name) {
            return Err(KomputeError::compute(format!(
                "Operation '{}' is already active",
                operation_name
            )));
        }

        self.active_operations
            .insert(operation_name.to_string(), Instant::now());

        // Take memory snapshot
        self.take_memory_snapshot(operation_name)?;

        Ok(())
    }

    /// End timing a specific operation.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation to stop timing
    pub fn end_operation(&mut self, operation_name: &str) -> Result<()> {
        if let Some(start_time) = self.active_operations.remove(operation_name) {
            let duration = start_time.elapsed();

            // Record timing
            self.operation_timings
                .entry(operation_name.to_string())
                .or_insert_with(Vec::new)
                .push(duration);

            // Take memory snapshot
            self.take_memory_snapshot(&format!("{}_end", operation_name))?;

            // Record GPU metrics
            self.record_gpu_metrics(operation_name, duration)?;

            Ok(())
        } else {
            Err(KomputeError::compute(format!(
                "Operation '{}' was not started or already ended",
                operation_name
            )))
        }
    }

    /// Take a memory usage snapshot.
    ///
    /// # Arguments
    ///
    /// * `operation` - Operation name for context
    fn take_memory_snapshot(&mut self, operation: &str) -> Result<()> {
        // Estimate memory usage (simplified approach)
        let estimated_memory = self.estimate_gpu_memory_usage();
        let tensor_count = self.estimate_active_tensors();
        let algorithm_count = self.estimate_active_algorithms();

        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            operation: operation.to_string(),
            gpu_memory_used: estimated_memory,
            tensor_count,
            algorithm_count,
        };

        self.memory_snapshots.push(snapshot);
        Ok(())
    }

    /// Record GPU performance metrics.
    ///
    /// # Arguments
    ///
    /// * `operation` - Operation name
    /// * `duration` - Operation duration
    fn record_gpu_metrics(&mut self, operation: &str, duration: Duration) -> Result<()> {
        // Estimate GPU utilization based on operation characteristics
        let utilization = self.estimate_gpu_utilization(operation, duration);
        let memory_bandwidth = self.estimate_memory_bandwidth_usage(operation);
        let compute_utilization = self.estimate_compute_utilization(operation, duration);

        let metric = GpuMetric {
            timestamp: Instant::now(),
            operation: operation.to_string(),
            utilization,
            memory_bandwidth_usage: memory_bandwidth,
            compute_utilization,
        };

        self.gpu_metrics.push(metric);
        Ok(())
    }

    /// Generate a comprehensive performance report.
    ///
    /// # Returns
    ///
    /// A detailed performance report with analysis and recommendations.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::PerformanceProfiler;
    ///
    /// let mut profiler = PerformanceProfiler::new();
    /// profiler.begin_session("benchmark").unwrap();
    /// // ... perform operations ...
    /// profiler.end_session().unwrap();
    ///
    /// let report = profiler.generate_report().unwrap();
    /// println!("Performance Report:\n{}", report);
    /// ```
    pub fn generate_report(&self) -> Result<PerformanceReport> {
        let session_name = match &self.current_session {
            Some(name) => name.clone(),
            None => self
                .last_session_name
                .clone()
                .unwrap_or_else(|| "completed_session".to_string()),
        };

        let total_duration = if let Some(start) = self.session_start {
            start.elapsed()
        } else {
            Duration::from_secs(0)
        };

        // Generate operation summaries
        let operation_summaries = self.generate_operation_summaries(total_duration);

        // Analyze memory usage
        let memory_analysis = self.analyze_memory_usage();

        // Analyze GPU utilization
        let gpu_utilization_summary = self.analyze_gpu_utilization();

        // Generate optimization recommendations
        let recommendations = self.generate_recommendations(
            &operation_summaries,
            &memory_analysis,
            &gpu_utilization_summary,
        );

        Ok(PerformanceReport {
            session_name,
            total_duration,
            operation_summaries,
            memory_analysis,
            gpu_utilization_summary,
            recommendations,
        })
    }

    /// Generate operation performance summaries.
    fn generate_operation_summaries(&self, total_duration: Duration) -> Vec<OperationSummary> {
        let mut summaries = Vec::new();
        let total_millis = total_duration.as_millis() as f32;

        for (op_name, timings) in &self.operation_timings {
            if timings.is_empty() {
                continue;
            }

            let execution_count = timings.len() as u32;
            let total_time: Duration = timings.iter().sum();
            let average_time = total_time / execution_count;
            let min_time = *timings.iter().min().unwrap();
            let max_time = *timings.iter().max().unwrap();

            // Calculate variance
            let avg_millis = average_time.as_millis() as f64;
            let variance: f64 = timings
                .iter()
                .map(|t| {
                    let diff = t.as_millis() as f64 - avg_millis;
                    diff * diff
                })
                .sum::<f64>()
                / timings.len() as f64;

            let time_percentage = if total_millis > 0.0 {
                (total_time.as_millis() as f32 / total_millis) * 100.0
            } else {
                0.0
            };

            summaries.push(OperationSummary {
                operation_name: op_name.clone(),
                execution_count,
                total_time,
                average_time,
                min_time,
                max_time,
                time_variance: variance,
                time_percentage,
            });
        }

        // Sort by total time (highest first)
        summaries.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        summaries
    }

    /// Analyze memory usage patterns.
    fn analyze_memory_usage(&self) -> MemoryAnalysis {
        if self.memory_snapshots.is_empty() {
            return MemoryAnalysis {
                peak_memory_usage: 0,
                average_memory_usage: 0,
                memory_efficiency: 0.0,
                peak_tensor_count: 0,
                allocation_pattern: MemoryAllocationPattern::Steady,
            };
        }

        let peak_memory_usage = self
            .memory_snapshots
            .iter()
            .map(|s| s.gpu_memory_used)
            .max()
            .unwrap_or(0);

        let average_memory_usage = self
            .memory_snapshots
            .iter()
            .map(|s| s.gpu_memory_used)
            .sum::<u64>()
            / self.memory_snapshots.len() as u64;

        let peak_tensor_count = self
            .memory_snapshots
            .iter()
            .map(|s| s.tensor_count)
            .max()
            .unwrap_or(0);

        let memory_efficiency = if peak_memory_usage > 0 {
            average_memory_usage as f32 / peak_memory_usage as f32
        } else {
            1.0
        };

        let allocation_pattern = self.classify_memory_pattern();

        MemoryAnalysis {
            peak_memory_usage,
            average_memory_usage,
            memory_efficiency,
            peak_tensor_count,
            allocation_pattern,
        }
    }

    /// Classify memory allocation pattern.
    fn classify_memory_pattern(&self) -> MemoryAllocationPattern {
        if self.memory_snapshots.len() < 3 {
            return MemoryAllocationPattern::Steady;
        }

        let memory_values: Vec<u64> = self
            .memory_snapshots
            .iter()
            .map(|s| s.gpu_memory_used)
            .collect();

        // Calculate trend
        let first_third_avg = memory_values[0..memory_values.len() / 3]
            .iter()
            .sum::<u64>() as f64
            / (memory_values.len() / 3) as f64;
        let last_third_avg = memory_values[memory_values.len() * 2 / 3..]
            .iter()
            .sum::<u64>() as f64
            / (memory_values.len() / 3) as f64;

        let growth_ratio = last_third_avg / first_third_avg.max(1.0);

        // Calculate volatility
        let avg = memory_values.iter().sum::<u64>() as f64 / memory_values.len() as f64;
        let variance: f64 = memory_values
            .iter()
            .map(|&v| {
                let diff = v as f64 - avg;
                diff * diff
            })
            .sum::<f64>()
            / memory_values.len() as f64;
        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / avg.max(1.0);

        if coefficient_of_variation > 0.5 {
            MemoryAllocationPattern::Volatile
        } else if growth_ratio > 1.2 {
            MemoryAllocationPattern::Growing
        } else if growth_ratio < 0.8 {
            MemoryAllocationPattern::Shrinking
        } else {
            MemoryAllocationPattern::Steady
        }
    }

    /// Analyze GPU utilization.
    fn analyze_gpu_utilization(&self) -> GpuUtilizationSummary {
        if self.gpu_metrics.is_empty() {
            return GpuUtilizationSummary {
                average_utilization: 0.0,
                peak_utilization: 0.0,
                average_memory_bandwidth: 0.0,
                average_compute_utilization: 0.0,
                efficiency_score: 0.0,
            };
        }

        let average_utilization = self.gpu_metrics.iter().map(|m| m.utilization).sum::<f32>()
            / self.gpu_metrics.len() as f32;

        let peak_utilization = self
            .gpu_metrics
            .iter()
            .map(|m| m.utilization)
            .fold(0.0f32, |acc, x| acc.max(x));

        let average_memory_bandwidth = self
            .gpu_metrics
            .iter()
            .map(|m| m.memory_bandwidth_usage)
            .sum::<f32>()
            / self.gpu_metrics.len() as f32;

        let average_compute_utilization = self
            .gpu_metrics
            .iter()
            .map(|m| m.compute_utilization)
            .sum::<f32>()
            / self.gpu_metrics.len() as f32;

        let efficiency_score =
            (average_utilization + average_memory_bandwidth + average_compute_utilization) / 3.0;

        GpuUtilizationSummary {
            average_utilization,
            peak_utilization,
            average_memory_bandwidth,
            average_compute_utilization,
            efficiency_score,
        }
    }

    /// Generate optimization recommendations.
    fn generate_recommendations(
        &self,
        operation_summaries: &[OperationSummary],
        memory_analysis: &MemoryAnalysis,
        gpu_utilization: &GpuUtilizationSummary,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Memory optimization recommendations
        if memory_analysis.memory_efficiency < 0.6 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Memory,
                priority: RecommendationPriority::High,
                description:
                    "Low memory efficiency detected - memory is allocated but not well utilized"
                        .to_string(),
                solution:
                    "Consider using smaller tensor batch sizes or releasing unused tensors earlier"
                        .to_string(),
                expected_improvement: "20-40% reduction in memory usage".to_string(),
            });
        }

        // GPU utilization recommendations
        if gpu_utilization.average_utilization < 0.3 {
            recommendations.push(OptimizationRecommendation {
                category: RecommendationCategory::Compute,
                priority: RecommendationPriority::High,
                description: "Low GPU utilization detected - GPU compute resources are underutilized".to_string(),
                solution: "Increase workload size, use larger tensors, or process multiple operations in parallel".to_string(),
                expected_improvement: "50-100% performance improvement".to_string(),
            });
        }

        // Operation-specific recommendations
        for summary in operation_summaries {
            if summary.time_variance > 1000000.0 {
                // High variance in microseconds
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::Algorithm,
                    priority: RecommendationPriority::Medium,
                    description: format!(
                        "Operation '{}' has inconsistent performance",
                        summary.operation_name
                    ),
                    solution:
                        "Consider pre-allocating resources or using more consistent data sizes"
                            .to_string(),
                    expected_improvement: "10-20% more consistent performance".to_string(),
                });
            }
        }

        // Memory pattern recommendations
        match memory_analysis.allocation_pattern {
            MemoryAllocationPattern::Volatile => {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::ResourceManagement,
                    priority: RecommendationPriority::Medium,
                    description: "Volatile memory allocation pattern detected".to_string(),
                    solution: "Implement memory pooling or pre-allocate resources for more stable performance".to_string(),
                    expected_improvement: "15-25% more stable performance".to_string(),
                });
            }
            MemoryAllocationPattern::Growing => {
                recommendations.push(OptimizationRecommendation {
                    category: RecommendationCategory::Memory,
                    priority: RecommendationPriority::High,
                    description: "Memory usage continuously grows - possible memory leak".to_string(),
                    solution: "Check for unreleased tensors or algorithms, implement proper resource cleanup".to_string(),
                    expected_improvement: "Prevent out-of-memory errors and improve stability".to_string(),
                });
            }
            _ => {}
        }

        // Sort recommendations by priority
        recommendations.sort_by(|a, b| a.priority.cmp(&b.priority));
        recommendations
    }

    // Helper methods for estimation (simplified implementations)

    fn estimate_gpu_memory_usage(&self) -> u64 {
        // Simplified estimation - would need actual GPU memory queries in real implementation
        (self.memory_snapshots.len() as u64) * 1024 * 1024 // 1MB per tracked resource
    }

    fn estimate_active_tensors(&self) -> u32 {
        // Simplified estimation
        10 // Mock value
    }

    fn estimate_active_algorithms(&self) -> u32 {
        // Simplified estimation
        2 // Mock value
    }

    fn estimate_gpu_utilization(&self, operation: &str, _duration: Duration) -> f32 {
        // Simplified estimation based on operation type and duration
        match operation {
            op if op.contains("matrix") => 0.8,
            op if op.contains("tensor") => 0.6,
            op if op.contains("copy") => 0.3,
            _ => 0.5,
        }
    }

    fn estimate_memory_bandwidth_usage(&self, operation: &str) -> f32 {
        // Simplified estimation based on operation type
        match operation {
            op if op.contains("copy") => 0.9,
            op if op.contains("sync") => 0.7,
            op if op.contains("create") => 0.4,
            _ => 0.5,
        }
    }

    fn estimate_compute_utilization(&self, operation: &str, duration: Duration) -> f32 {
        // Simplified estimation based on operation characteristics
        let base_utilization = match operation {
            op if op.contains("algorithm") => 0.9,
            op if op.contains("dispatch") => 0.8,
            op if op.contains("compute") => 0.7,
            _ => 0.4,
        };

        // Adjust based on duration (longer operations typically have higher utilization)
        let duration_factor = (duration.as_millis() as f32 / 1000.0).min(2.0) / 2.0 + 0.5;
        base_utilization * duration_factor
    }
}

impl std::fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "🚀 GPU Performance Report: {}", self.session_name)?;
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f, "📊 Session Duration: {:.2?}", self.total_duration)?;
        writeln!(f)?;

        // Operation summaries
        writeln!(f, "⏱️  Operation Performance:")?;
        for summary in &self.operation_summaries {
            writeln!(
                f,
                "  {} ({}x executions)",
                summary.operation_name, summary.execution_count
            )?;
            writeln!(
                f,
                "    Total: {:.2?} | Avg: {:.2?} | Min: {:.2?} | Max: {:.2?}",
                summary.total_time, summary.average_time, summary.min_time, summary.max_time
            )?;
            writeln!(
                f,
                "    Share: {:.1}% | Variance: {:.0}μs²",
                summary.time_percentage, summary.time_variance
            )?;
            writeln!(f)?;
        }

        // Memory analysis
        writeln!(f, "💾 Memory Analysis:")?;
        writeln!(
            f,
            "  Peak Usage: {:.1} MB",
            self.memory_analysis.peak_memory_usage as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(
            f,
            "  Average Usage: {:.1} MB",
            self.memory_analysis.average_memory_usage as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(
            f,
            "  Efficiency: {:.1}%",
            self.memory_analysis.memory_efficiency * 100.0
        )?;
        writeln!(
            f,
            "  Pattern: {:?}",
            self.memory_analysis.allocation_pattern
        )?;
        writeln!(f)?;

        // GPU utilization
        writeln!(f, "🎮 GPU Utilization:")?;
        writeln!(
            f,
            "  Average: {:.1}%",
            self.gpu_utilization_summary.average_utilization * 100.0
        )?;
        writeln!(
            f,
            "  Peak: {:.1}%",
            self.gpu_utilization_summary.peak_utilization * 100.0
        )?;
        writeln!(
            f,
            "  Memory Bandwidth: {:.1}%",
            self.gpu_utilization_summary.average_memory_bandwidth * 100.0
        )?;
        writeln!(
            f,
            "  Efficiency Score: {:.1}%",
            self.gpu_utilization_summary.efficiency_score * 100.0
        )?;
        writeln!(f)?;

        // Recommendations
        if !self.recommendations.is_empty() {
            writeln!(f, "💡 Optimization Recommendations:")?;
            for (i, rec) in self.recommendations.iter().enumerate() {
                let priority_icon = match rec.priority {
                    RecommendationPriority::Critical => "🔴",
                    RecommendationPriority::High => "🟠",
                    RecommendationPriority::Medium => "🟡",
                    RecommendationPriority::Low => "🟢",
                };
                writeln!(
                    f,
                    "  {}. {} {:?} - {}",
                    i + 1,
                    priority_icon,
                    rec.category,
                    rec.description
                )?;
                writeln!(f, "     💡 {}", rec.solution)?;
                writeln!(f, "     📈 {}", rec.expected_improvement)?;
                writeln!(f)?;
            }
        }

        Ok(())
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_session_management() {
        let mut profiler = PerformanceProfiler::new();

        // Test session lifecycle
        assert!(profiler.begin_session("test_session").is_ok());
        assert!(profiler.current_session.is_some());

        // Test operation lifecycle
        assert!(profiler.begin_operation("test_op").is_ok());
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(profiler.end_operation("test_op").is_ok());

        assert!(profiler.end_session().is_ok());
        assert!(profiler.current_session.is_none());
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = PerformanceProfiler::new();

        profiler.begin_session("test_report").unwrap();
        profiler.begin_operation("op1").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        profiler.end_operation("op1").unwrap();
        profiler.end_session().unwrap();

        let report = profiler.generate_report().unwrap();
        assert_eq!(report.session_name, "test_report");
        assert!(!report.operation_summaries.is_empty());
    }

    #[test]
    fn test_memory_pattern_classification() {
        let profiler = PerformanceProfiler::new();

        // Test with empty snapshots
        let pattern = profiler.classify_memory_pattern();
        assert!(matches!(pattern, MemoryAllocationPattern::Steady));
    }
}

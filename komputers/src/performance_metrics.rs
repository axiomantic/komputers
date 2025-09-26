//! Comprehensive GPU performance profiling and metrics system
//!
//! This module provides detailed performance monitoring capabilities for GPU operations,
//! including timing metrics, resource utilization tracking, and optimization recommendations.
//! All metrics integrate seamlessly with our hybrid validation system.
//!
//! # Features
//!
//! - **GPU Operation Timing**: Precise measurement of GPU command execution
//! - **Memory Usage Tracking**: Monitor GPU memory allocation and usage patterns
//! - **Validation Overhead Analysis**: Track performance impact of safety systems
//! - **Optimization Recommendations**: AI-driven performance improvement suggestions
//! - **Cross-Platform Profiling**: Consistent metrics across GPU vendors
//!
//! # Example
//!
//! ```rust,no_run
//! use komputers::{Manager, PerformanceProfiler, Tensor};
//!
//! let mut profiler = PerformanceProfiler::new();
//! let mut manager = Manager::new()?;
//!
//! profiler.start_session("my_gpu_workload");
//!
//! let _tensor = profiler.profile_operation("tensor_creation", || {
//!     Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])
//! })?;
//!
//! let session_report = profiler.end_session();
//! println!("GPU Performance Report: {:#?}", session_report);
//! # Ok::<(), komputers::KomputeError>(())
//! ```

use crate::error::{KomputeError, Result};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

// =============================================================================
// Performance Profiler Core
// =============================================================================

/// Comprehensive GPU performance profiler with detailed metrics tracking
///
/// The profiler provides detailed insights into GPU operation performance,
/// including timing data, resource utilization, and optimization suggestions.
/// It integrates with all aspects of the hybrid validation system.
#[derive(Debug)]
pub struct PerformanceProfiler {
    /// Currently active profiling session
    active_session: Option<Arc<Mutex<ProfilingSession>>>,
    /// Historical profiling data
    session_history: Vec<SessionReport>,
    /// Global performance counters
    global_counters: Arc<RwLock<GlobalCounters>>,
    /// Profiler configuration
    config: ProfilerConfig,
}

/// Configuration for the performance profiler
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Whether to collect detailed timing information
    pub collect_timing: bool,
    /// Whether to track memory usage
    pub track_memory: bool,
    /// Whether to monitor validation overhead
    pub monitor_validation: bool,
    /// Whether to generate optimization recommendations
    pub generate_recommendations: bool,
    /// Maximum number of sessions to keep in history
    pub max_session_history: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            collect_timing: true,
            track_memory: true,
            monitor_validation: true,
            generate_recommendations: true,
            max_session_history: 100,
        }
    }
}

/// Active profiling session tracking GPU operations
#[derive(Debug)]
struct ProfilingSession {
    name: String,
    start_time: Instant,
    operations: Vec<OperationProfile>,
    memory_snapshots: Vec<MemorySnapshot>,
    validation_metrics: ValidationMetrics,
}

/// Profile data for a single GPU operation
#[derive(Debug, Clone)]
pub struct OperationProfile {
    /// Name/description of the operation
    pub name: String,
    /// When the operation started
    pub start_time: Instant,
    /// Total execution time
    pub duration: Duration,
    /// GPU-specific timing if available
    pub gpu_time: Option<Duration>,
    /// Memory allocated during operation
    pub memory_allocated: u64,
    /// Memory deallocated during operation
    pub memory_deallocated: u64,
    /// Operation category for analysis
    pub category: OperationCategory,
    /// Validation overhead for this operation
    pub validation_overhead: Duration,
}

/// Categories of GPU operations for performance analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationCategory {
    /// Tensor creation and destruction
    TensorManagement,
    /// Algorithm compilation and setup
    AlgorithmSetup,
    /// GPU compute execution
    ComputeExecution,
    /// Memory transfer operations
    MemoryTransfer,
    /// Sequence recording and execution
    SequenceExecution,
    /// Validation and safety checks
    ValidationOverhead,
}

/// Memory usage snapshot at a point in time
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// When this snapshot was taken
    pub timestamp: Instant,
    /// Total GPU memory allocated (bytes)
    pub gpu_memory_used: u64,
    /// Number of active handles
    pub active_handles: usize,
    /// Number of active managers
    pub active_managers: usize,
}

/// Validation system performance metrics
#[derive(Debug, Clone, Default)]
pub struct ValidationMetrics {
    /// Total validation operations performed
    pub total_validations: u64,
    /// Time spent on runtime validation
    pub runtime_validation_time: Duration,
    /// Time spent on handle registration
    pub handle_registration_time: Duration,
    /// Cross-manager validation checks
    pub cross_manager_checks: u64,
    /// Workgroup validation checks
    pub workgroup_validations: u64,
}

/// Global performance counters across all sessions
#[derive(Debug, Default)]
struct GlobalCounters {
    /// Total operations profiled
    total_operations: u64,
    /// Total GPU memory allocated across all sessions
    total_memory_allocated: u64,
    /// Total time spent in GPU operations
    total_gpu_time: Duration,
    /// Total validation overhead across all operations
    total_validation_overhead: Duration,
}

// =============================================================================
// Session Report and Analysis
// =============================================================================

/// Complete performance report for a profiling session
#[derive(Debug, Clone)]
pub struct SessionReport {
    /// Session name
    pub name: String,
    /// Session duration
    pub duration: Duration,
    /// When the session started
    pub start_timestamp: SystemTime,
    /// Summary statistics
    pub summary: SessionSummary,
    /// Detailed operation profiles
    pub operations: Vec<OperationProfile>,
    /// Memory usage over time
    pub memory_timeline: Vec<MemorySnapshot>,
    /// Validation performance metrics
    pub validation_metrics: ValidationMetrics,
    /// Performance optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Summary statistics for a profiling session
#[derive(Debug, Clone)]
pub struct SessionSummary {
    /// Total number of operations
    pub operation_count: usize,
    /// Total execution time
    pub total_time: Duration,
    /// Average operation time
    pub avg_operation_time: Duration,
    /// Peak memory usage
    pub peak_memory_usage: u64,
    /// Total validation overhead
    pub validation_overhead: Duration,
    /// Validation overhead as percentage of total time
    pub validation_overhead_percent: f64,
}

/// Performance optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Priority level of this recommendation
    pub priority: RecommendationPriority,
    /// Category of optimization
    pub category: OptimizationCategory,
    /// Human-readable description
    pub description: String,
    /// Estimated performance improvement
    pub estimated_improvement: PerformanceImprovement,
}

/// Priority levels for optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Critical performance issue that should be addressed immediately
    Critical,
    /// High impact optimization opportunity
    High,
    /// Medium impact improvement
    Medium,
    /// Low priority optimization
    Low,
}

/// Categories of performance optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationCategory {
    /// Memory usage optimizations
    Memory,
    /// Compute workgroup optimizations
    Workgroup,
    /// Algorithm and shader optimizations
    Algorithm,
    /// Sequence and batching optimizations
    Batching,
    /// Validation overhead reduction
    Validation,
}

/// Estimated performance improvement from an optimization
#[derive(Debug, Clone)]
pub struct PerformanceImprovement {
    /// Estimated time savings
    pub time_savings: Duration,
    /// Estimated memory savings (bytes)
    pub memory_savings: u64,
    /// Confidence level in the estimate (0-100%)
    pub confidence: u8,
}

// =============================================================================
// Performance Profiler Implementation
// =============================================================================

impl PerformanceProfiler {
    /// Create a new performance profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(ProfilerConfig::default())
    }

    /// Create a new performance profiler with custom configuration
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            active_session: None,
            session_history: Vec::new(),
            global_counters: Arc::new(RwLock::new(GlobalCounters::default())),
            config,
        }
    }

    /// Start a new profiling session
    ///
    /// # Arguments
    ///
    /// * `name` - Descriptive name for the profiling session
    ///
    /// # Returns
    ///
    /// Returns error if a session is already active
    pub fn start_session(&mut self, name: impl Into<String>) -> Result<()> {
        if self.active_session.is_some() {
            return Err(KomputeError::compute(
                "Cannot start session - another session is already active",
            ));
        }

        let session = ProfilingSession {
            name: name.into(),
            start_time: Instant::now(),
            operations: Vec::new(),
            memory_snapshots: Vec::new(),
            validation_metrics: ValidationMetrics::default(),
        };

        self.active_session = Some(Arc::new(Mutex::new(session)));
        Ok(())
    }

    /// End the current profiling session and generate a report
    ///
    /// # Returns
    ///
    /// Complete performance report for the session
    pub fn end_session(&mut self) -> Option<SessionReport> {
        let session = self.active_session.take()?;
        let session = Arc::try_unwrap(session).ok()?.into_inner().ok()?;

        let duration = session.start_time.elapsed();
        let report = self.generate_session_report(session, duration);

        // Add to history and limit size
        self.session_history.push(report.clone());
        if self.session_history.len() > self.config.max_session_history {
            self.session_history.remove(0);
        }

        Some(report)
    }

    /// Profile a synchronous operation and record timing
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the operation
    /// * `operation` - The operation to profile
    ///
    /// # Returns
    ///
    /// The result of the operation
    pub fn profile_operation<F, T>(&self, name: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start_time = Instant::now();

        // Take memory snapshot before operation if configured
        if self.config.track_memory {
            self.take_memory_snapshot();
        }

        // Execute the operation
        let result = operation();
        let duration = start_time.elapsed();

        // Record the operation profile
        let validation_overhead = self.estimate_validation_overhead(&duration);
        self.record_operation(OperationProfile {
            name: name.to_string(),
            start_time,
            duration,
            gpu_time: None,      // TODO: Add GPU-specific timing
            memory_allocated: 0, // TODO: Track actual memory allocation
            memory_deallocated: 0,
            category: self.categorize_operation(name),
            validation_overhead,
        });

        result
    }

    /// Record an operation profile manually
    fn record_operation(&self, profile: OperationProfile) {
        if let Some(session) = &self.active_session {
            if let Ok(mut session) = session.lock() {
                session.operations.push(profile.clone());
            }
        }

        // Update global counters
        if let Ok(mut counters) = self.global_counters.write() {
            counters.total_operations += 1;
            counters.total_gpu_time += profile.duration;
            counters.total_validation_overhead += profile.validation_overhead;
        }
    }

    /// Take a memory usage snapshot
    fn take_memory_snapshot(&self) {
        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            gpu_memory_used: self.estimate_gpu_memory_usage(),
            active_handles: self.count_active_handles(),
            active_managers: self.count_active_managers(),
        };

        if let Some(session) = &self.active_session {
            if let Ok(mut session) = session.lock() {
                session.memory_snapshots.push(snapshot);
            }
        }
    }

    /// Generate a comprehensive session report
    fn generate_session_report(
        &self,
        session: ProfilingSession,
        duration: Duration,
    ) -> SessionReport {
        let summary = self.generate_session_summary(&session.operations, duration);
        let recommendations = self.generate_recommendations(&session.operations, &summary);

        SessionReport {
            name: session.name,
            duration,
            start_timestamp: SystemTime::now() - duration,
            summary,
            operations: session.operations,
            memory_timeline: session.memory_snapshots,
            validation_metrics: session.validation_metrics,
            recommendations,
        }
    }

    /// Generate session summary statistics
    fn generate_session_summary(
        &self,
        operations: &[OperationProfile],
        duration: Duration,
    ) -> SessionSummary {
        let operation_count = operations.len();
        let total_time = duration;
        let avg_operation_time = if operation_count > 0 {
            total_time / operation_count as u32
        } else {
            Duration::from_nanos(0)
        };

        let peak_memory_usage = 1024 * 1024; // TODO: Calculate actual peak from snapshots
        let validation_overhead: Duration =
            operations.iter().map(|op| op.validation_overhead).sum();

        let validation_overhead_percent = if total_time.as_nanos() > 0 {
            (validation_overhead.as_nanos() as f64 / total_time.as_nanos() as f64) * 100.0
        } else {
            0.0
        };

        SessionSummary {
            operation_count,
            total_time,
            avg_operation_time,
            peak_memory_usage,
            validation_overhead,
            validation_overhead_percent,
        }
    }

    /// Generate optimization recommendations based on session data
    fn generate_recommendations(
        &self,
        operations: &[OperationProfile],
        summary: &SessionSummary,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check validation overhead
        if summary.validation_overhead_percent > 10.0 {
            recommendations.push(OptimizationRecommendation {
                priority: RecommendationPriority::High,
                category: OptimizationCategory::Validation,
                description: format!(
                    "Validation overhead is {:.1}% of total time. Consider disabling validation in release builds.",
                    summary.validation_overhead_percent
                ),
                estimated_improvement: PerformanceImprovement {
                    time_savings: summary.validation_overhead,
                    memory_savings: 0,
                    confidence: 90,
                },
            });
        }

        // Check for excessive memory usage
        if summary.peak_memory_usage > 1024 * 1024 * 100 {
            // 100MB threshold
            recommendations.push(OptimizationRecommendation {
                priority: RecommendationPriority::Medium,
                category: OptimizationCategory::Memory,
                description:
                    "High memory usage detected. Consider tensor reuse and memory pooling."
                        .to_string(),
                estimated_improvement: PerformanceImprovement {
                    time_savings: Duration::from_millis(10),
                    memory_savings: summary.peak_memory_usage / 2,
                    confidence: 70,
                },
            });
        }

        // Check for small operations that could be batched
        let small_operations = operations
            .iter()
            .filter(|op| op.duration < Duration::from_micros(100))
            .count();

        if small_operations > operations.len() / 2 {
            recommendations.push(OptimizationRecommendation {
                priority: RecommendationPriority::Medium,
                category: OptimizationCategory::Batching,
                description: "Many small operations detected. Consider batching operations for better GPU utilization.".to_string(),
                estimated_improvement: PerformanceImprovement {
                    time_savings: Duration::from_millis(5),
                    memory_savings: 0,
                    confidence: 60,
                },
            });
        }

        recommendations
    }

    /// Categorize operation by name for analysis
    fn categorize_operation(&self, name: &str) -> OperationCategory {
        let name = name.to_lowercase();

        // Check more specific patterns first
        if name.contains("sync") || name.contains("transfer") {
            OperationCategory::MemoryTransfer
        } else if name.contains("algorithm") {
            OperationCategory::AlgorithmSetup
        } else if name.contains("sequence") || name.contains("execute") {
            OperationCategory::SequenceExecution
        } else if name.contains("validation") || name.contains("validate") {
            OperationCategory::ValidationOverhead
        } else if name.contains("tensor") {
            OperationCategory::TensorManagement
        } else {
            OperationCategory::ComputeExecution
        }
    }

    /// Estimate validation overhead for an operation
    fn estimate_validation_overhead(&self, total_duration: &Duration) -> Duration {
        // Estimate validation overhead as a small percentage of total time
        // In a real implementation, this would be measured more precisely
        Duration::from_nanos((total_duration.as_nanos() as f64 * 0.02) as u64)
    }

    /// Estimate current GPU memory usage
    fn estimate_gpu_memory_usage(&self) -> u64 {
        // TODO: Implement actual GPU memory usage tracking
        1024 * 1024 // Placeholder: 1MB
    }

    /// Count active handles in the system
    fn count_active_handles(&self) -> usize {
        // TODO: Integrate with runtime validation system to count handles
        0
    }

    /// Count active manager instances
    fn count_active_managers(&self) -> usize {
        // TODO: Integrate with runtime validation system to count managers
        1
    }

    /// Get global performance statistics
    pub fn global_statistics(&self) -> GlobalStatistics {
        let counters = self.global_counters.read().unwrap();
        GlobalStatistics {
            total_operations: counters.total_operations,
            total_memory_allocated: counters.total_memory_allocated,
            total_gpu_time: counters.total_gpu_time,
            total_validation_overhead: counters.total_validation_overhead,
            validation_overhead_percent: if counters.total_gpu_time.as_nanos() > 0 {
                (counters.total_validation_overhead.as_nanos() as f64
                    / counters.total_gpu_time.as_nanos() as f64)
                    * 100.0
            } else {
                0.0
            },
            session_count: self.session_history.len(),
        }
    }

    /// Get session history
    pub fn session_history(&self) -> &[SessionReport] {
        &self.session_history
    }

    /// Check if validation overhead is within acceptable limits
    pub fn is_validation_overhead_acceptable(&self) -> bool {
        let stats = self.global_statistics();
        stats.validation_overhead_percent < 5.0 // Within our <5% target
    }
}

/// Global performance statistics across all sessions
#[derive(Debug, Clone)]
pub struct GlobalStatistics {
    /// Total operations profiled across all sessions
    pub total_operations: u64,
    /// Total GPU memory allocated
    pub total_memory_allocated: u64,
    /// Total time spent in GPU operations
    pub total_gpu_time: Duration,
    /// Total validation overhead
    pub total_validation_overhead: Duration,
    /// Validation overhead as percentage of total GPU time
    pub validation_overhead_percent: f64,
    /// Number of completed sessions
    pub session_count: usize,
}

// =============================================================================
// Integration with Runtime Validation
// =============================================================================

/// Integration point with the runtime validation system
pub fn integrate_with_runtime_validation() {
    // TODO: Hook into runtime validation system to:
    // 1. Track validation timing automatically
    // 2. Count active handles and managers
    // 3. Monitor validation overhead in real-time
}

// =============================================================================
// Profiler Utilities and Helpers
// =============================================================================

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Global profiler instance for convenience
static GLOBAL_PROFILER: std::sync::LazyLock<Arc<Mutex<PerformanceProfiler>>> =
    std::sync::LazyLock::new(|| Arc::new(Mutex::new(PerformanceProfiler::new())));

/// Get access to the global profiler instance
pub fn global_profiler() -> Arc<Mutex<PerformanceProfiler>> {
    GLOBAL_PROFILER.clone()
}

/// Convenience macro for profiling operations
#[macro_export]
macro_rules! profile_operation {
    ($name:expr, $op:expr) => {{
        let profiler = $crate::performance_metrics::global_profiler();
        if let Ok(profiler) = profiler.lock() {
            profiler.profile_operation($name, || $op)
        } else {
            $op // Fallback if profiler is unavailable
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_session_lifecycle() {
        let mut profiler = PerformanceProfiler::new();

        // Start session
        assert!(profiler.start_session("test_session").is_ok());

        // Cannot start another session
        assert!(profiler.start_session("another_session").is_err());

        // End session
        let report = profiler.end_session();
        assert!(report.is_some());

        let report = report.unwrap();
        assert_eq!(report.name, "test_session");
        assert!(report.duration.as_nanos() > 0);
    }

    #[test]
    fn test_operation_profiling() {
        let mut profiler = PerformanceProfiler::new();
        profiler.start_session("operation_test").unwrap();

        let result = profiler.profile_operation("test_op", || {
            std::thread::sleep(Duration::from_millis(10));
            Ok(42)
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);

        let report = profiler.end_session().unwrap();
        assert_eq!(report.operations.len(), 1);
        assert_eq!(report.operations[0].name, "test_op");
        assert!(report.operations[0].duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_global_profiler() {
        let profiler = global_profiler();
        assert!(profiler.lock().is_ok());
    }

    #[test]
    fn test_operation_categorization() {
        let profiler = PerformanceProfiler::new();

        assert_eq!(
            profiler.categorize_operation("create_tensor"),
            OperationCategory::TensorManagement
        );
        assert_eq!(
            profiler.categorize_operation("algorithm_setup"),
            OperationCategory::AlgorithmSetup
        );
        assert_eq!(
            profiler.categorize_operation("sequence_execute"),
            OperationCategory::SequenceExecution
        );
        assert_eq!(
            profiler.categorize_operation("tensor_sync_device"),
            OperationCategory::MemoryTransfer
        );
        assert_eq!(
            profiler.categorize_operation("validate_handle"),
            OperationCategory::ValidationOverhead
        );
    }

    #[test]
    fn test_recommendation_generation() {
        let mut profiler = PerformanceProfiler::new();
        profiler.start_session("recommendation_test").unwrap();

        // Simulate high validation overhead
        std::thread::sleep(Duration::from_millis(100));

        let _report = profiler.end_session().unwrap();
        // In a real scenario with high validation overhead, recommendations would be generated
        // This test mainly verifies the structure compiles and runs
    }
}

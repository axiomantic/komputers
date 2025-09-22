//! Advanced Debugging and Diagnostic Tools
//!
//! This module provides comprehensive debugging utilities for GPU operations,
//! memory management, and system diagnostics to help developers identify and resolve
//! issues in their GPU computing applications.

use crate::error::{KomputeError, Result};
use crate::manager::Manager;
use crate::tensor::{Tensor, TensorData};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Advanced debugging context that tracks system state and operations
pub struct DebugContext {
    session_id: String,
    start_time: Instant,
    operation_log: Vec<DebugEvent>,
    memory_snapshots: Vec<MemorySnapshot>,
    performance_counters: PerformanceCounters,
    validation_errors: Vec<ValidationError>,
    gpu_state_history: Vec<GpuStateSnapshot>,
}

/// Individual debug event with timing and context information
#[derive(Debug, Clone)]
pub struct DebugEvent {
    /// Timestamp when the debug event occurred
    pub timestamp: Instant,
    /// Type of debug event that was recorded
    pub event_type: DebugEventType,
    /// Human-readable message describing the event
    pub message: String,
    /// Severity level of the debug event
    pub severity: DebugSeverity,
    /// Additional context information as key-value pairs
    pub context: HashMap<String, String>,
    /// Optional stack trace captured at event time
    pub stack_trace: Option<String>,
}

/// Types of debug events that can be tracked
#[derive(Debug, Clone, Copy)]
pub enum DebugEventType {
    /// GPU operation started
    OperationStart,
    /// GPU operation completed
    OperationEnd,
    /// Memory allocation
    MemoryAllocation,
    /// Memory deallocation
    MemoryDeallocation,
    /// Tensor operation
    TensorOperation,
    /// Algorithm execution
    AlgorithmExecution,
    /// Sequence recording
    SequenceRecording,
    /// Validation check
    Validation,
    /// Performance measurement
    Performance,
    /// Error condition
    Error,
    /// Warning condition
    Warning,
}

/// Severity levels for debug events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DebugSeverity {
    /// Informational message
    Info,
    /// Warning that doesn't prevent operation
    Warning,
    /// Error that may affect functionality
    Error,
    /// Critical error that prevents operation
    Critical,
}

/// Snapshot of memory usage at a specific point in time
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// When this memory snapshot was taken
    pub timestamp: Instant,
    /// Total GPU memory allocated in bytes
    pub total_allocated: usize,
    /// Peak memory usage observed in bytes
    pub peak_usage: usize,
    /// Number of active tensor objects
    pub active_tensors: usize,
    /// Number of active algorithm objects
    pub active_algorithms: usize,
    /// Memory fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_ratio: f32,
    /// Device-specific memory information as key-value pairs
    pub device_memory_info: HashMap<String, usize>,
}

/// Performance counters for various system operations
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PerformanceCounters {
    /// Total number of tensor operations performed
    pub tensor_operations: u64,
    /// Total number of algorithm dispatches to GPU
    pub algorithm_dispatches: u64,
    /// Total number of memory allocation operations
    pub memory_allocations: u64,
    /// Total number of sequence evaluations
    pub sequence_evaluations: u64,
    /// Total GPU execution time in milliseconds
    pub total_gpu_time_ms: f64,
    /// Total CPU processing time in milliseconds
    pub total_cpu_time_ms: f64,
    /// Total bytes transferred between host and device
    pub data_transfer_bytes: u64,
}

/// Validation error with detailed context
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// When this validation error was detected
    pub timestamp: Instant,
    /// Type category of the validation error
    pub error_type: ValidationErrorType,
    /// Human-readable error message
    pub message: String,
    /// Optional suggested fix for the error
    pub suggested_fix: Option<String>,
    /// List of resource names affected by this error
    pub affected_resources: Vec<String>,
}

impl serde::Serialize for ValidationError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ValidationError", 5)?;
        state.serialize_field("timestamp_ms", &self.timestamp.elapsed().as_millis())?;
        state.serialize_field("error_type", &self.error_type)?;
        state.serialize_field("message", &self.message)?;
        state.serialize_field("suggested_fix", &self.suggested_fix)?;
        state.serialize_field("affected_resources", &self.affected_resources)?;
        state.end()
    }
}

/// Types of validation errors
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum ValidationErrorType {
    /// Tensor size mismatch
    TensorSizeMismatch,
    /// Data type incompatibility
    DataTypeIncompatible,
    /// Memory access violation
    MemoryViolation,
    /// Algorithm parameter error
    AlgorithmParameter,
    /// Resource handle invalid
    InvalidHandle,
    /// GPU driver issue
    DriverIssue,
    /// Platform compatibility problem
    PlatformCompatibility,
}

/// Snapshot of GPU state including device capabilities and current usage
#[derive(Debug, Clone)]
pub struct GpuStateSnapshot {
    /// When this GPU state snapshot was captured
    pub timestamp: Instant,
    /// Vulkan device ID
    pub device_id: u32,
    /// Human-readable device name
    pub device_name: String,
    /// Total GPU memory in bytes
    pub memory_total: usize,
    /// Available GPU memory in bytes
    pub memory_available: usize,
    /// Number of active compute units/cores
    pub active_compute_units: u32,
    /// GPU temperature in Celsius (if available)
    pub temperature_celsius: Option<f32>,
    /// GPU power usage in watts (if available)
    pub power_usage_watts: Option<f32>,
    /// GPU utilization percentage (if available)
    pub utilization_percent: Option<f32>,
}

impl DebugContext {
    /// Create a new debug context for a session
    pub fn new(session_name: &str) -> Self {
        Self {
            session_id: format!("{}_{}", session_name, uuid::Uuid::new_v4()),
            start_time: Instant::now(),
            operation_log: Vec::new(),
            memory_snapshots: Vec::new(),
            performance_counters: PerformanceCounters::default(),
            validation_errors: Vec::new(),
            gpu_state_history: Vec::new(),
        }
    }

    /// Log a debug event with context
    pub fn log_event(
        &mut self,
        event_type: DebugEventType,
        severity: DebugSeverity,
        message: &str,
        context: Option<HashMap<String, String>>,
    ) {
        let event = DebugEvent {
            timestamp: Instant::now(),
            event_type,
            message: message.to_string(),
            severity,
            context: context.unwrap_or_default(),
            stack_trace: if severity >= DebugSeverity::Error {
                Some(self.capture_stack_trace())
            } else {
                None
            },
        };

        self.operation_log.push(event);
    }

    /// Capture current memory state
    pub fn snapshot_memory(&mut self, manager: &Manager) -> Result<()> {
        let snapshot = MemorySnapshot {
            timestamp: Instant::now(),
            total_allocated: self.estimate_total_memory(manager)?,
            peak_usage: self.get_peak_memory_usage(),
            active_tensors: self.count_active_tensors(),
            active_algorithms: self.count_active_algorithms(),
            fragmentation_ratio: self.calculate_fragmentation_ratio(),
            device_memory_info: self.collect_device_memory_info(manager)?,
        };

        self.memory_snapshots.push(snapshot);
        Ok(())
    }

    /// Capture GPU state information
    pub fn snapshot_gpu_state(&mut self, _manager: &Manager) -> Result<()> {
        let state = GpuStateSnapshot {
            timestamp: Instant::now(),
            device_id: 0,                          // Would query actual device ID
            device_name: "GPU Device".to_string(), // Would query actual device name
            memory_total: 1024 * 1024 * 1024,      // 1GB placeholder
            memory_available: 512 * 1024 * 1024,   // 512MB placeholder
            active_compute_units: 16,              // Placeholder
            temperature_celsius: None,             // Would require platform-specific queries
            power_usage_watts: None,
            utilization_percent: None,
        };

        self.gpu_state_history.push(state);
        Ok(())
    }

    /// Validate tensor operations for common issues
    pub fn validate_tensor_operation<T>(
        &mut self,
        operation: &str,
        tensors: &[&Tensor<T>],
    ) -> Result<()>
    where
        T: TensorData,
    {
        for (i, tensor) in tensors.iter().enumerate() {
            // Check tensor handle validity
            if tensor.handle() == 0 {
                self.add_validation_error(
                    ValidationErrorType::InvalidHandle,
                    &format!(
                        "Tensor {} has invalid handle in operation '{}'",
                        i, operation
                    ),
                    Some("Ensure tensor was created successfully before use".to_string()),
                    vec![format!("tensor_{}", i)],
                );
            }

            // Check tensor size reasonableness
            if tensor.size() == 0 {
                self.add_validation_error(
                    ValidationErrorType::TensorSizeMismatch,
                    &format!("Tensor {} has zero size in operation '{}'", i, operation),
                    Some("Ensure tensor is initialized with data or proper size".to_string()),
                    vec![format!("tensor_{}", i)],
                );
            }
        }

        // Cross-tensor validation for operations requiring multiple tensors
        if tensors.len() > 1 && operation.contains("binary") {
            for i in 1..tensors.len() {
                if tensors[0].size() != tensors[i].size() {
                    self.add_validation_error(
                        ValidationErrorType::TensorSizeMismatch,
                        &format!(
                            "Tensor size mismatch in '{}': tensor 0 has size {}, tensor {} has size {}",
                            operation,
                            tensors[0].size(),
                            i,
                            tensors[i].size()
                        ),
                        Some("Ensure all tensors have the same size for binary operations".to_string()),
                        vec!["tensor_0".to_string(), format!("tensor_{}", i)],
                    );
                }
            }
        }

        Ok(())
    }

    /// Generate comprehensive debug report
    pub fn generate_report(&self) -> DebugReport {
        DebugReport {
            session_id: self.session_id.clone(),
            session_duration: self.start_time.elapsed(),
            total_events: self.operation_log.len(),
            error_count: self.count_events_by_severity(DebugSeverity::Error),
            warning_count: self.count_events_by_severity(DebugSeverity::Warning),
            performance_summary: self.performance_counters.clone(),
            memory_analysis: self.analyze_memory_usage(),
            gpu_utilization_summary: self.analyze_gpu_utilization(),
            validation_issues: self.validation_errors.clone(),
            recommendations: self.generate_recommendations(),
        }
    }

    /// Export debug log to JSON format
    pub fn export_to_json(&self) -> Result<String> {
        let report = self.generate_report();
        serde_json::to_string_pretty(&report)
            .map_err(|e| KomputeError::platform(format!("Failed to serialize debug report: {}", e)))
    }

    // Private helper methods

    fn capture_stack_trace(&self) -> String {
        // In a real implementation, this would capture actual stack trace
        // For now, we'll return a placeholder
        "Stack trace not available in this implementation".to_string()
    }

    fn estimate_total_memory(&self, _manager: &Manager) -> Result<usize> {
        // Placeholder implementation
        Ok(1024 * 1024) // 1MB
    }

    fn get_peak_memory_usage(&self) -> usize {
        self.memory_snapshots
            .iter()
            .map(|s| s.peak_usage)
            .max()
            .unwrap_or(0)
    }

    fn count_active_tensors(&self) -> usize {
        // Placeholder - would track actual tensor handles
        10
    }

    fn count_active_algorithms(&self) -> usize {
        // Placeholder - would track actual algorithm handles
        5
    }

    fn calculate_fragmentation_ratio(&self) -> f32 {
        // Placeholder - would calculate actual fragmentation
        0.15
    }

    fn collect_device_memory_info(&self, _manager: &Manager) -> Result<HashMap<String, usize>> {
        let mut info = HashMap::new();
        info.insert("total".to_string(), 1024 * 1024 * 1024); // 1GB
        info.insert("available".to_string(), 512 * 1024 * 1024); // 512MB
        info.insert("used".to_string(), 512 * 1024 * 1024); // 512MB
        Ok(info)
    }

    fn add_validation_error(
        &mut self,
        error_type: ValidationErrorType,
        message: &str,
        suggested_fix: Option<String>,
        affected_resources: Vec<String>,
    ) {
        let error = ValidationError {
            timestamp: Instant::now(),
            error_type,
            message: message.to_string(),
            suggested_fix,
            affected_resources,
        };
        self.validation_errors.push(error);
    }

    fn count_events_by_severity(&self, severity: DebugSeverity) -> usize {
        self.operation_log
            .iter()
            .filter(|event| event.severity == severity)
            .count()
    }

    fn analyze_memory_usage(&self) -> MemoryAnalysis {
        let peak_usage = self.get_peak_memory_usage();
        let avg_usage = if !self.memory_snapshots.is_empty() {
            self.memory_snapshots
                .iter()
                .map(|s| s.total_allocated)
                .sum::<usize>()
                / self.memory_snapshots.len()
        } else {
            0
        };

        MemoryAnalysis {
            peak_usage_bytes: peak_usage,
            average_usage_bytes: avg_usage,
            total_allocations: self.performance_counters.memory_allocations,
            fragmentation_score: self.calculate_fragmentation_ratio(),
            efficiency_rating: if peak_usage > 0 {
                avg_usage as f32 / peak_usage as f32
            } else {
                0.0
            },
        }
    }

    fn analyze_gpu_utilization(&self) -> GpuUtilizationAnalysis {
        GpuUtilizationAnalysis {
            average_utilization_percent: 75.0, // Placeholder
            peak_utilization_percent: 95.0,
            total_compute_time_ms: self.performance_counters.total_gpu_time_ms,
            efficiency_score: 0.8,
        }
    }

    fn generate_recommendations(&self) -> Vec<DebugRecommendation> {
        let mut recommendations = Vec::new();

        // Memory recommendations
        if self.get_peak_memory_usage() > 1024 * 1024 * 1024 {
            recommendations.push(DebugRecommendation {
                category: RecommendationCategory::Memory,
                priority: RecommendationPriority::Medium,
                title: "High memory usage detected".to_string(),
                description: "Consider optimizing tensor sizes or using memory pooling".to_string(),
                suggested_actions: vec![
                    "Review tensor allocation patterns".to_string(),
                    "Implement memory pooling".to_string(),
                    "Consider tensor reuse strategies".to_string(),
                ],
            });
        }

        // Performance recommendations
        if self.performance_counters.total_gpu_time_ms
            > self.performance_counters.total_cpu_time_ms * 2.0
        {
            recommendations.push(DebugRecommendation {
                category: RecommendationCategory::Performance,
                priority: RecommendationPriority::High,
                title: "GPU utilization could be improved".to_string(),
                description:
                    "GPU time significantly exceeds CPU time, suggesting potential bottlenecks"
                        .to_string(),
                suggested_actions: vec![
                    "Profile GPU operations for bottlenecks".to_string(),
                    "Consider batching operations".to_string(),
                    "Review algorithm efficiency".to_string(),
                ],
            });
        }

        // Validation recommendations
        if !self.validation_errors.is_empty() {
            recommendations.push(DebugRecommendation {
                category: RecommendationCategory::Correctness,
                priority: RecommendationPriority::High,
                title: format!("{} validation errors found", self.validation_errors.len()),
                description: "Address validation errors to ensure correct operation".to_string(),
                suggested_actions: vec![
                    "Review validation error details".to_string(),
                    "Fix tensor size mismatches".to_string(),
                    "Validate algorithm parameters".to_string(),
                ],
            });
        }

        recommendations
    }
}

/// Comprehensive debug report
#[derive(Debug, Clone, serde::Serialize)]
pub struct DebugReport {
    /// Unique identifier for this debug session
    pub session_id: String,
    /// Total duration of the debug session
    pub session_duration: Duration,
    /// Total number of debug events captured
    pub total_events: usize,
    /// Number of error events in the session
    pub error_count: usize,
    /// Number of warning events in the session
    pub warning_count: usize,
    /// Performance summary and metrics
    pub performance_summary: PerformanceCounters,
    /// Memory usage analysis
    pub memory_analysis: MemoryAnalysis,
    /// GPU utilization analysis
    pub gpu_utilization_summary: GpuUtilizationAnalysis,
    /// List of validation issues found
    pub validation_issues: Vec<ValidationError>,
    /// List of recommended improvements
    pub recommendations: Vec<DebugRecommendation>,
}

/// Memory usage analysis
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryAnalysis {
    /// Peak memory usage observed in bytes
    pub peak_usage_bytes: usize,
    /// Average memory usage in bytes
    pub average_usage_bytes: usize,
    /// Total number of allocation operations
    pub total_allocations: u64,
    /// Memory fragmentation score (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation_score: f32,
    /// Memory efficiency rating (0.0 = inefficient, 1.0 = perfectly efficient)
    pub efficiency_rating: f32,
}

/// GPU utilization analysis
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuUtilizationAnalysis {
    /// Average GPU utilization percentage
    pub average_utilization_percent: f32,
    /// Peak GPU utilization percentage observed
    pub peak_utilization_percent: f32,
    /// Total compute time in milliseconds
    pub total_compute_time_ms: f64,
    /// GPU efficiency score (0.0 = poor, 1.0 = excellent)
    pub efficiency_score: f32,
}

/// Debug recommendation for improving system performance or correctness
#[derive(Debug, Clone, serde::Serialize)]
pub struct DebugRecommendation {
    /// Category of the recommendation
    pub category: RecommendationCategory,
    /// Priority level of the recommendation
    pub priority: RecommendationPriority,
    /// Short title summarizing the recommendation
    pub title: String,
    /// Detailed description of the recommendation
    pub description: String,
    /// List of suggested actions to implement the recommendation
    pub suggested_actions: Vec<String>,
}

/// Categories of recommendations
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum RecommendationCategory {
    /// Memory-related optimizations and fixes
    Memory,
    /// Performance improvements and optimizations
    Performance,
    /// Correctness and reliability improvements
    Correctness,
    /// Platform and driver compatibility issues
    Compatibility,
    /// Security-related recommendations
    Security,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum RecommendationPriority {
    /// Low priority, nice to have improvements
    Low,
    /// Medium priority, should be addressed
    Medium,
    /// High priority, important to fix
    High,
    /// Critical priority, must be fixed immediately
    Critical,
}

/// Utility functions for quick debugging
pub struct DebugUtils;

impl DebugUtils {
    /// Quick tensor validation check
    pub fn validate_tensor<T>(tensor: &Tensor<T>, operation: &str) -> Result<()>
    where
        T: TensorData,
    {
        if tensor.handle() == 0 {
            return Err(KomputeError::tensor(format!(
                "Invalid tensor handle in operation '{}'",
                operation
            )));
        }

        if tensor.size() == 0 {
            return Err(KomputeError::tensor(format!(
                "Zero-sized tensor in operation '{}'",
                operation
            )));
        }

        Ok(())
    }

    /// Memory usage estimation for tensors
    pub fn estimate_tensor_memory<T>(tensor: &Tensor<T>) -> usize
    where
        T: TensorData,
    {
        tensor.size() * std::mem::size_of::<T>()
    }

    /// Create debug context with auto-generated session name
    pub fn create_debug_session() -> DebugContext {
        let session_name = format!("debug_session_{}", chrono::Utc::now().timestamp());
        DebugContext::new(&session_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_context_creation() {
        let debug_ctx = DebugContext::new("test_session");
        assert!(debug_ctx.session_id.starts_with("test_session"));
        assert_eq!(debug_ctx.operation_log.len(), 0);
        assert_eq!(debug_ctx.validation_errors.len(), 0);
    }

    #[test]
    fn test_debug_event_logging() {
        let mut debug_ctx = DebugContext::new("test_session");

        debug_ctx.log_event(
            DebugEventType::OperationStart,
            DebugSeverity::Info,
            "Test operation started",
            None,
        );

        assert_eq!(debug_ctx.operation_log.len(), 1);
        assert_eq!(debug_ctx.operation_log[0].message, "Test operation started");
        assert_eq!(debug_ctx.operation_log[0].severity, DebugSeverity::Info);
    }

    #[test]
    fn test_debug_utils() {
        let session = DebugUtils::create_debug_session();
        assert!(session.session_id.starts_with("debug_session_"));
    }
}

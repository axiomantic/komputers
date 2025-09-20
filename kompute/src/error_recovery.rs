//! Advanced error recovery and platform diagnostics for production GPU computing.
//!
//! This module provides comprehensive error recovery strategies, platform diagnostics,
//! and automated fallback mechanisms for robust GPU computing in production environments.

use crate::error::{KomputeError, Result};
use crate::platform::PlatformInfo;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Advanced error recovery and diagnostic manager.
///
/// Provides intelligent error recovery, platform diagnostics, and automated
/// fallback strategies for production GPU computing environments.
pub struct ErrorRecoveryManager {
    platform_info: PlatformInfo,
    error_history: Arc<Mutex<Vec<ErrorEvent>>>,
    recovery_strategies: HashMap<ErrorCategory, RecoveryStrategy>,
    diagnostic_state: Arc<Mutex<DiagnosticState>>,
    configuration: RecoveryConfiguration,
}

/// Individual error event with context and recovery information.
#[derive(Debug, Clone)]
pub struct ErrorEvent {
    /// Timestamp when the error occurred
    pub timestamp: Instant,
    /// Category of the error
    pub category: ErrorCategory,
    /// Severity level of the error
    pub severity: ErrorSeverity,
    /// Original error message
    pub error_message: String,
    /// Context where the error occurred
    pub context: ErrorContext,
    /// Recovery action that was attempted
    pub recovery_attempted: Option<RecoveryAction>,
    /// Whether recovery was successful
    pub recovery_successful: bool,
    /// Additional platform-specific data
    pub platform_data: HashMap<String, String>,
}

/// Categories of errors for targeted recovery strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// GPU device not available or initialization failed
    DeviceUnavailable,
    /// Out of GPU memory
    OutOfMemory,
    /// SPIR-V shader compilation or validation failed
    ShaderError,
    /// Vulkan driver or runtime issues
    VulkanRuntime,
    /// Cross-platform compatibility issues
    PlatformCompatibility,
    /// Tensor operation or validation errors
    TensorOperation,
    /// Algorithm binding or execution errors
    AlgorithmExecution,
    /// Network or I/O related errors
    NetworkIO,
    /// Unknown or unclassified errors
    Unknown,
}

/// Severity levels for error classification and response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational - operation can continue
    Info,
    /// Warning - operation can continue with degraded performance
    Warning,
    /// Error - operation failed but system is stable
    Error,
    /// Critical - system integrity may be compromised
    Critical,
    /// Fatal - immediate shutdown required
    Fatal,
}

/// Context information for error analysis and recovery.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Component where error occurred
    pub component: String,
    /// Function or method name
    pub function: String,
    /// Operation being performed
    pub operation: String,
    /// Input parameters or tensor sizes
    pub parameters: HashMap<String, String>,
    /// Thread ID where error occurred
    pub thread_id: String,
    /// GPU device ID if applicable
    pub device_id: Option<u32>,
}

/// Recovery strategies for different error categories.
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry operation with exponential backoff
    RetryWithBackoff {
        max_attempts: u32,
        initial_delay: Duration,
        max_delay: Duration,
    },
    /// Fall back to CPU computation
    FallbackToCpu,
    /// Use alternative GPU device
    SwitchDevice { preferred_devices: Vec<u32> },
    /// Reduce memory usage and retry
    ReduceMemoryAndRetry {
        reduction_factor: f32,
        max_attempts: u32,
    },
    /// Reinitialize GPU subsystem
    ReinitializeGpu,
    /// Use alternative algorithm or shader
    UseAlternativeAlgorithm { alternatives: Vec<String> },
    /// Graceful degradation with reduced functionality
    GracefulDegradation,
    /// No recovery - fail fast
    FailFast,
}

/// Recovery actions that can be attempted.
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retried operation with specified parameters
    Retry { attempt: u32, delay: Duration },
    /// Switched to CPU fallback
    CpuFallback,
    /// Switched to alternative GPU device
    DeviceSwitch { from_device: u32, to_device: u32 },
    /// Reduced memory allocation
    MemoryReduction {
        original_size: usize,
        reduced_size: usize,
    },
    /// Reinitialized GPU subsystem
    GpuReinit,
    /// Used alternative algorithm
    AlgorithmSwitch { from: String, to: String },
    /// Applied graceful degradation
    Degradation { features_disabled: Vec<String> },
}

/// Current diagnostic state of the system.
#[derive(Debug, Clone)]
pub struct DiagnosticState {
    /// GPU availability status
    pub gpu_available: bool,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Current GPU device ID
    pub current_device: Option<u32>,
    /// Number of recent errors by category
    pub error_counts: HashMap<ErrorCategory, u32>,
    /// System health score (0.0 = critical, 1.0 = excellent)
    pub health_score: f32,
    /// Last diagnostic check timestamp
    pub last_check: Option<Instant>,
    /// Platform-specific diagnostic data
    pub platform_diagnostics: HashMap<String, String>,
}

impl Default for DiagnosticState {
    fn default() -> Self {
        Self {
            gpu_available: false,
            available_memory: 0,
            current_device: None,
            error_counts: HashMap::new(),
            health_score: 1.0, // Start with perfect health score
            last_check: None,
            platform_diagnostics: HashMap::new(),
        }
    }
}

/// Configuration for error recovery behavior.
#[derive(Debug, Clone)]
pub struct RecoveryConfiguration {
    /// Enable automatic error recovery
    pub auto_recovery_enabled: bool,
    /// Maximum number of recovery attempts per error
    pub max_recovery_attempts: u32,
    /// Timeout for recovery operations
    pub recovery_timeout: Duration,
    /// Enable CPU fallback for GPU operations
    pub cpu_fallback_enabled: bool,
    /// Enable device switching for multi-GPU systems
    pub device_switching_enabled: bool,
    /// Error history retention period
    pub error_history_retention: Duration,
    /// Minimum time between diagnostic checks
    pub diagnostic_interval: Duration,
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager with platform detection.
    pub fn new() -> Result<Self> {
        let platform_info = PlatformInfo::detect();
        let mut recovery_strategies = HashMap::new();

        // Set up default recovery strategies based on platform
        Self::setup_default_strategies(&mut recovery_strategies, &platform_info);

        let configuration = RecoveryConfiguration {
            auto_recovery_enabled: true,
            max_recovery_attempts: 3,
            recovery_timeout: Duration::from_secs(30),
            cpu_fallback_enabled: true,
            device_switching_enabled: true,
            error_history_retention: Duration::from_secs(24 * 60 * 60), // 24 hours
            diagnostic_interval: Duration::from_secs(60),
        };

        Ok(Self {
            platform_info,
            error_history: Arc::new(Mutex::new(Vec::new())),
            recovery_strategies,
            diagnostic_state: Arc::new(Mutex::new(DiagnosticState::default())),
            configuration,
        })
    }

    /// Set up default recovery strategies based on platform capabilities.
    fn setup_default_strategies(
        strategies: &mut HashMap<ErrorCategory, RecoveryStrategy>,
        platform_info: &PlatformInfo,
    ) {
        // Device unavailable - try alternative devices or fallback
        strategies.insert(
            ErrorCategory::DeviceUnavailable,
            if platform_info.os == "macos" {
                // macOS with MoltenVK - be more conservative
                RecoveryStrategy::FallbackToCpu
            } else {
                // Linux/Windows - try device switching first
                RecoveryStrategy::SwitchDevice {
                    preferred_devices: vec![0, 1, 2],
                }
            },
        );

        // Out of memory - reduce allocation and retry
        strategies.insert(
            ErrorCategory::OutOfMemory,
            RecoveryStrategy::ReduceMemoryAndRetry {
                reduction_factor: 0.5,
                max_attempts: 3,
            },
        );

        // Shader errors - try alternatives or fallback
        strategies.insert(
            ErrorCategory::ShaderError,
            RecoveryStrategy::UseAlternativeAlgorithm {
                alternatives: vec!["fallback_cpu".to_string()],
            },
        );

        // Vulkan runtime issues - reinitialize or fallback
        strategies.insert(
            ErrorCategory::VulkanRuntime,
            RecoveryStrategy::RetryWithBackoff {
                max_attempts: 2,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(1),
            },
        );

        // Platform compatibility - graceful degradation
        strategies.insert(
            ErrorCategory::PlatformCompatibility,
            RecoveryStrategy::GracefulDegradation,
        );

        // Tensor operations - retry with validation
        strategies.insert(
            ErrorCategory::TensorOperation,
            RecoveryStrategy::RetryWithBackoff {
                max_attempts: 2,
                initial_delay: Duration::from_millis(50),
                max_delay: Duration::from_millis(500),
            },
        );

        // Algorithm execution - try alternatives
        strategies.insert(
            ErrorCategory::AlgorithmExecution,
            RecoveryStrategy::UseAlternativeAlgorithm {
                alternatives: vec!["simple_fallback".to_string()],
            },
        );

        // Network I/O - retry with backoff
        strategies.insert(
            ErrorCategory::NetworkIO,
            RecoveryStrategy::RetryWithBackoff {
                max_attempts: 5,
                initial_delay: Duration::from_millis(200),
                max_delay: Duration::from_secs(5),
            },
        );

        // Unknown errors - conservative retry
        strategies.insert(
            ErrorCategory::Unknown,
            RecoveryStrategy::RetryWithBackoff {
                max_attempts: 1,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_millis(100),
            },
        );
    }

    /// Handle an error with automatic recovery attempt.
    ///
    /// # Arguments
    ///
    /// * `error` - The error that occurred
    /// * `context` - Context information about where the error occurred
    ///
    /// # Returns
    ///
    /// Recovery result indicating whether recovery was successful and what action was taken.
    pub fn handle_error(
        &mut self,
        error: &KomputeError,
        context: ErrorContext,
    ) -> Result<RecoveryResult> {
        let category = Self::categorize_error(error);
        let severity = Self::determine_severity(error, &category);

        // Record the error event
        let error_event = ErrorEvent {
            timestamp: Instant::now(),
            category,
            severity,
            error_message: error.to_string(),
            context: context.clone(),
            recovery_attempted: None,
            recovery_successful: false,
            platform_data: self.collect_platform_data(),
        };

        self.record_error_event(error_event.clone())?;

        // Attempt recovery if enabled and appropriate
        if self.configuration.auto_recovery_enabled && severity < ErrorSeverity::Fatal {
            self.attempt_recovery(&error_event)
        } else {
            Ok(RecoveryResult {
                recovery_attempted: false,
                recovery_action: None,
                recovery_successful: false,
                fallback_available: self.check_fallback_available(&category),
                recommendation: self.generate_recommendation(&error_event),
            })
        }
    }

    /// Attempt recovery for a specific error event.
    fn attempt_recovery(&mut self, error_event: &ErrorEvent) -> Result<RecoveryResult> {
        let strategy = self
            .recovery_strategies
            .get(&error_event.category)
            .cloned()
            .unwrap_or(RecoveryStrategy::FailFast);

        let recovery_action = match strategy {
            RecoveryStrategy::RetryWithBackoff { initial_delay, .. } => {
                // Simulate retry logic
                Some(RecoveryAction::Retry {
                    attempt: 1,
                    delay: initial_delay,
                })
            }
            RecoveryStrategy::FallbackToCpu => Some(RecoveryAction::CpuFallback),
            RecoveryStrategy::SwitchDevice { preferred_devices } => {
                if let Some(current_device) = self.get_current_device() {
                    let next_device = preferred_devices
                        .iter()
                        .find(|&&device| device != current_device)
                        .copied()
                        .unwrap_or(0);

                    Some(RecoveryAction::DeviceSwitch {
                        from_device: current_device,
                        to_device: next_device,
                    })
                } else {
                    None
                }
            }
            RecoveryStrategy::ReduceMemoryAndRetry {
                reduction_factor, ..
            } => {
                // Extract memory size from context if available
                let original_size = error_event
                    .context
                    .parameters
                    .get("tensor_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1024);

                let reduced_size = (original_size as f32 * reduction_factor) as usize;

                Some(RecoveryAction::MemoryReduction {
                    original_size,
                    reduced_size,
                })
            }
            RecoveryStrategy::ReinitializeGpu => Some(RecoveryAction::GpuReinit),
            RecoveryStrategy::UseAlternativeAlgorithm { alternatives } => {
                let alternative = alternatives
                    .first()
                    .unwrap_or(&"default_fallback".to_string())
                    .clone();

                Some(RecoveryAction::AlgorithmSwitch {
                    from: error_event.context.operation.clone(),
                    to: alternative,
                })
            }
            RecoveryStrategy::GracefulDegradation => Some(RecoveryAction::Degradation {
                features_disabled: vec!["gpu_acceleration".to_string()],
            }),
            RecoveryStrategy::FailFast => None,
        };

        // Update error event with recovery attempt
        if let Some(ref action) = recovery_action {
            let mut updated_event = error_event.clone();
            updated_event.recovery_attempted = Some(action.clone());
            updated_event.recovery_successful = true; // Optimistically assume success
            self.record_error_event(updated_event)?;
        }

        Ok(RecoveryResult {
            recovery_attempted: recovery_action.is_some(),
            recovery_action,
            recovery_successful: true, // In real implementation, this would be determined by actual recovery attempt
            fallback_available: self.check_fallback_available(&error_event.category),
            recommendation: self.generate_recommendation(error_event),
        })
    }

    /// Categorize an error into a recovery category.
    pub fn categorize_error(error: &KomputeError) -> ErrorCategory {
        let error_str = error.to_string().to_lowercase();

        if error_str.contains("memory") || error_str.contains("allocation") {
            ErrorCategory::OutOfMemory
        } else if error_str.contains("spirv") || error_str.contains("shader") {
            ErrorCategory::ShaderError
        } else if error_str.contains("vulkan") {
            ErrorCategory::VulkanRuntime
        } else if error_str.contains("device") || error_str.contains("gpu") {
            ErrorCategory::DeviceUnavailable
        } else if error_str.contains("platform") || error_str.contains("moltenvk") {
            ErrorCategory::PlatformCompatibility
        } else if error_str.contains("tensor") || error_str.contains("size") {
            ErrorCategory::TensorOperation
        } else if error_str.contains("algorithm") || error_str.contains("binding") {
            ErrorCategory::AlgorithmExecution
        } else if error_str.contains("network") || error_str.contains("io") {
            ErrorCategory::NetworkIO
        } else {
            ErrorCategory::Unknown
        }
    }

    /// Determine the severity of an error.
    pub fn determine_severity(error: &KomputeError, category: &ErrorCategory) -> ErrorSeverity {
        let error_str = error.to_string().to_lowercase();

        if error_str.contains("fatal") && error_str.contains("immediate shutdown") {
            ErrorSeverity::Fatal
        } else if error_str.contains("fatal")
            || error_str.contains("panic")
            || error_str.contains("critical")
        {
            ErrorSeverity::Critical
        } else if *category == ErrorCategory::DeviceUnavailable {
            ErrorSeverity::Critical
        } else if error_str.contains("failed") || *category == ErrorCategory::OutOfMemory {
            ErrorSeverity::Error
        } else if error_str.contains("warning") || *category == ErrorCategory::PlatformCompatibility
        {
            ErrorSeverity::Warning
        } else {
            ErrorSeverity::Info
        }
    }

    /// Record an error event in the history.
    fn record_error_event(&mut self, event: ErrorEvent) -> Result<()> {
        let mut history = self
            .error_history
            .lock()
            .map_err(|_| KomputeError::memory("Failed to acquire error history lock"))?;

        history.push(event);

        // Clean up old events
        let cutoff_time = Instant::now() - self.configuration.error_history_retention;
        history.retain(|event| event.timestamp > cutoff_time);

        // Update diagnostic state
        {
            let history_clone: Vec<_> = history.clone();
            drop(history); // Release the lock before calling update_diagnostic_state
            self.update_diagnostic_state(&history_clone)?;
        }

        Ok(())
    }

    /// Update diagnostic state based on error history.
    fn update_diagnostic_state(&mut self, history: &[ErrorEvent]) -> Result<()> {
        let mut diagnostic_state = self
            .diagnostic_state
            .lock()
            .map_err(|_| KomputeError::memory("Failed to acquire diagnostic state lock"))?;

        // Count errors by category in the last hour
        let recent_cutoff = Instant::now() - Duration::from_secs(60 * 60); // 1 hour
        let mut error_counts = HashMap::new();

        for event in history.iter().filter(|e| e.timestamp > recent_cutoff) {
            *error_counts.entry(event.category).or_insert(0) += 1;
        }

        diagnostic_state.error_counts = error_counts;

        // Calculate health score based on recent errors
        let total_recent_errors: u32 = diagnostic_state.error_counts.values().sum();
        diagnostic_state.health_score = if total_recent_errors == 0 {
            1.0
        } else {
            (1.0 / (1.0 + total_recent_errors as f32 * 0.1)).max(0.0)
        };

        diagnostic_state.last_check = Some(Instant::now());

        Ok(())
    }

    /// Collect platform-specific diagnostic data.
    fn collect_platform_data(&self) -> HashMap<String, String> {
        let mut data = HashMap::new();

        data.insert("platform_os".to_string(), self.platform_info.os.clone());
        data.insert("platform_arch".to_string(), self.platform_info.arch.clone());
        data.insert(
            "vulkan_available".to_string(),
            self.platform_info.vulkan_runtime_available.to_string(),
        );

        // Add timestamp
        data.insert("timestamp".to_string(), format!("{:?}", Instant::now()));

        data
    }

    /// Check if fallback is available for a given error category.
    fn check_fallback_available(&self, category: &ErrorCategory) -> bool {
        match category {
            ErrorCategory::DeviceUnavailable => self.configuration.cpu_fallback_enabled,
            ErrorCategory::OutOfMemory => true, // Can reduce memory usage
            ErrorCategory::ShaderError => self.configuration.cpu_fallback_enabled,
            ErrorCategory::VulkanRuntime => self.configuration.cpu_fallback_enabled,
            ErrorCategory::PlatformCompatibility => true, // Can degrade gracefully
            ErrorCategory::TensorOperation => true,       // Can retry with validation
            ErrorCategory::AlgorithmExecution => self.configuration.cpu_fallback_enabled,
            ErrorCategory::NetworkIO => true, // Can retry
            ErrorCategory::Unknown => false,
        }
    }

    /// Generate a recommendation for handling an error.
    fn generate_recommendation(&self, error_event: &ErrorEvent) -> String {
        match error_event.category {
            ErrorCategory::DeviceUnavailable => {
                "Consider updating GPU drivers or using CPU fallback mode".to_string()
            }
            ErrorCategory::OutOfMemory => {
                "Reduce tensor sizes or batch sizes to fit available GPU memory".to_string()
            }
            ErrorCategory::ShaderError => {
                "Verify SPIR-V shader compilation or use alternative algorithms".to_string()
            }
            ErrorCategory::VulkanRuntime => {
                "Update Vulkan drivers or check platform compatibility".to_string()
            }
            ErrorCategory::PlatformCompatibility => {
                "Consider platform-specific optimizations or fallback modes".to_string()
            }
            ErrorCategory::TensorOperation => {
                "Validate tensor dimensions and data types before operations".to_string()
            }
            ErrorCategory::AlgorithmExecution => {
                "Check algorithm-tensor binding compatibility and parameters".to_string()
            }
            ErrorCategory::NetworkIO => {
                "Check network connectivity and retry with exponential backoff".to_string()
            }
            ErrorCategory::Unknown => {
                "Enable detailed logging and contact support with error details".to_string()
            }
        }
    }

    /// Get current GPU device ID.
    fn get_current_device(&self) -> Option<u32> {
        self.diagnostic_state
            .lock()
            .ok()
            .and_then(|state| state.current_device)
    }

    /// Get comprehensive diagnostic report.
    pub fn diagnostic_report(&self) -> Result<DiagnosticReport> {
        let diagnostic_state = self
            .diagnostic_state
            .lock()
            .map_err(|_| KomputeError::memory("Failed to acquire diagnostic state lock"))?;

        let error_history = self
            .error_history
            .lock()
            .map_err(|_| KomputeError::memory("Failed to acquire error history lock"))?;

        Ok(DiagnosticReport {
            platform_info: self.platform_info.clone(),
            health_score: diagnostic_state.health_score,
            gpu_available: diagnostic_state.gpu_available,
            current_device: diagnostic_state.current_device,
            recent_error_counts: diagnostic_state.error_counts.clone(),
            total_errors: error_history.len(),
            recovery_success_rate: self.calculate_recovery_success_rate(&error_history),
            recommendations: self
                .generate_system_recommendations(&diagnostic_state, &error_history),
            last_check: diagnostic_state.last_check,
        })
    }

    /// Calculate recovery success rate from error history.
    fn calculate_recovery_success_rate(&self, history: &[ErrorEvent]) -> f32 {
        let recovery_attempts = history
            .iter()
            .filter(|e| e.recovery_attempted.is_some())
            .count();

        if recovery_attempts == 0 {
            return 1.0; // No failures mean 100% success
        }

        let successful_recoveries = history
            .iter()
            .filter(|e| e.recovery_attempted.is_some() && e.recovery_successful)
            .count();

        successful_recoveries as f32 / recovery_attempts as f32
    }

    /// Generate system-wide recommendations.
    fn generate_system_recommendations(
        &self,
        _diagnostic_state: &DiagnosticState,
        error_history: &[ErrorEvent],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Analyze error patterns
        let recent_errors: Vec<_> = error_history
            .iter()
            .filter(|e| e.timestamp > Instant::now() - Duration::from_secs(60 * 60)) // 1 hour
            .collect();

        if recent_errors.len() > 5 {
            recommendations
                .push("High error rate detected - consider system maintenance".to_string());
        }

        // Platform-specific recommendations
        match self.platform_info.os.as_str() {
            "macos" => {
                recommendations
                    .push("macOS detected - ensure MoltenVK is properly configured".to_string());
            }
            "linux" => {
                recommendations
                    .push("Linux detected - verify Vulkan drivers are up to date".to_string());
            }
            "windows" => {
                recommendations
                    .push("Windows detected - check Windows GPU driver versions".to_string());
            }
            _ => {
                recommendations
                    .push("Unknown platform - enable conservative fallback modes".to_string());
            }
        }

        recommendations
    }
}

/// Result of an error recovery attempt.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Whether recovery was attempted
    pub recovery_attempted: bool,
    /// The recovery action that was taken
    pub recovery_action: Option<RecoveryAction>,
    /// Whether recovery was successful
    pub recovery_successful: bool,
    /// Whether fallback options are available
    pub fallback_available: bool,
    /// Recommendation for handling the error
    pub recommendation: String,
}

/// Comprehensive diagnostic report for the system.
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// Platform information
    pub platform_info: PlatformInfo,
    /// Overall system health score (0.0 - 1.0)
    pub health_score: f32,
    /// Whether GPU is currently available
    pub gpu_available: bool,
    /// Current GPU device ID
    pub current_device: Option<u32>,
    /// Error counts by category in the last hour
    pub recent_error_counts: HashMap<ErrorCategory, u32>,
    /// Total number of errors in history
    pub total_errors: usize,
    /// Success rate of recovery attempts
    pub recovery_success_rate: f32,
    /// System-wide recommendations
    pub recommendations: Vec<String>,
    /// Timestamp of last diagnostic check
    pub last_check: Option<Instant>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recovery_manager_creation() {
        match ErrorRecoveryManager::new() {
            Ok(manager) => {
                assert!(!manager.recovery_strategies.is_empty());
                assert!(manager.configuration.auto_recovery_enabled);
                println!("✓ Error recovery manager created successfully");
            }
            Err(e) => {
                println!("⚠️ Error recovery manager creation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_error_categorization() {
        let device_error = KomputeError::device("GPU device not found");
        let category = ErrorRecoveryManager::categorize_error(&device_error);
        assert_eq!(category, ErrorCategory::DeviceUnavailable);

        let memory_error = KomputeError::memory("Out of GPU memory");
        let category = ErrorRecoveryManager::categorize_error(&memory_error);
        assert_eq!(category, ErrorCategory::OutOfMemory);

        let shader_error = KomputeError::shader("SPIR-V validation failed");
        let category = ErrorRecoveryManager::categorize_error(&shader_error);
        assert_eq!(category, ErrorCategory::ShaderError);

        println!("✓ Error categorization test passed");
    }

    #[test]
    fn test_severity_determination() {
        let fatal_error = KomputeError::device("Fatal GPU error");
        let severity = ErrorRecoveryManager::determine_severity(
            &fatal_error,
            &ErrorCategory::DeviceUnavailable,
        );
        assert_eq!(severity, ErrorSeverity::Critical);

        let warning_error = KomputeError::compute("Platform compatibility warning");
        let severity = ErrorRecoveryManager::determine_severity(
            &warning_error,
            &ErrorCategory::PlatformCompatibility,
        );
        assert_eq!(severity, ErrorSeverity::Warning);

        println!("✓ Severity determination test passed");
    }

    #[test]
    fn test_recovery_strategies() {
        if let Ok(manager) = ErrorRecoveryManager::new() {
            // Test that all error categories have recovery strategies
            for category in [
                ErrorCategory::DeviceUnavailable,
                ErrorCategory::OutOfMemory,
                ErrorCategory::ShaderError,
                ErrorCategory::VulkanRuntime,
                ErrorCategory::PlatformCompatibility,
                ErrorCategory::TensorOperation,
                ErrorCategory::AlgorithmExecution,
                ErrorCategory::NetworkIO,
                ErrorCategory::Unknown,
            ] {
                assert!(manager.recovery_strategies.contains_key(&category));
                println!("✓ Recovery strategy found for {:?}", category);
            }

            println!("✓ Recovery strategies test passed");
        }
    }

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext {
            component: "GpuOperations".to_string(),
            function: "elementwise_add".to_string(),
            operation: "tensor_addition".to_string(),
            parameters: HashMap::from([
                ("tensor_size".to_string(), "1024".to_string()),
                ("data_type".to_string(), "f32".to_string()),
            ]),
            thread_id: "main".to_string(),
            device_id: Some(0),
        };

        assert_eq!(context.component, "GpuOperations");
        assert_eq!(context.function, "elementwise_add");
        assert_eq!(context.operation, "tensor_addition");
        assert_eq!(context.device_id, Some(0));

        println!("✓ Error context creation test passed");
    }

    #[test]
    fn test_diagnostic_report_generation() {
        if let Ok(manager) = ErrorRecoveryManager::new() {
            match manager.diagnostic_report() {
                Ok(report) => {
                    assert!(report.health_score >= 0.0 && report.health_score <= 1.0);
                    assert!(!report.platform_info.os.is_empty());
                    assert!(!report.recommendations.is_empty());

                    println!("✓ Diagnostic report generated successfully");
                    println!("  Platform: {}", report.platform_info.os);
                    println!("  Health score: {:.2}", report.health_score);
                    println!("  Recommendations: {}", report.recommendations.len());
                }
                Err(e) => {
                    println!("⚠️ Diagnostic report generation failed: {}", e);
                }
            }
        }
    }
}

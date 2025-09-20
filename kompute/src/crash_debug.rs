//! Crash Debugging and Analysis Module
//!
//! This module provides comprehensive crash debugging capabilities for analyzing
//! segmentation faults and other critical errors in GPU operations, particularly
//! helpful for debugging MoltenVK compatibility issues.

use crate::error::{KomputeError, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

#[cfg(feature = "crash-debug")]
use {
    backtrace::Backtrace,
    signal_hook::{consts::SIGABRT, consts::SIGBUS, consts::SIGSEGV, iterator::Signals},
    std::sync::Once,
};

static CRASH_HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "crash-debug")]
static INIT: Once = Once::new();

/// Comprehensive crash information for analysis
#[derive(Debug, Clone)]
pub struct CrashInfo {
    /// Signal that caused the crash
    pub signal: i32,
    /// Timestamp when crash occurred
    pub timestamp: Instant,
    /// Stack trace at time of crash
    pub backtrace: String,
    /// Process information
    pub process_info: ProcessInfo,
    /// Memory information at crash time
    pub memory_info: MemoryInfo,
    /// Vulkan/GPU context information
    pub gpu_context: GpuContext,
}

/// Process information at crash time
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    /// Process ID
    pub pid: u32,
    /// Thread information
    pub thread_info: String,
    /// Environment variables relevant to GPU/Vulkan
    pub env_vars: Vec<(String, String)>,
}

/// Memory information at crash time
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Available system memory
    pub available_memory: u64,
    /// Process memory usage
    pub process_memory: u64,
    /// Memory mapping information
    pub memory_maps: Vec<String>,
}

/// GPU context information
#[derive(Debug, Clone)]
pub struct GpuContext {
    /// Vulkan SDK path
    pub vulkan_sdk: Option<String>,
    /// MoltenVK configuration
    pub moltenvk_config: Vec<(String, String)>,
    /// Available devices
    pub device_count: Option<u32>,
    /// Driver information
    pub driver_info: Vec<String>,
}

/// Crash analysis result with actionable recommendations
#[derive(Debug)]
pub struct CrashAnalysis {
    /// Likely cause of the crash
    pub likely_cause: String,
    /// Detailed technical analysis
    pub technical_details: Vec<String>,
    /// Recommended fixes
    pub recommendations: Vec<String>,
    /// Similar known issues
    pub known_issues: Vec<String>,
}

/// Install comprehensive crash handlers for debugging GPU segfaults
pub fn install_crash_handlers() -> Result<()> {
    if CRASH_HANDLER_INSTALLED.load(Ordering::Acquire) {
        return Ok(());
    }

    #[cfg(feature = "crash-debug")]
    {
        INIT.call_once(|| {
            // Install signal handlers for crash analysis
            thread::spawn(|| {
                let mut signals = Signals::new(&[SIGSEGV, SIGABRT, SIGBUS])
                    .expect("Failed to install signal handlers");

                for signal in signals.forever() {
                    let crash_info = collect_crash_info(signal);
                    let analysis = analyze_crash(&crash_info);

                    eprintln!("\n🚨 CRASH DETECTED - KOMPUTE-RS DEBUG ANALYSIS 🚨");
                    eprintln!("================================================");
                    print_crash_report(&crash_info, &analysis);

                    // Exit gracefully after analysis
                    std::process::exit(1);
                }
            });
        });
    }

    CRASH_HANDLER_INSTALLED.store(true, Ordering::Release);
    println!("🔍 Crash debugging handlers installed");
    Ok(())
}

/// Execute a potentially crashing operation with comprehensive debugging
pub fn debug_execute<T, F>(operation_name: &str, operation: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + std::panic::UnwindSafe,
{
    println!("🔍 Debug executing: {}", operation_name);
    install_crash_handlers()?;

    // Set up panic hook for additional debugging
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("\n🚨 PANIC DETECTED in debug_execute 🚨");
        eprintln!("Panic info: {:?}", panic_info);

        #[cfg(feature = "crash-debug")]
        {
            let bt = Backtrace::new();
            eprintln!("Backtrace:\n{:?}", bt);
        }

        eprintln!("Debug context: Manager constructor in MoltenVK environment");
    }));

    let result = std::panic::catch_unwind(|| operation());

    // Restore original panic hook
    std::panic::set_hook(original_hook);

    match result {
        Ok(Ok(value)) => {
            println!("✅ Debug execute successful: {}", operation_name);
            Ok(value)
        }
        Ok(Err(e)) => {
            println!(
                "❌ Debug execute failed: {} - Error: {:?}",
                operation_name, e
            );
            Err(e)
        }
        Err(panic) => {
            let panic_msg = if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic".to_string()
            };

            Err(KomputeError::internal(format!(
                "Panic in debug_execute({}): {}",
                operation_name, panic_msg
            )))
        }
    }
}

#[cfg(feature = "crash-debug")]
fn collect_crash_info(signal: i32) -> CrashInfo {
    let timestamp = Instant::now();
    let backtrace = format!("{:?}", Backtrace::new());

    CrashInfo {
        signal,
        timestamp,
        backtrace,
        process_info: collect_process_info(),
        memory_info: collect_memory_info(),
        gpu_context: collect_gpu_context(),
    }
}

#[cfg(not(feature = "crash-debug"))]
fn collect_crash_info(signal: i32) -> CrashInfo {
    CrashInfo {
        signal,
        timestamp: Instant::now(),
        backtrace: "Backtrace disabled - enable 'crash-debug' feature".to_string(),
        process_info: ProcessInfo {
            pid: std::process::id(),
            thread_info: "Thread info unavailable".to_string(),
            env_vars: vec![],
        },
        memory_info: MemoryInfo {
            available_memory: 0,
            process_memory: 0,
            memory_maps: vec!["Memory info unavailable".to_string()],
        },
        gpu_context: GpuContext {
            vulkan_sdk: None,
            moltenvk_config: vec![],
            device_count: None,
            driver_info: vec!["GPU context unavailable".to_string()],
        },
    }
}

fn collect_process_info() -> ProcessInfo {
    let pid = std::process::id();
    let thread_info = format!("Main thread - PID: {}", pid);

    let mut env_vars = Vec::new();
    for (key, value) in std::env::vars() {
        if key.contains("VULKAN") || key.contains("VK_") || key.contains("KOMPUTE") {
            env_vars.push((key, value));
        }
    }

    ProcessInfo {
        pid,
        thread_info,
        env_vars,
    }
}

fn collect_memory_info() -> MemoryInfo {
    // Basic memory information - could be expanded with platform-specific details
    MemoryInfo {
        available_memory: 0, // Would need platform-specific implementation
        process_memory: 0,   // Would need platform-specific implementation
        memory_maps: vec![
            "Memory mapping details would require platform-specific implementation".to_string(),
        ],
    }
}

fn collect_gpu_context() -> GpuContext {
    let vulkan_sdk = std::env::var("VULKAN_SDK").ok();

    let mut moltenvk_config = Vec::new();
    if let Ok(icd) = std::env::var("VK_ICD_FILENAMES") {
        moltenvk_config.push(("VK_ICD_FILENAMES".to_string(), icd));
    }
    if let Ok(layers) = std::env::var("VK_LAYER_PATH") {
        moltenvk_config.push(("VK_LAYER_PATH".to_string(), layers));
    }

    // Try to get device count safely using FFI
    let device_count = match kompute_sys::ffi::get_device_count() {
        count if count >= 0 => Some(count as u32),
        _ => None,
    };

    GpuContext {
        vulkan_sdk,
        moltenvk_config,
        device_count,
        driver_info: vec!["Driver info collection not implemented".to_string()],
    }
}

fn analyze_crash(crash_info: &CrashInfo) -> CrashAnalysis {
    let mut likely_cause = "Unknown".to_string();
    let mut technical_details = Vec::new();
    let mut recommendations = Vec::new();
    let mut known_issues = Vec::new();

    // Analyze signal type
    match crash_info.signal {
        11 => {
            // SIGSEGV
            likely_cause = "Segmentation fault in Manager constructor".to_string();
            technical_details.push(
                "SIGSEGV indicates invalid memory access during Manager creation".to_string(),
            );

            // Check for MoltenVK indicators
            if crash_info.gpu_context.vulkan_sdk.is_some() {
                if crash_info
                    .gpu_context
                    .moltenvk_config
                    .iter()
                    .any(|(_, v)| v.contains("MoltenVK"))
                {
                    likely_cause = "Kompute v0.9.0 + MoltenVK API compatibility issue".to_string();
                    technical_details.push("MoltenVK detected - known incompatibility with Kompute v0.9.0 Manager constructor".to_string());

                    recommendations.push(
                        "Use Kompute v0.8.1 or earlier for MoltenVK compatibility".to_string(),
                    );
                    recommendations.push(
                        "Set VK_APPLICATION_API_VERSION to 1.2 or lower in Kompute source"
                            .to_string(),
                    );
                    recommendations.push(
                        "Consider using external Manager creation with specific API version"
                            .to_string(),
                    );

                    known_issues.push(
                        "GitHub issue: Kompute + MoltenVK segfault on Manager constructor"
                            .to_string(),
                    );
                    known_issues.push(
                        "MoltenVK loads with API 1.3 but causes segfaults in Manager creation"
                            .to_string(),
                    );
                }
            }

            recommendations.push(
                "Enable AddressSanitizer with RUST_BACKTRACE=1 for detailed analysis".to_string(),
            );
            recommendations.push(
                "Check Vulkan validation layers for additional error information".to_string(),
            );
        }
        6 => {
            // SIGABRT
            likely_cause = "Abort signal - assertion failure or explicit abort".to_string();
            technical_details.push(
                "SIGABRT indicates program was aborted, possibly by assertion failure".to_string(),
            );
        }
        7 => {
            // SIGBUS
            likely_cause = "Bus error - misaligned memory access".to_string();
            technical_details.push(
                "SIGBUS indicates bus error, often from misaligned memory access".to_string(),
            );
        }
        _ => {
            likely_cause = format!("Unknown signal: {}", crash_info.signal);
        }
    }

    // Analyze backtrace for Kompute-specific patterns
    if crash_info.backtrace.contains("kp::Manager") {
        technical_details.push("Crash occurred in Kompute Manager constructor".to_string());
    }
    if crash_info.backtrace.contains("vulkan") || crash_info.backtrace.contains("vk") {
        technical_details.push("Vulkan API calls present in backtrace".to_string());
    }

    CrashAnalysis {
        likely_cause,
        technical_details,
        recommendations,
        known_issues,
    }
}

fn print_crash_report(crash_info: &CrashInfo, analysis: &CrashAnalysis) {
    eprintln!(
        "Signal: {} at {:?}",
        crash_info.signal, crash_info.timestamp
    );
    eprintln!("Process: PID {}", crash_info.process_info.pid);

    eprintln!("\n🔍 LIKELY CAUSE:");
    eprintln!("  {}", analysis.likely_cause);

    eprintln!("\n🔧 TECHNICAL DETAILS:");
    for detail in &analysis.technical_details {
        eprintln!("  • {}", detail);
    }

    eprintln!("\n💡 RECOMMENDATIONS:");
    for rec in &analysis.recommendations {
        eprintln!("  ✅ {}", rec);
    }

    eprintln!("\n📚 KNOWN ISSUES:");
    for issue in &analysis.known_issues {
        eprintln!("  ⚠️  {}", issue);
    }

    eprintln!("\n🌍 ENVIRONMENT:");
    for (key, value) in &crash_info.process_info.env_vars {
        eprintln!("  {}: {}", key, value);
    }

    if let Some(device_count) = crash_info.gpu_context.device_count {
        eprintln!("\n🎮 GPU CONTEXT:");
        eprintln!("  Device count: {}", device_count);
        if let Some(ref sdk) = crash_info.gpu_context.vulkan_sdk {
            eprintln!("  Vulkan SDK: {}", sdk);
        }
    }

    eprintln!("\n📊 STACK TRACE:");
    eprintln!("{}", crash_info.backtrace);

    eprintln!("\n================================================");
    eprintln!("End of crash analysis report");
}

/// Test the crash debugging system
pub fn test_crash_debug() -> Result<()> {
    println!("🧪 Testing crash debugging system...");
    install_crash_handlers()?;

    // Test with a safe operation first
    debug_execute("safe_test", || {
        println!("Safe operation completed");
        Ok(())
    })?;

    println!("✅ Crash debugging system ready");
    Ok(())
}

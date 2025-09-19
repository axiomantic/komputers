//! Cross-platform compatibility and environment detection.
//!
//! This module provides utilities for detecting platform capabilities,
//! validating environment setup, and handling platform-specific requirements
//! for GPU computing with Kompute.

use crate::error::Result;
use std::env;
use std::path::Path;

/// Platform information and capabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct PlatformInfo {
    /// Operating system name
    pub os: String,
    /// Architecture (x86_64, aarch64, etc.)
    pub arch: String,
    /// Vulkan SDK availability
    pub vulkan_sdk_available: bool,
    /// Vulkan runtime availability
    pub vulkan_runtime_available: bool,
    /// GPU driver support level
    pub gpu_support: GpuSupport,
    /// Platform-specific notes
    pub notes: Vec<String>,
}

/// GPU support levels across platforms.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuSupport {
    /// Full native GPU support (Linux/Windows with discrete GPU)
    Native,
    /// Translation layer support (macOS with MoltenVK)
    Translated,
    /// Software emulation only (CPU fallback)
    Emulated,
    /// Unknown or unsupported
    Unknown,
}

impl PlatformInfo {
    /// Detect current platform information and capabilities.
    pub fn detect() -> Self {
        let os = detect_os();
        let arch = detect_architecture();
        let vulkan_sdk_available = check_vulkan_sdk();
        let vulkan_runtime_available = check_vulkan_runtime();
        let gpu_support = detect_gpu_support(&os);
        let notes =
            generate_platform_notes(&os, &arch, vulkan_sdk_available, vulkan_runtime_available);

        Self {
            os,
            arch,
            vulkan_sdk_available,
            vulkan_runtime_available,
            gpu_support,
            notes,
        }
    }

    /// Check if the platform meets minimum requirements for GPU computing.
    pub fn meets_requirements(&self) -> bool {
        self.vulkan_runtime_available
            && matches!(
                self.gpu_support,
                GpuSupport::Native | GpuSupport::Translated
            )
    }

    /// Get platform-specific troubleshooting guidance.
    pub fn troubleshooting_guide(&self) -> Vec<String> {
        let mut guide = Vec::new();

        if !self.vulkan_sdk_available {
            match self.os.as_str() {
                "macos" => {
                    guide.push("Install Vulkan SDK from https://vulkan.lunarg.com/ or use 'brew install molten-vk'".to_string());
                    guide.push("MoltenVK is required for Vulkan support on macOS".to_string());
                }
                "linux" => {
                    guide.push("Install Vulkan development packages: apt install vulkan-tools libvulkan-dev (Ubuntu) or equivalent".to_string());
                    guide.push("Ensure your GPU drivers support Vulkan".to_string());
                }
                "windows" => {
                    guide.push("Install Vulkan SDK from https://vulkan.lunarg.com/".to_string());
                    guide.push("Update your GPU drivers to the latest version".to_string());
                }
                _ => {
                    guide.push("Install Vulkan SDK and runtime for your platform".to_string());
                }
            }
        }

        if !self.vulkan_runtime_available {
            guide.push(
                "Vulkan runtime not available - check GPU drivers and Vulkan installation"
                    .to_string(),
            );
            match self.os.as_str() {
                "macos" => {
                    guide.push("Run 'vulkaninfo' to test MoltenVK installation".to_string());
                }
                _ => {
                    guide.push("Run 'vulkaninfo' to verify Vulkan setup".to_string());
                }
            }
        }

        if matches!(self.gpu_support, GpuSupport::Unknown) {
            guide.push(
                "GPU support level unknown - verify graphics drivers are installed and up to date"
                    .to_string(),
            );
        }

        if guide.is_empty() {
            guide.push("Platform appears to be properly configured for GPU computing".to_string());
        }

        guide
    }

    /// Get recommended environment configuration for the platform.
    pub fn recommended_environment(&self) -> Vec<(String, String)> {
        let mut env_vars = Vec::new();

        match self.os.as_str() {
            "macos" => {
                if self.vulkan_sdk_available {
                    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
                        env_vars.push((
                            "VK_ICD_FILENAMES".to_string(),
                            format!("{}/share/vulkan/icd.d/MoltenVK_icd.json", vulkan_sdk),
                        ));
                        env_vars.push((
                            "VK_LAYER_PATH".to_string(),
                            format!("{}/share/vulkan/explicit_layer.d", vulkan_sdk),
                        ));
                    }
                }
                // Disable validation layers by default on macOS for better compatibility
                env_vars.push(("VK_INSTANCE_LAYERS".to_string(), "".to_string()));
            }
            "linux" => {
                // Linux typically uses system defaults, but provide guidance
                if self.vulkan_sdk_available {
                    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
                        env_vars.push((
                            "VK_LAYER_PATH".to_string(),
                            format!("{}/share/vulkan/explicit_layer.d", vulkan_sdk),
                        ));
                    }
                }
            }
            "windows" => {
                // Windows uses registry, but SDK environment can override
                if self.vulkan_sdk_available {
                    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
                        env_vars.push(("VK_LAYER_PATH".to_string(), format!("{}/Bin", vulkan_sdk)));
                    }
                }
            }
            _ => {}
        }

        env_vars
    }

    /// Format platform information for display.
    pub fn summary(&self) -> String {
        format!(
            "Platform: {} ({})\n\
            Vulkan SDK: {}\n\
            Vulkan Runtime: {}\n\
            GPU Support: {:?}\n\
            Requirements Met: {}",
            self.os,
            self.arch,
            if self.vulkan_sdk_available {
                "Available"
            } else {
                "Not Found"
            },
            if self.vulkan_runtime_available {
                "Available"
            } else {
                "Not Found"
            },
            self.gpu_support,
            if self.meets_requirements() {
                "Yes"
            } else {
                "No"
            }
        )
    }
}

/// Environment validation utilities.
pub struct EnvironmentValidator;

impl EnvironmentValidator {
    /// Perform comprehensive environment validation.
    pub fn validate() -> Result<ValidationReport> {
        let platform = PlatformInfo::detect();
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        // Check core requirements
        if !platform.vulkan_runtime_available {
            issues.push("Vulkan runtime not available".to_string());
        }

        if !platform.vulkan_sdk_available {
            warnings.push("Vulkan SDK not found - development features may be limited".to_string());
        }

        if matches!(platform.gpu_support, GpuSupport::Unknown) {
            warnings.push("GPU support level unknown - performance may be limited".to_string());
        }

        // Platform-specific checks
        match platform.os.as_str() {
            "macos" => {
                if !Self::check_moltenvk() {
                    issues.push("MoltenVK not properly configured on macOS".to_string());
                }
            }
            "linux" => {
                if !Self::check_gpu_drivers_linux() {
                    warnings.push("GPU drivers may not be properly installed".to_string());
                }
            }
            "windows" => {
                if !Self::check_gpu_drivers_windows() {
                    warnings.push("GPU drivers may be outdated".to_string());
                }
            }
            _ => {
                warnings.push("Untested platform - compatibility not guaranteed".to_string());
            }
        }

        // Generate recommendations
        if !issues.is_empty() || !warnings.is_empty() {
            recommendations.extend(platform.troubleshooting_guide());
        }

        Ok(ValidationReport {
            platform,
            issues,
            warnings,
            recommendations,
        })
    }

    fn check_moltenvk() -> bool {
        // Check for MoltenVK presence
        let moltenvk_paths = [
            "/usr/local/share/vulkan/icd.d/MoltenVK_icd.json",
            "/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json",
        ];

        if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
            let sdk_moltenvk = format!("{}/share/vulkan/icd.d/MoltenVK_icd.json", vulkan_sdk);
            if Path::new(&sdk_moltenvk).exists() {
                return true;
            }
        }

        moltenvk_paths.iter().any(|path| Path::new(path).exists())
    }

    fn check_gpu_drivers_linux() -> bool {
        // Basic check for GPU driver presence
        Path::new("/dev/dri").exists() || Path::new("/proc/driver/nvidia").exists()
    }

    fn check_gpu_drivers_windows() -> bool {
        // On Windows, assume drivers are present if Vulkan runtime is available
        // More sophisticated checks would require Windows-specific APIs
        true
    }
}

/// Environment validation report.
#[derive(Debug)]
pub struct ValidationReport {
    /// Platform information
    pub platform: PlatformInfo,
    /// Critical issues that prevent operation
    pub issues: Vec<String>,
    /// Non-critical warnings
    pub warnings: Vec<String>,
    /// Recommended actions to resolve issues
    pub recommendations: Vec<String>,
}

impl ValidationReport {
    /// Check if the environment is suitable for GPU computing.
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty() && self.platform.meets_requirements()
    }

    /// Get a user-friendly summary of the validation results.
    pub fn summary(&self) -> String {
        let mut summary = format!("Environment Validation Report\n{}\n\n", "=".repeat(40));
        summary.push_str(&self.platform.summary());
        summary.push_str("\n\n");

        if !self.issues.is_empty() {
            summary.push_str("Issues:\n");
            for issue in &self.issues {
                summary.push_str(&format!("  ❌ {}\n", issue));
            }
            summary.push('\n');
        }

        if !self.warnings.is_empty() {
            summary.push_str("Warnings:\n");
            for warning in &self.warnings {
                summary.push_str(&format!("  ⚠️  {}\n", warning));
            }
            summary.push('\n');
        }

        if !self.recommendations.is_empty() {
            summary.push_str("Recommendations:\n");
            for rec in &self.recommendations {
                summary.push_str(&format!("  💡 {}\n", rec));
            }
        }

        if self.is_valid() {
            summary.push_str("\n✅ Environment is ready for GPU computing!\n");
        } else {
            summary.push_str("\n❌ Environment requires attention before GPU computing.\n");
        }

        summary
    }
}

// Helper functions

fn detect_os() -> String {
    env::consts::OS.to_string()
}

fn detect_architecture() -> String {
    env::consts::ARCH.to_string()
}

fn check_vulkan_sdk() -> bool {
    env::var("VULKAN_SDK").is_ok()
}

fn check_vulkan_runtime() -> bool {
    // Try to check if Vulkan is available through device count
    match crate::Manager::device_count() {
        Ok(count) => count > 0,
        Err(_) => false,
    }
}

fn detect_gpu_support(os: &str) -> GpuSupport {
    match os {
        "macos" => GpuSupport::Translated, // MoltenVK translation layer
        "linux" | "windows" => GpuSupport::Native, // Native Vulkan support
        _ => GpuSupport::Unknown,
    }
}

fn generate_platform_notes(
    os: &str,
    arch: &str,
    vulkan_sdk: bool,
    vulkan_runtime: bool,
) -> Vec<String> {
    let mut notes = Vec::new();

    match os {
        "macos" => {
            notes.push("macOS uses MoltenVK for Vulkan compatibility".to_string());
            if arch == "aarch64" {
                notes.push("Apple Silicon (M1/M2) support via MoltenVK".to_string());
            }
        }
        "linux" => {
            notes.push("Native Vulkan support available".to_string());
            if !vulkan_runtime {
                notes.push("Install mesa-vulkan-drivers or proprietary GPU drivers".to_string());
            }
        }
        "windows" => {
            notes.push("Native Vulkan support available".to_string());
            notes.push("Ensure GPU drivers are up to date".to_string());
        }
        _ => {
            notes.push("Platform support not verified - experimental".to_string());
        }
    }

    if !vulkan_sdk {
        notes.push("Vulkan SDK recommended for development".to_string());
    }

    notes
}

/// Cross-platform utilities for runtime environment management.
pub struct RuntimeUtils;

impl RuntimeUtils {
    /// Apply platform-specific optimizations and configurations.
    pub fn optimize_for_platform() -> Result<()> {
        let platform = PlatformInfo::detect();

        match platform.os.as_str() {
            "macos" => Self::optimize_macos(),
            "linux" => Self::optimize_linux(),
            "windows" => Self::optimize_windows(),
            _ => Ok(()),
        }
    }

    fn optimize_macos() -> Result<()> {
        // Configure MoltenVK optimizations
        env::set_var("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "1");
        env::set_var("MVK_CONFIG_PERFORMANCE_TRACKING", "0");
        Ok(())
    }

    fn optimize_linux() -> Result<()> {
        // Linux-specific GPU optimizations
        // Most optimizations happen at driver level
        Ok(())
    }

    fn optimize_windows() -> Result<()> {
        // Windows-specific optimizations
        Ok(())
    }

    /// Get recommended memory allocation limits for the platform.
    pub fn memory_limits() -> (usize, usize) {
        let platform = PlatformInfo::detect();

        match platform.os.as_str() {
            "macos" => {
                // Conservative limits for MoltenVK
                (512 * 1024 * 1024, 2 * 1024 * 1024 * 1024) // 512MB - 2GB
            }
            "linux" | "windows" => {
                // More aggressive limits for native Vulkan
                (1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024) // 1GB - 8GB
            }
            _ => {
                // Conservative defaults
                (256 * 1024 * 1024, 1024 * 1024 * 1024) // 256MB - 1GB
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = PlatformInfo::detect();
        assert!(!platform.os.is_empty());
        assert!(!platform.arch.is_empty());
        println!("Detected platform: {} ({})", platform.os, platform.arch);
    }

    #[test]
    fn test_platform_summary() {
        let platform = PlatformInfo::detect();
        let summary = platform.summary();
        assert!(!summary.is_empty());
        println!("Platform summary:\n{}", summary);
    }

    #[test]
    fn test_environment_validation() {
        match EnvironmentValidator::validate() {
            Ok(report) => {
                println!("Validation report:\n{}", report.summary());
                // Don't assert validity since this depends on the test environment
            }
            Err(e) => {
                println!("Validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_troubleshooting_guide() {
        let platform = PlatformInfo::detect();
        let guide = platform.troubleshooting_guide();
        assert!(!guide.is_empty());

        for item in guide {
            println!("💡 {}", item);
        }
    }

    #[test]
    fn test_memory_limits() {
        let (min, max) = RuntimeUtils::memory_limits();
        assert!(min < max);
        assert!(min > 0);
        println!("Memory limits: {} - {} bytes", min, max);
    }

    #[test]
    fn test_platform_optimizations() {
        match RuntimeUtils::optimize_for_platform() {
            Ok(()) => println!("✓ Platform optimizations applied"),
            Err(e) => println!("⚠️ Platform optimization failed: {}", e),
        }
    }
}

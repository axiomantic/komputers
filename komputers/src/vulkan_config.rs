//! Explicit Vulkan configuration for reliable cross-platform GPU setup.
//!
//! This module provides explicit configuration structures that eliminate
//! environment variable synchronization issues in multi-threaded contexts.

use std::path::PathBuf;

/// Comprehensive Vulkan configuration for Manager creation.
///
/// This structure contains all the settings needed to configure Vulkan
/// properly across different platforms and execution contexts. It replaces
/// the previous environment variable-based approach with explicit parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VulkanConfig {
    /// Whether to use MoltenVK for Vulkan-to-Metal translation (macOS)
    pub use_moltenvk: bool,

    /// Whether to enable Vulkan validation layers for debugging
    pub use_validation: bool,

    /// Whether to use Vulkan portability extensions (required for MoltenVK)
    pub use_portability_extension: bool,

    /// Path to the Vulkan ICD (Installable Client Driver) file
    pub icd_path: Option<PathBuf>,

    /// Paths to Vulkan validation layer files
    pub layer_paths: Vec<PathBuf>,

    /// Platform-specific reasoning for configuration choices
    pub configuration_reasoning: String,

    /// Warnings or notes about the configuration
    pub warnings: Vec<String>,
}

impl VulkanConfig {
    /// Create a new VulkanConfig with default settings.
    ///
    /// This provides a baseline configuration that disables all optional
    /// features. Use `auto_detect()` for intelligent platform detection.
    pub fn new() -> Self {
        Self {
            use_moltenvk: false,
            use_validation: false,
            use_portability_extension: false,
            icd_path: None,
            layer_paths: Vec::new(),
            configuration_reasoning: "Default configuration with minimal features".to_string(),
            warnings: Vec::new(),
        }
    }

    /// Auto-detect optimal Vulkan configuration for the current platform.
    ///
    /// This performs intelligent platform detection and configuration,
    /// including:
    /// - Apple Silicon vs Intel Mac detection
    /// - MoltenVK availability and configuration
    /// - Validation layer detection
    /// - GPU type classification (integrated vs external)
    ///
    /// # Errors
    ///
    /// Returns an error if platform detection fails or required components
    /// are missing.
    pub fn auto_detect() -> Result<Self, String> {
        use crate::platform::MetalGpuDetection;
        use std::env;

        let mut config = Self::new();
        let mut reasoning = String::new();

        // Check for user overrides first
        let force_moltenvk = env::var("KOMPUTE_FORCE_MOLTENVK")
            .map(|v| v == "1")
            .unwrap_or(false);
        let disable_moltenvk = env::var("KOMPUTE_DISABLE_MOLTENVK")
            .map(|v| v == "1")
            .unwrap_or(false);
        let force_validation = env::var("KOMPUTE_FORCE_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let disable_validation = env::var("KOMPUTE_DISABLE_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let verbose_detection = env::var("KOMPUTE_VERBOSE_DETECTION")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Handle conflicting flags
        if force_moltenvk && disable_moltenvk {
            config.warnings.push("⚠️ Conflicting flags: KOMPUTE_FORCE_MOLTENVK and KOMPUTE_DISABLE_MOLTENVK both set".to_string());
        }
        if force_validation && disable_validation {
            config.warnings.push("⚠️ Conflicting flags: KOMPUTE_FORCE_VALIDATION and KOMPUTE_DISABLE_VALIDATION both set".to_string());
        }

        // Platform detection and MoltenVK configuration
        if force_moltenvk {
            config.use_moltenvk = true;
            config.use_portability_extension = true;
            reasoning.push_str("[FORCED] ");

            // Try to find MoltenVK ICD path
            config.icd_path = Self::find_moltenvk_icd_path();
            if config.icd_path.is_none() {
                config
                    .warnings
                    .push("⚠️ MoltenVK forced but ICD file not found".to_string());
            }
        } else if disable_moltenvk {
            config.use_moltenvk = false;
            reasoning.push_str("[DISABLED] ");
        } else {
            // Intelligent platform-based decision
            #[cfg(target_os = "macos")]
            {
                if MetalGpuDetection::should_auto_configure_moltenvk() {
                    match Self::detect_macos_gpu_config() {
                        Ok((should_use_moltenvk, gpu_reasoning, icd_path)) => {
                            config.use_moltenvk = should_use_moltenvk;
                            config.use_portability_extension = should_use_moltenvk;
                            config.icd_path = icd_path;
                            reasoning.push_str(&gpu_reasoning);
                        }
                        Err(e) => {
                            config
                                .warnings
                                .push(format!("⚠️ GPU detection failed: {}", e));
                            reasoning.push_str("[GPU_DETECTION_FAILED] ");
                        }
                    }
                } else {
                    reasoning.push_str("[AUTO_MOLTENVK_DISABLED] ");
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                config.use_moltenvk = false;
                reasoning.push_str("[NON_MACOS] ");
            }
        }

        // Validation layer configuration
        if force_validation {
            config.use_validation = true;
            reasoning.push_str("[VALIDATION_FORCED] ");
            config.layer_paths = Self::find_validation_layer_paths();
            if config.layer_paths.is_empty() {
                config
                    .warnings
                    .push("❌ Validation layers forced but not found".to_string());
            }
        } else if disable_validation {
            config.use_validation = false;
            reasoning.push_str("[VALIDATION_DISABLED] ");
        } else {
            // Intelligent validation decision - conservative in CI
            let is_ci = env::var("CI").is_ok() || env::var("GITHUB_ACTIONS").is_ok();
            if is_ci {
                config.use_validation = false;
                reasoning.push_str("[CI_CONSERVATIVE] ");
            } else {
                let validation_paths = Self::find_validation_layer_paths();
                if !validation_paths.is_empty() {
                    config.use_validation = true;
                    config.layer_paths = validation_paths;
                    reasoning.push_str("[VALIDATION_AVAILABLE] ");
                } else {
                    config.use_validation = false;
                    reasoning.push_str("[NO_VALIDATION] ");
                    config.warnings.push("ℹ️ Validation layers not available - install via 'brew install vulkan-validationlayers'".to_string());
                }
            }
        }

        config.configuration_reasoning = reasoning;

        // Validate the resulting configuration
        if let Err(validation_error) = config.validate() {
            return Err(format!(
                "Auto-detected configuration is invalid: {}",
                validation_error
            ));
        }

        if verbose_detection {
            eprintln!("🔍 Auto-detected Vulkan Configuration:");
            eprintln!("{}", config.summary());
        }

        Ok(config)
    }

    /// Create a VulkanConfig optimized for testing environments.
    ///
    /// This configuration prioritizes reliability and speed over feature
    /// completeness, suitable for CI environments.
    pub fn for_testing() -> Self {
        Self {
            use_moltenvk: false,
            use_validation: false,
            use_portability_extension: false,
            icd_path: None,
            layer_paths: Vec::new(),
            configuration_reasoning: "Minimal configuration for testing".to_string(),
            warnings: vec!["Testing configuration - reduced feature set".to_string()],
        }
    }

    /// Create a VulkanConfig with MoltenVK explicitly enabled.
    ///
    /// # Arguments
    ///
    /// * `icd_path` - Path to the MoltenVK ICD file
    ///
    /// # Examples
    ///
    /// ```
    /// use komputers::VulkanConfig;
    ///
    /// let config = VulkanConfig::with_moltenvk(
    ///     "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
    /// );
    /// ```
    pub fn with_moltenvk<P: Into<PathBuf>>(icd_path: P) -> Self {
        Self {
            use_moltenvk: true,
            use_validation: false,
            use_portability_extension: true, // Required for MoltenVK
            icd_path: Some(icd_path.into()),
            layer_paths: Vec::new(),
            configuration_reasoning: "Explicit MoltenVK configuration".to_string(),
            warnings: Vec::new(),
        }
    }

    /// Enable validation layers for debugging.
    ///
    /// # Arguments
    ///
    /// * `layer_paths` - Paths to validation layer files
    pub fn with_validation(mut self, layer_paths: Vec<PathBuf>) -> Self {
        self.use_validation = true;
        self.layer_paths = layer_paths;
        self.configuration_reasoning
            .push_str(" + validation layers");
        self
    }

    /// Add a warning message to the configuration.
    pub fn with_warning<S: Into<String>>(mut self, warning: S) -> Self {
        self.warnings.push(warning.into());
        self
    }

    /// Validate that the configuration is usable.
    ///
    /// This checks that required files exist and the configuration
    /// is internally consistent.
    pub fn validate(&self) -> Result<(), String> {
        // Check MoltenVK ICD path exists if specified
        if self.use_moltenvk {
            if let Some(ref icd_path) = self.icd_path {
                if !icd_path.exists() {
                    return Err(format!(
                        "MoltenVK ICD file not found: {}",
                        icd_path.display()
                    ));
                }
            } else {
                return Err("MoltenVK enabled but no ICD path specified".to_string());
            }

            // MoltenVK requires portability extension
            if !self.use_portability_extension {
                return Err("MoltenVK requires portability extension to be enabled".to_string());
            }
        }

        // Check validation layer paths exist if specified
        if self.use_validation {
            for layer_path in &self.layer_paths {
                if !layer_path.exists() {
                    return Err(format!(
                        "Validation layer file not found: {}",
                        layer_path.display()
                    ));
                }
            }
        }

        Ok(())
    }

    /// Get a summary of the configuration for logging.
    pub fn summary(&self) -> String {
        let mut summary = format!("Vulkan Configuration: {}", self.configuration_reasoning);

        if self.use_moltenvk {
            summary.push_str(&format!(
                "\n  MoltenVK: enabled ({})",
                self.icd_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "no path".to_string())
            ));
        } else {
            summary.push_str("\n  MoltenVK: disabled");
        }

        if self.use_validation {
            summary.push_str(&format!(
                "\n  Validation: enabled ({} layers)",
                self.layer_paths.len()
            ));
        } else {
            summary.push_str("\n  Validation: disabled");
        }

        if self.use_portability_extension {
            summary.push_str("\n  Portability: enabled");
        }

        for warning in &self.warnings {
            summary.push_str(&format!("\n  ⚠️ {}", warning));
        }

        summary
    }

    // Helper methods for auto-detection

    /// Find MoltenVK ICD file path on the system.
    fn find_moltenvk_icd_path() -> Option<PathBuf> {
        let candidates = [
            "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
            "/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json",
        ];

        for path in &candidates {
            let path_buf = PathBuf::from(path);
            if path_buf.exists() {
                return Some(path_buf);
            }
        }

        // Check VULKAN_SDK environment variable
        if let Ok(vulkan_sdk) = std::env::var("VULKAN_SDK") {
            let sdk_icd_path = PathBuf::from(vulkan_sdk).join("etc/vulkan/icd.d/MoltenVK_icd.json");
            if sdk_icd_path.exists() {
                return Some(sdk_icd_path);
            }
        }

        None
    }

    /// Find validation layer paths on the system.
    fn find_validation_layer_paths() -> Vec<PathBuf> {
        let candidates = [
            "/opt/homebrew/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json",
            "/usr/local/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json",
        ];

        let mut found_paths = Vec::new();

        for path in &candidates {
            let path_buf = PathBuf::from(path);
            if path_buf.exists() {
                found_paths.push(path_buf);
            }
        }

        // Check VULKAN_SDK environment variable
        if let Ok(vulkan_sdk) = std::env::var("VULKAN_SDK") {
            let sdk_layer_path = PathBuf::from(vulkan_sdk)
                .join("share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json");
            if sdk_layer_path.exists() && !found_paths.contains(&sdk_layer_path) {
                found_paths.push(sdk_layer_path);
            }
        }

        found_paths
    }

    /// Detect macOS GPU configuration and decide on MoltenVK usage.
    #[cfg(target_os = "macos")]
    fn detect_macos_gpu_config() -> Result<(bool, String, Option<PathBuf>), String> {
        // Check CPU architecture
        let is_apple_silicon = Self::is_apple_silicon()?;

        // Detect GPU type
        let gpu_info = Self::detect_macos_gpu_type()?;

        let (should_use_moltenvk, reasoning) = match gpu_info.as_str() {
            gpu if gpu.contains("Apple")
                || gpu.contains("M1")
                || gpu.contains("M2")
                || gpu.contains("M3")
                || gpu.contains("M4") =>
            {
                (true, "[APPLE_SILICON] ".to_string())
            }
            gpu if gpu.contains("Intel") => (true, "[INTEL_INTEGRATED] ".to_string()),
            gpu if gpu.contains("NVIDIA") => (false, "[EXTERNAL_NVIDIA] ".to_string()),
            gpu if gpu.contains("AMD") || gpu.contains("Radeon") => {
                (false, "[EXTERNAL_AMD] ".to_string())
            }
            _ => {
                if is_apple_silicon {
                    (true, "[APPLE_SILICON_FALLBACK] ".to_string())
                } else {
                    (true, "[INTEL_MAC_FALLBACK] ".to_string())
                }
            }
        };

        let icd_path = if should_use_moltenvk {
            Self::find_moltenvk_icd_path()
        } else {
            None
        };

        Ok((should_use_moltenvk, reasoning, icd_path))
    }

    /// Check if running on Apple Silicon.
    #[cfg(target_os = "macos")]
    fn is_apple_silicon() -> Result<bool, String> {
        use std::process::Command;

        let output = Command::new("uname")
            .arg("-m")
            .output()
            .map_err(|e| format!("Failed to run uname: {}", e))?;

        let arch = String::from_utf8_lossy(&output.stdout);
        Ok(arch.trim().contains("arm"))
    }

    /// Detect GPU type on macOS using system_profiler.
    #[cfg(target_os = "macos")]
    fn detect_macos_gpu_type() -> Result<String, String> {
        use std::process::Command;

        let output = Command::new("system_profiler")
            .args(["SPDisplaysDataType"])
            .output()
            .map_err(|e| format!("Failed to run system_profiler: {}", e))?;

        let profiler_output = String::from_utf8_lossy(&output.stdout);

        // Extract GPU name from the output
        for line in profiler_output.lines() {
            if line.trim_start().starts_with("Chipset Model:") {
                if let Some(gpu_name) = line.split(':').nth(1) {
                    return Ok(gpu_name.trim().to_string());
                }
            }
        }

        // Fallback if we can't parse the output
        if Self::is_apple_silicon()? {
            Ok("Apple Silicon GPU (detected via uname)".to_string())
        } else {
            Ok("Unknown GPU (Intel Mac)".to_string())
        }
    }
}

impl Default for VulkanConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VulkanConfig::new();
        assert!(!config.use_moltenvk);
        assert!(!config.use_validation);
        assert!(!config.use_portability_extension);
        assert!(config.icd_path.is_none());
        assert!(config.layer_paths.is_empty());
    }

    #[test]
    fn test_moltenvk_config() {
        let config = VulkanConfig::with_moltenvk("/path/to/moltenvk.json");
        assert!(config.use_moltenvk);
        assert!(config.use_portability_extension); // Should be auto-enabled
        assert_eq!(
            config.icd_path,
            Some(PathBuf::from("/path/to/moltenvk.json"))
        );
    }

    #[test]
    fn test_validation_config() {
        let config =
            VulkanConfig::new().with_validation(vec![PathBuf::from("/path/to/validation.json")]);
        assert!(config.use_validation);
        assert_eq!(config.layer_paths.len(), 1);
    }

    #[test]
    fn test_config_chaining() {
        let config = VulkanConfig::with_moltenvk("/path/to/moltenvk.json")
            .with_validation(vec![PathBuf::from("/path/to/validation.json")])
            .with_warning("Test warning");

        assert!(config.use_moltenvk);
        assert!(config.use_validation);
        assert_eq!(config.warnings.len(), 1);
    }

    #[test]
    fn test_validation_requires_moltenvk_consistency() {
        let mut config = VulkanConfig::new();
        config.use_moltenvk = true;
        // Missing ICD path should fail validation
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_testing_config() {
        let config = VulkanConfig::for_testing();
        assert!(!config.use_moltenvk);
        assert!(!config.use_validation);
        assert!(!config.warnings.is_empty());
    }
}

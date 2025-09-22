#pragma once

#include <string>
#include <vector>
#include <optional>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/utsname.h>
#endif

// =============================================================================
// Platform Detection and GPU Configuration System
// =============================================================================

namespace platform_detection {

// Platform configuration flags (can be overridden by environment variables)
struct PlatformConfig {
    bool force_moltenvk = false;        // KOMPUTE_FORCE_MOLTENVK=1
    bool disable_moltenvk = false;      // KOMPUTE_DISABLE_MOLTENVK=1
    bool force_validation = false;      // KOMPUTE_FORCE_VALIDATION=1
    bool disable_validation = false;    // KOMPUTE_DISABLE_VALIDATION=1
    bool verbose_detection = false;     // KOMPUTE_VERBOSE_DETECTION=1
};

// GPU type classification
enum class GPUType {
    Unknown,
    AppleSilicon,      // Apple M1/M2/M3 integrated GPU
    IntelIntegrated,   // Intel integrated graphics on Mac
    ExternalNvidia,    // NVIDIA GPU (Thunderbolt/PCIe)
    ExternalAMD,       // AMD GPU (Thunderbolt/PCIe)
    ExternalOther      // Other external GPU
};

// Platform information structure
struct PlatformInfo {
    bool is_apple = false;
    bool is_arm64 = false;
    bool is_x86_64 = false;
    bool has_metal_support = false;
    bool has_moltenvk_available = false;
    bool has_validation_layers = false;
    GPUType primary_gpu_type = GPUType::Unknown;
    std::vector<std::string> available_gpus;
    std::string platform_description;
};

// Decision structure for MoltenVK and validation
struct PlatformDecision {
    bool use_moltenvk = false;
    bool use_validation = false;
    bool use_portability_extension = false;
    std::string reasoning;
    std::vector<std::string> warnings;
};

// =============================================================================
// Detection Functions
// =============================================================================

// Get environment variable with default value
inline std::string get_env_var(const std::string& name, const std::string& default_value = "") {
    const char* value = std::getenv(name.c_str());
    return value ? std::string(value) : default_value;
}

// Check if environment variable is set to "1" or "true"
inline bool is_env_enabled(const std::string& name) {
    std::string value = get_env_var(name);
    return value == "1" || value == "true" || value == "True" || value == "TRUE";
}

// Load platform configuration from environment variables
inline PlatformConfig load_platform_config() {
    PlatformConfig config;
    config.force_moltenvk = is_env_enabled("KOMPUTE_FORCE_MOLTENVK");
    config.disable_moltenvk = is_env_enabled("KOMPUTE_DISABLE_MOLTENVK");
    config.force_validation = is_env_enabled("KOMPUTE_FORCE_VALIDATION");
    config.disable_validation = is_env_enabled("KOMPUTE_DISABLE_VALIDATION");
    config.verbose_detection = is_env_enabled("KOMPUTE_VERBOSE_DETECTION");
    return config;
}

#ifdef __APPLE__

// Detect CPU architecture on macOS
inline bool is_apple_silicon() {
    struct utsname sysinfo;
    if (uname(&sysinfo) == 0) {
        std::string machine(sysinfo.machine);
        return machine.find("arm") != std::string::npos;
    }
    return false;
}

// Simple Metal detection using system profiler (no Objective-C needed)
inline bool check_metal_support() {
    // Check if we're on Apple Silicon (which always has Metal)
    if (is_apple_silicon()) {
        return true;
    }

    // For Intel Macs, check for discrete GPU via system profiler
    FILE* pipe = popen("system_profiler SPDisplaysDataType 2>/dev/null | grep -i 'chipset model' | head -1", "r");
    if (pipe) {
        char buffer[256];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        pclose(pipe);
        return !result.empty();
    }
    return false;
}

// Detect GPU types using system profiler (no Objective-C needed)
inline std::vector<std::string> detect_macos_gpus() {
    std::vector<std::string> gpus;

    FILE* pipe = popen("system_profiler SPDisplaysDataType 2>/dev/null | grep -i 'chipset model' | awk -F': ' '{print $2}' | sed 's/^ *//'", "r");
    if (pipe) {
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            std::string gpu(buffer);
            // Remove trailing newline
            if (!gpu.empty() && gpu.back() == '\n') {
                gpu.pop_back();
            }
            if (!gpu.empty()) {
                gpus.push_back(gpu);
            }
        }
        pclose(pipe);
    }

    // Fallback detection based on architecture
    if (gpus.empty()) {
        if (is_apple_silicon()) {
            gpus.push_back("Apple Silicon GPU (detected via uname)");
        } else {
            gpus.push_back("Unknown GPU (Intel Mac)");
        }
    }

    return gpus;
}

// Classify primary GPU type based on name
inline GPUType classify_primary_gpu(const std::vector<std::string>& gpus) {
    if (gpus.empty()) return GPUType::Unknown;

    std::string primary = gpus[0];

    // Apple Silicon detection
    if (primary.find("Apple") != std::string::npos ||
        primary.find("M1") != std::string::npos ||
        primary.find("M2") != std::string::npos ||
        primary.find("M3") != std::string::npos ||
        primary.find("M4") != std::string::npos) {
        return GPUType::AppleSilicon;
    }

    // Intel integrated graphics
    if (primary.find("Intel") != std::string::npos) {
        return GPUType::IntelIntegrated;
    }

    // NVIDIA external
    if (primary.find("NVIDIA") != std::string::npos ||
        primary.find("GeForce") != std::string::npos ||
        primary.find("Quadro") != std::string::npos ||
        primary.find("Tesla") != std::string::npos) {
        return GPUType::ExternalNvidia;
    }

    // AMD external
    if (primary.find("AMD") != std::string::npos ||
        primary.find("Radeon") != std::string::npos ||
        primary.find("FirePro") != std::string::npos) {
        return GPUType::ExternalAMD;
    }

    return GPUType::ExternalOther;
}

#endif // __APPLE__

// Check if MoltenVK is available
inline bool check_moltenvk_available() {
    // Check common MoltenVK installation paths
    std::vector<std::string> moltenvk_paths = {
        "/opt/homebrew/lib/libMoltenVK.dylib",
        "/usr/local/lib/libMoltenVK.dylib",
        "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
        "/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json"
    };

    for (const auto& path : moltenvk_paths) {
        if (access(path.c_str(), R_OK) == 0) {
            return true;
        }
    }

    // Check environment variables
    const char* vulkan_sdk = std::getenv("VULKAN_SDK");
    if (vulkan_sdk) {
        std::string icd_path = std::string(vulkan_sdk) + "/etc/vulkan/icd.d/MoltenVK_icd.json";
        if (access(icd_path.c_str(), R_OK) == 0) {
            return true;
        }
    }

    return false;
}

// Check if Vulkan validation layers are available AND actually work at runtime
inline bool check_validation_layers_available() {
    // In CI environments, be conservative and don't use validation layers
    if (std::getenv("CI") || std::getenv("GITHUB_ACTIONS")) {
        return false;
    }

    // Check for validation layer shared libraries
    std::vector<std::string> validation_paths = {
        "/opt/homebrew/lib/libVkLayer_khronos_validation.dylib",
        "/usr/local/lib/libVkLayer_khronos_validation.dylib",
        "/opt/homebrew/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json",
        "/usr/local/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json"
    };

    bool files_exist = false;
    for (const auto& path : validation_paths) {
        if (access(path.c_str(), R_OK) == 0) {
            files_exist = true;
            break;
        }
    }

    if (!files_exist) {
        // Check VULKAN_SDK path
        const char* vulkan_sdk = std::getenv("VULKAN_SDK");
        if (vulkan_sdk) {
            std::string layer_path = std::string(vulkan_sdk) + "/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json";
            if (access(layer_path.c_str(), R_OK) == 0) {
                files_exist = true;
            }
        }
    }

    return files_exist;
}

// Main platform detection function
inline PlatformInfo detect_platform() {
    PlatformInfo info;

#ifdef __APPLE__
    info.is_apple = true;
    info.is_arm64 = is_apple_silicon();
    info.is_x86_64 = !info.is_arm64;
    info.has_metal_support = check_metal_support();
    info.available_gpus = detect_macos_gpus();
    info.primary_gpu_type = classify_primary_gpu(info.available_gpus);

    // Build platform description
    info.platform_description = "macOS";
    if (info.is_arm64) {
        info.platform_description += " (Apple Silicon)";
    } else {
        info.platform_description += " (Intel)";
    }

    if (!info.available_gpus.empty()) {
        info.platform_description += " with " + info.available_gpus[0];
    }
#else
    info.platform_description = "Non-macOS platform";
#endif

    info.has_moltenvk_available = check_moltenvk_available();
    info.has_validation_layers = check_validation_layers_available();

    return info;
}

// Generate GPU-specific Vulkan configuration decisions
inline PlatformDecision make_platform_decision(const PlatformInfo& info, const PlatformConfig& config) {
    PlatformDecision decision;

    // Handle explicit user overrides first
    if (config.force_moltenvk && config.disable_moltenvk) {
        decision.warnings.push_back("⚠️  Conflicting flags: KOMPUTE_FORCE_MOLTENVK and KOMPUTE_DISABLE_MOLTENVK both set");
    }

    if (config.force_validation && config.disable_validation) {
        decision.warnings.push_back("⚠️  Conflicting flags: KOMPUTE_FORCE_VALIDATION and KOMPUTE_DISABLE_VALIDATION both set");
    }

    // MoltenVK decision logic
    if (config.force_moltenvk) {
        decision.use_moltenvk = true;
        decision.reasoning += "[FORCED] ";
        if (!info.has_moltenvk_available) {
            decision.warnings.push_back("⚠️  MoltenVK forced but not detected on system");
        }
    } else if (config.disable_moltenvk) {
        decision.use_moltenvk = false;
        decision.reasoning += "[DISABLED] ";
    } else if (!info.is_apple) {
        decision.use_moltenvk = false;
        decision.reasoning += "[NON-APPLE] ";
    } else {
        // Intelligent decision for Apple platforms
        switch (info.primary_gpu_type) {
            case GPUType::AppleSilicon:
                decision.use_moltenvk = true;
                decision.reasoning += "[APPLE_SILICON] ";
                break;
            case GPUType::IntelIntegrated:
                decision.use_moltenvk = true;
                decision.reasoning += "[INTEL_INTEGRATED] ";
                break;
            case GPUType::ExternalNvidia:
                decision.use_moltenvk = false;
                decision.reasoning += "[EXTERNAL_NVIDIA] ";
                decision.warnings.push_back("🎯 Detected external NVIDIA GPU - using native drivers instead of MoltenVK");
                break;
            case GPUType::ExternalAMD:
                decision.use_moltenvk = false;
                decision.reasoning += "[EXTERNAL_AMD] ";
                decision.warnings.push_back("🎯 Detected external AMD GPU - using native drivers instead of MoltenVK");
                break;
            case GPUType::ExternalOther:
                decision.use_moltenvk = false;
                decision.reasoning += "[EXTERNAL_OTHER] ";
                decision.warnings.push_back("🎯 Detected external GPU - using native drivers instead of MoltenVK");
                break;
            default:
                // Default to MoltenVK on Apple if we can't classify
                decision.use_moltenvk = info.has_moltenvk_available;
                decision.reasoning += "[DEFAULT_APPLE] ";
                break;
        }

        if (decision.use_moltenvk && !info.has_moltenvk_available) {
            decision.use_moltenvk = false;
            decision.warnings.push_back("❌ MoltenVK required but not available - install via 'brew install molten-vk'");
        }
    }

    // Portability extension decision (required for MoltenVK)
    decision.use_portability_extension = decision.use_moltenvk;

    // Validation layer decision logic
    if (config.force_validation) {
        decision.use_validation = true;
        decision.reasoning += "[VALIDATION_FORCED] ";
        if (!info.has_validation_layers) {
            decision.warnings.push_back("❌ Validation layers forced but not available");
        }
    } else if (config.disable_validation) {
        decision.use_validation = false;
        decision.reasoning += "[VALIDATION_DISABLED] ";
    } else {
        // Intelligent validation decision
        decision.use_validation = info.has_validation_layers;
        if (info.has_validation_layers) {
            decision.reasoning += "[VALIDATION_AVAILABLE] ";
        } else {
            decision.reasoning += "[NO_VALIDATION] ";
            decision.warnings.push_back("ℹ️  Validation layers not available - install via 'brew install vulkan-validationlayers'");
        }
    }

    return decision;
}

// Print platform detection results (if verbose mode enabled)
inline void print_platform_info(const PlatformInfo& info, const PlatformDecision& decision, bool verbose) {
    if (!verbose) return;

    std::cerr << "🔍 Platform Detection Results:" << std::endl;
    std::cerr << "  Platform: " << info.platform_description << std::endl;
    std::cerr << "  Architecture: " << (info.is_arm64 ? "ARM64" : "x86_64") << std::endl;
    std::cerr << "  Metal Support: " << (info.has_metal_support ? "Yes" : "No") << std::endl;
    std::cerr << "  MoltenVK Available: " << (info.has_moltenvk_available ? "Yes" : "No") << std::endl;
    std::cerr << "  Validation Layers: " << (info.has_validation_layers ? "Yes" : "No") << std::endl;

    if (!info.available_gpus.empty()) {
        std::cerr << "  Available GPUs:" << std::endl;
        for (const auto& gpu : info.available_gpus) {
            std::cerr << "    - " << gpu << std::endl;
        }
    }

    std::cerr << "🎯 Decisions:" << std::endl;
    std::cerr << "  Use MoltenVK: " << (decision.use_moltenvk ? "Yes" : "No") << std::endl;
    std::cerr << "  Use Validation: " << (decision.use_validation ? "Yes" : "No") << std::endl;
    std::cerr << "  Use Portability Ext: " << (decision.use_portability_extension ? "Yes" : "No") << std::endl;
    std::cerr << "  Reasoning: " << decision.reasoning << std::endl;

    for (const auto& warning : decision.warnings) {
        std::cerr << warning << std::endl;
    }
}

} // namespace platform_detection

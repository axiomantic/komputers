use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Detect and configure Vulkan
    let vulkan_config = detect_vulkan(&target_os).expect("Vulkan SDK not found");
    configure_vulkan_linking(&vulkan_config, &target_os);

    // Build Kompute C++ library
    let kompute_lib_path = build_kompute_library(&out_dir, &vulkan_config, &target_os)
        .expect("Failed to build Kompute C++ library");

    // Configure linking for Kompute
    configure_kompute_linking(&kompute_lib_path, &target_os);

    // Generate Rust bindings using bindgen
    generate_bindings(&out_dir, &vulkan_config).expect("Failed to generate Rust bindings");

    println!("cargo:rustc-link-lib=static=kompute_wrapper");
    println!("cargo:rustc-link-lib=static=kompute");
    println!("cargo:rustc-link-lib=static=fmt"); // fmt dependency
    println!("cargo:rustc-link-lib=static=kp_logger"); // logger dependency
    println!(
        "cargo:rustc-link-search=native={}",
        kompute_lib_path.display()
    );

    // Add search paths for Kompute dependencies built by CMake FetchContent
    let deps_lib_path = out_dir.join("kompute_build/build/_deps/kompute-build/src");
    println!("cargo:rustc-link-search=native={}", deps_lib_path.display());

    let fmt_lib_path = out_dir.join("kompute_build/build/_deps/fmt-build");
    println!("cargo:rustc-link-search=native={}", fmt_lib_path.display());

    let logger_lib_path = out_dir.join("kompute_build/build/_deps/kompute-build/src/logger");
    println!("cargo:rustc-link-search=native={}", logger_lib_path.display());
}

#[derive(Debug)]
struct VulkanConfig {
    sdk_path: PathBuf,
    #[allow(dead_code)]
    include_path: PathBuf,
    lib_path: PathBuf,
}

fn detect_vulkan(target_os: &str) -> Result<VulkanConfig, Box<dyn std::error::Error>> {
    // Try environment variable first
    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
        let sdk_path = PathBuf::from(vulkan_sdk);
        let include_path = sdk_path.join("include");
        let lib_path = match target_os {
            "windows" => sdk_path.join("Lib"),
            _ => sdk_path.join("lib"),
        };

        if include_path.exists() && lib_path.exists() {
            return Ok(VulkanConfig {
                sdk_path,
                include_path,
                lib_path,
            });
        }
    }

    // Platform-specific detection
    match target_os {
        "linux" => detect_vulkan_linux(),
        "windows" => detect_vulkan_windows(),
        "macos" => detect_vulkan_macos(),
        _ => Err(format!("Unsupported target OS: {}", target_os).into()),
    }
}

fn detect_vulkan_linux() -> Result<VulkanConfig, Box<dyn std::error::Error>> {
    // Try common system paths
    let possible_paths = vec![
        "/usr/include/vulkan",
        "/usr/local/include/vulkan",
        "/opt/vulkan-sdk/include/vulkan",
    ];

    for include_base in &possible_paths {
        let include_path = PathBuf::from(include_base).parent().unwrap().to_path_buf();
        let lib_path = include_path.parent().unwrap().join("lib");

        if include_path.join("vulkan/vulkan.h").exists() {
            return Ok(VulkanConfig {
                sdk_path: include_path.clone(),
                include_path,
                lib_path,
            });
        }
    }

    Err("Vulkan SDK not found on Linux. Please install vulkan-sdk or set VULKAN_SDK environment variable.".into())
}

fn detect_vulkan_windows() -> Result<VulkanConfig, Box<dyn std::error::Error>> {
    // Check common Windows installation paths
    let possible_base_paths = vec![
        "C:\\VulkanSDK",
        "C:\\Program Files\\VulkanSDK",
        "C:\\Program Files (x86)\\VulkanSDK",
    ];

    for base_path in &possible_base_paths {
        if let Ok(entries) = std::fs::read_dir(base_path) {
            for entry in entries.flatten() {
                if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                    let sdk_path = entry.path();
                    let include_path = sdk_path.join("Include");
                    let lib_path = sdk_path.join("Lib");

                    if include_path.join("vulkan/vulkan.h").exists() {
                        return Ok(VulkanConfig {
                            sdk_path,
                            include_path,
                            lib_path,
                        });
                    }
                }
            }
        }
    }

    Err("Vulkan SDK not found on Windows. Please install Vulkan SDK from LunarG or set VULKAN_SDK environment variable.".into())
}

fn detect_vulkan_macos() -> Result<VulkanConfig, Box<dyn std::error::Error>> {
    // Check for Vulkan SDK (MoltenVK)
    let possible_paths = vec![
        "/usr/local/include/vulkan",
        "/opt/homebrew/include/vulkan",
        "/usr/local/Cellar/vulkan-headers",
        "/opt/homebrew/Cellar/vulkan-headers",
    ];

    for include_base in &possible_paths {
        let include_path = if include_base.contains("Cellar") {
            // Handle Homebrew Cellar structure - find latest version
            if let Ok(entries) = std::fs::read_dir(include_base) {
                if let Some(version_dir) = entries.flatten().next() {
                    version_dir.path().join("include")
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else {
            PathBuf::from(include_base).parent().unwrap().to_path_buf()
        };

        let lib_path = include_path.parent().unwrap().join("lib");

        if include_path.join("vulkan/vulkan.h").exists() {
            return Ok(VulkanConfig {
                sdk_path: include_path.clone(),
                include_path,
                lib_path,
            });
        }
    }

    Err("Vulkan SDK not found on macOS. Please install vulkan-sdk via Homebrew: brew install vulkan-headers vulkan-loader molten-vk".into())
}

fn configure_vulkan_linking(vulkan_config: &VulkanConfig, target_os: &str) {
    println!(
        "cargo:rustc-link-search=native={}",
        vulkan_config.lib_path.display()
    );

    match target_os {
        "windows" => {
            println!("cargo:rustc-link-lib=vulkan-1");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=vulkan");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
            println!("cargo:rustc-link-lib=framework=IOKit");
            println!("cargo:rustc-link-lib=framework=IOSurface");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=QuartzCore");
        }
        _ => {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }
}

fn build_kompute_library(
    out_dir: &Path,
    vulkan_config: &VulkanConfig,
    target_os: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Create a CMakeLists.txt that uses FetchContent to download Kompute
    let cmake_dir = out_dir.join("cmake_build");
    std::fs::create_dir_all(&cmake_dir)?;

    create_fetchcontent_cmake(&cmake_dir)?;

    let build_dir = out_dir.join("kompute_build");
    std::fs::create_dir_all(&build_dir)?;

    // Configure CMake build using the generated CMakeLists.txt
    let mut cmake_config = cmake::Config::new(&cmake_dir);
    cmake_config
        .out_dir(&build_dir)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("KOMPUTE_OPT_BUILD_TESTS", "OFF")
        .define("KOMPUTE_OPT_BUILD_DOCS", "OFF")
        .define("KOMPUTE_OPT_BUILD_EXAMPLES", "OFF")
        .define("KOMPUTE_OPT_INSTALL", "OFF")
        .define("KOMPUTE_OPT_BUILD_SINGLE_HEADER", "OFF")
        .define("VULKAN_SDK", vulkan_config.sdk_path.to_str().unwrap());

    // Platform-specific configuration
    match target_os {
        "windows" => {
            cmake_config.define("CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS", "ON");
        }
        "macos" => {
            cmake_config
                .define("CMAKE_OSX_DEPLOYMENT_TARGET", "10.15")
                .define("KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADERS", "ON");
        }
        _ => {}
    }

    let kompute_build_path = cmake_config.build();

    // Find the built library
    let lib_dir = find_kompute_lib_dir(&kompute_build_path)?;
    Ok(lib_dir)
}

fn find_kompute_lib_dir(build_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // CMake typically puts libraries in lib/ or build/lib/
    let possible_lib_dirs = vec![
        build_path.join("lib"),
        build_path.join("build/lib"),
        build_path.join("lib/Release"),
        build_path.join("lib/Debug"),
    ];

    for lib_dir in possible_lib_dirs {
        if lib_dir.exists() {
            // Check for static or dynamic library
            let lib_files: Vec<_> = std::fs::read_dir(&lib_dir)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    let path = entry.path();
                    let name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                    name.contains("kompute_wrapper")
                        && (name.ends_with(".a")
                            || name.ends_with(".so")
                            || name.ends_with(".dylib")
                            || name.ends_with(".lib"))
                })
                .collect();

            if !lib_files.is_empty() {
                return Ok(lib_dir);
            }
        }
    }

    Err("Built Kompute library not found".into())
}

fn configure_kompute_linking(lib_path: &Path, target_os: &str) {
    println!("cargo:rustc-link-search=native={}", lib_path.display());

    // Link against C++ standard library
    match target_os {
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "windows" => {
            // MSVC automatically links C++ runtime
        }
        _ => {}
    }
}

fn generate_bindings(
    out_dir: &Path,
    _vulkan_config: &VulkanConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let wrapper_h = Path::new(&manifest_dir).join("wrapper.h");

    if !wrapper_h.exists() {
        create_wrapper_header()?;
    }

    let bindings = bindgen::Builder::default()
        .header(wrapper_h.to_str().unwrap())
        // Allow only the C wrapper functions
        .allowlist_function("kp_.*")
        .allowlist_type("kp_.*")
        // Simplify generation
        .use_core()
        .generate_comments(false)
        .derive_debug(false) // Keep it simple for testing
        .generate()?;

    let out_path = out_dir.join("bindings.rs");
    bindings.write_to_file(out_path)?;

    Ok(())
}

fn create_wrapper_header() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let wrapper_path = Path::new(&manifest_dir).join("wrapper.h");
    let wrapper_content = r#"
#ifndef KOMPUTE_RS_WRAPPER_H
#define KOMPUTE_RS_WRAPPER_H

// Pure C wrapper functions for critical operations - no C++ headers needed
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Opaque types
typedef void kp_Manager;
typedef void kp_Tensor;
typedef void kp_Algorithm;
typedef void kp_Sequence;

// Manager operations
kp_Manager* kp_manager_new(uint32_t device_id);
void kp_manager_destroy(kp_Manager* manager);
int kp_manager_get_device_count(void);

// Tensor operations
kp_Tensor* kp_tensor_new(kp_Manager* manager, const void* data, uint64_t size, uint32_t data_type);
void kp_tensor_destroy(kp_Tensor* tensor);
int kp_tensor_map_to_host(kp_Tensor* tensor);
int kp_tensor_map_to_device(kp_Tensor* tensor);

// Algorithm operations
kp_Algorithm* kp_algorithm_new(kp_Manager* manager, const uint32_t* spirv_data, uint64_t spirv_size);
void kp_algorithm_destroy(kp_Algorithm* algorithm);

// Sequence operations
kp_Sequence* kp_sequence_new(kp_Manager* manager);
void kp_sequence_destroy(kp_Sequence* sequence);
int kp_sequence_record_tensor_sync_device(kp_Sequence* sequence, kp_Tensor* tensor);
int kp_sequence_record_tensor_sync_local(kp_Sequence* sequence, kp_Tensor* tensor);
int kp_sequence_record_algo_dispatch(kp_Sequence* sequence, kp_Algorithm* algorithm);
int kp_sequence_eval(kp_Sequence* sequence);

#ifdef __cplusplus
}
#endif

#endif // KOMPUTE_RS_WRAPPER_H
"#;

    std::fs::write(wrapper_path, wrapper_content)?;
    Ok(())
}

fn create_fetchcontent_cmake(cmake_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let cmake_content = r#"cmake_minimum_required(VERSION 3.14)
project(kompute-wrapper)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use FetchContent to download Kompute
include(FetchContent)

FetchContent_Declare(
  kompute
  GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
  GIT_TAG v0.9.0
  GIT_SHALLOW TRUE
)

# Configure Kompute options before making it available
set(KOMPUTE_OPT_BUILD_TESTS OFF CACHE BOOL "")
set(KOMPUTE_OPT_BUILD_DOCS OFF CACHE BOOL "")
set(KOMPUTE_OPT_BUILD_EXAMPLES OFF CACHE BOOL "")
set(KOMPUTE_OPT_INSTALL OFF CACHE BOOL "")
set(KOMPUTE_OPT_BUILD_SINGLE_HEADER OFF CACHE BOOL "")

# Make Kompute available
FetchContent_MakeAvailable(kompute)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Create wrapper library that includes our C wrapper functions
add_library(kompute_wrapper STATIC
    wrapper.cpp
)

# Link against the fetched kompute library
target_link_libraries(kompute_wrapper
    PUBLIC kompute
    PUBLIC ${Vulkan_LIBRARIES}
)

target_include_directories(kompute_wrapper
    PRIVATE ${kompute_SOURCE_DIR}/src
    PRIVATE ${Vulkan_INCLUDE_DIRS}
)

# Install the wrapper library
install(TARGETS kompute_wrapper DESTINATION lib)
"#;

    let cmake_path = cmake_dir.join("CMakeLists.txt");
    std::fs::write(cmake_path, cmake_content)?;

    // Create the C++ wrapper implementation file
    create_cpp_wrapper(cmake_dir)?;

    Ok(())
}

fn create_cpp_wrapper(cmake_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let wrapper_cpp_content = r#"// C wrapper implementation for Kompute library
#include "kompute/Kompute.hpp"
#include <memory>
#include <vector>
#include <stdexcept>
#include <unordered_map>

// Global storage for shared_ptrs to maintain object lifetimes
static std::unordered_map<void*, std::shared_ptr<kp::Tensor>> tensor_registry;
static std::unordered_map<void*, std::shared_ptr<kp::Algorithm>> algorithm_registry;
static std::unordered_map<void*, std::shared_ptr<kp::Sequence>> sequence_registry;

extern "C" {

// Manager operations
kp::Manager* kp_manager_new(uint32_t device_id) {
    try {
        return new kp::Manager(device_id);
    } catch (const std::exception&) {
        return nullptr;
    }
}

void kp_manager_destroy(kp::Manager* manager) {
    delete manager;
}

int kp_manager_get_device_count() {
    try {
        // Create a temporary manager to get device count
        kp::Manager manager;
        // Note: This is a simplified approach - real implementation would
        // enumerate Vulkan physical devices
        return 1; // Placeholder - would use Vulkan instance enumeration
    } catch (const std::exception&) {
        return -1;
    }
}

// Tensor operations
kp::Tensor* kp_tensor_new(kp::Manager* manager, const void* data, uint64_t size, uint32_t data_type) {
    if (!manager) return nullptr;

    try {
        std::shared_ptr<kp::Tensor> tensor;

        if (data && size > 0) {
            // Convert size from bytes to element count based on data type
            size_t element_size = 4; // Default to float32
            switch (data_type) {
                case 0: element_size = 4; break; // float32
                case 1: element_size = 8; break; // float64
                case 2: element_size = 4; break; // int32
                case 3: element_size = 4; break; // uint32
                case 4: element_size = 1; break; // int8
                case 5: element_size = 1; break; // uint8
                default: return nullptr;
            }

            size_t element_count = size / element_size;

            // Create tensor based on data type using Manager
            if (data_type == 0) { // float32
                const float* float_data = static_cast<const float*>(data);
                std::vector<float> vec_data(float_data, float_data + element_count);
                tensor = manager->tensor(vec_data);
            } else {
                return nullptr; // Unsupported type for now
            }
        } else {
            // Create empty tensor for specified size
            if (data_type == 0) {
                std::vector<float> vec_data(size / 4, 0.0f);
                tensor = manager->tensor(vec_data);
            } else {
                return nullptr;
            }
        }

        // Store in registry and return raw pointer
        void* raw_ptr = tensor.get();
        tensor_registry[raw_ptr] = tensor;
        return tensor.get();

    } catch (const std::exception&) {
        return nullptr;
    }
}

void kp_tensor_destroy(kp::Tensor* tensor) {
    if (tensor) {
        tensor_registry.erase(tensor);
    }
}

int kp_tensor_map_to_host(kp::Tensor* tensor) {
    if (!tensor) return -1;
    try {
        // In Kompute v0.9.0, tensor sync is handled via sequences
        // This is a placeholder - actual sync would be done via OpTensorSyncLocal
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int kp_tensor_map_to_device(kp::Tensor* tensor) {
    if (!tensor) return -1;
    try {
        // In Kompute v0.9.0, tensor sync is handled via sequences
        // This is a placeholder - actual sync would be done via OpTensorSyncDevice
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

// Algorithm operations
kp::Algorithm* kp_algorithm_new(kp::Manager* manager, const uint32_t* spirv_data, uint64_t spirv_size) {
    if (!manager || !spirv_data) return nullptr;

    try {
        std::vector<uint32_t> spirv(spirv_data, spirv_data + (spirv_size / sizeof(uint32_t)));

        // In Kompute v0.9.0, algorithm() requires tensors as first parameter
        std::vector<std::shared_ptr<kp::Tensor>> empty_tensors;
        kp::Workgroup workgroup{1, 1, 1};
        std::vector<float> empty_spec_constants;
        std::vector<float> empty_push_constants;

        auto algorithm = manager->algorithm(
            empty_tensors,
            spirv,
            workgroup,
            empty_spec_constants,
            empty_push_constants
        );

        // Store in registry and return raw pointer
        void* raw_ptr = algorithm.get();
        algorithm_registry[raw_ptr] = algorithm;
        return algorithm.get();

    } catch (const std::exception&) {
        return nullptr;
    }
}

void kp_algorithm_destroy(kp::Algorithm* algorithm) {
    if (algorithm) {
        algorithm_registry.erase(algorithm);
    }
}

// Sequence operations
kp::Sequence* kp_sequence_new(kp::Manager* manager) {
    if (!manager) return nullptr;

    try {
        auto sequence = manager->sequence();

        // Store in registry and return raw pointer
        void* raw_ptr = sequence.get();
        sequence_registry[raw_ptr] = sequence;
        return sequence.get();

    } catch (const std::exception&) {
        return nullptr;
    }
}

void kp_sequence_destroy(kp::Sequence* sequence) {
    if (sequence) {
        sequence_registry.erase(sequence);
    }
}

int kp_sequence_record_tensor_sync_device(kp::Sequence* sequence, kp::Tensor* tensor) {
    if (!sequence || !tensor) return -1;

    try {
        // Find the shared_ptr in our registry
        auto it = tensor_registry.find(tensor);
        if (it == tensor_registry.end()) return -1;

        std::vector<std::shared_ptr<kp::Tensor>> tensors = {it->second};
        sequence->record<kp::OpTensorSyncDevice>(tensors);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int kp_sequence_record_tensor_sync_local(kp::Sequence* sequence, kp::Tensor* tensor) {
    if (!sequence || !tensor) return -1;

    try {
        // Find the shared_ptr in our registry
        auto it = tensor_registry.find(tensor);
        if (it == tensor_registry.end()) return -1;

        std::vector<std::shared_ptr<kp::Tensor>> tensors = {it->second};
        sequence->record<kp::OpTensorSyncLocal>(tensors);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int kp_sequence_record_algo_dispatch(kp::Sequence* sequence, kp::Algorithm* algorithm) {
    if (!sequence || !algorithm) return -1;

    try {
        // Find the shared_ptr in our registry
        auto it = algorithm_registry.find(algorithm);
        if (it == algorithm_registry.end()) return -1;

        sequence->record<kp::OpAlgoDispatch>(it->second);
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

int kp_sequence_eval(kp::Sequence* sequence) {
    if (!sequence) return -1;

    try {
        sequence->eval();
        return 0;
    } catch (const std::exception&) {
        return -1;
    }
}

} // extern "C"
"#;

    let wrapper_cpp_path = cmake_dir.join("wrapper.cpp");
    std::fs::write(wrapper_cpp_path, wrapper_cpp_content)?;

    Ok(())
}

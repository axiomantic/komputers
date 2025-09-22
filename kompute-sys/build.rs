use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
enum KomputeVersion {
    Master,
    V090,
    V081,
}

impl KomputeVersion {
    fn git_tag(&self) -> &'static str {
        match self {
            KomputeVersion::Master => "master",
            KomputeVersion::V090 => "v0.9.0",
            KomputeVersion::V081 => "v0.8.1",
        }
    }
}

impl std::fmt::Display for KomputeVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KomputeVersion::Master => write!(f, "master"),
            KomputeVersion::V090 => write!(f, "v0.9.0"),
            KomputeVersion::V081 => write!(f, "v0.8.1"),
        }
    }
}

fn determine_kompute_version() -> KomputeVersion {
    if cfg!(feature = "kompute-master") {
        KomputeVersion::Master
    } else if cfg!(feature = "kompute-v090") {
        KomputeVersion::V090
    } else if cfg!(feature = "kompute-v081") {
        KomputeVersion::V081
    } else {
        // This should be caught by compile_error! but just in case
        panic!("No Kompute version feature enabled!")
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    // Determine which Kompute version to use
    let kompute_version = determine_kompute_version();
    println!(
        "cargo:warning=Building with Kompute version: {}",
        kompute_version
    );

    // Configure AddressSanitizer if asan feature is enabled
    // Allow disabling ASAN via environment variable for CI compatibility
    let disable_asan = env::var("KOMPUTE_DISABLE_ASAN").is_ok();
    if cfg!(feature = "asan") && !disable_asan {
        configure_address_sanitizer();
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Build Kompute C++ library (it will manage its own Vulkan headers)
    let kompute_lib_path = build_kompute_library(&out_dir, &target_os, &kompute_version)
        .expect("Failed to build Kompute C++ library");

    // Configure linking for Kompute and system libraries
    configure_kompute_linking(&kompute_lib_path, &target_os);

    // Build the C++ bridge using cxx
    build_cxx_bridge(&out_dir).expect("Failed to build CXX bridge");

    println!("cargo:rustc-link-lib=static=kompute");
    println!("cargo:rustc-link-lib=static=fmt");
    println!("cargo:rustc-link-lib=static=kp_logger");
    println!(
        "cargo:rustc-link-search=native={}",
        kompute_lib_path.display()
    );

    // Add search paths for Kompute dependencies
    let deps_lib_path = out_dir.join("kompute_build/build/_deps/kompute-build/src");
    println!("cargo:rustc-link-search=native={}", deps_lib_path.display());

    let fmt_lib_path = out_dir.join("kompute_build/build/_deps/fmt-build");
    println!("cargo:rustc-link-search=native={}", fmt_lib_path.display());

    let logger_lib_path = out_dir.join("kompute_build/build/_deps/kompute-build/src/logger");
    println!(
        "cargo:rustc-link-search=native={}",
        logger_lib_path.display()
    );
}

fn build_kompute_library(
    out_dir: &Path,
    target_os: &str,
    kompute_version: &KomputeVersion,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cmake_dir = out_dir.join("cmake_build");
    std::fs::create_dir_all(&cmake_dir)?;

    create_kompute_cmake(&cmake_dir, kompute_version)?;

    let build_dir = out_dir.join("kompute_build");
    std::fs::create_dir_all(&build_dir)?;

    let mut cmake_config = cmake::Config::new(&cmake_dir);
    cmake_config
        .out_dir(&build_dir)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("KOMPUTE_OPT_BUILD_TESTS", "OFF")
        .define("KOMPUTE_OPT_BUILD_DOCS", "OFF")
        .define("KOMPUTE_OPT_BUILD_EXAMPLES", "OFF")
        .define("KOMPUTE_OPT_INSTALL", "OFF")
        .define("KOMPUTE_OPT_BUILD_SINGLE_HEADER", "OFF")
        .define("KOMPUTE_OPT_USE_BUILT_IN_VULKAN_HEADERS", "ON");

    // Add AddressSanitizer flags to CMake build if asan feature is enabled
    if cfg!(feature = "asan") {
        cmake_config
            .define(
                "CMAKE_CXX_FLAGS",
                "-fsanitize=address -g -O1 -fno-omit-frame-pointer",
            )
            .define(
                "CMAKE_C_FLAGS",
                "-fsanitize=address -g -O1 -fno-omit-frame-pointer",
            )
            .define("CMAKE_EXE_LINKER_FLAGS", "-fsanitize=address")
            .define("CMAKE_SHARED_LINKER_FLAGS", "-fsanitize=address");
    }

    match target_os {
        "windows" => {
            cmake_config.define("CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS", "ON");
        }
        "macos" => {
            // Set deployment target for compatibility - also set as environment variable
            let deployment_target = "11.0"; // Changed to match modern CI environments
            cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", deployment_target);
            println!(
                "cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET={}",
                deployment_target
            );
            // Fix for Kompute + MoltenVK compatibility issue
            // Reference: https://github.com/KhronosGroup/MoltenVK/issues/2292
            cmake_config.define("KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK", "ON");
            // Disable all debugging and validation features for MoltenVK compatibility
            cmake_config.define("KOMPUTE_DISABLE_VK_DEBUG_LAYERS", "ON");
            cmake_config.define("CMAKE_BUILD_TYPE", "Release");
            // CRITICAL FIX: Force Vulkan API 1.2 for MoltenVK compatibility
            // "For anyone coming across this in the future, ensure that vk::ApplicationInfo.apiVersion is 1.2 or lower.
            // MoltenVK will gladly load with 1.3 but that causes issues like this."
            cmake_config.define("KOMPUTE_VK_API_MAJOR_VERSION", "1");
            cmake_config.define("KOMPUTE_VK_API_MINOR_VERSION", "2");
        }
        _ => {}
    }

    let kompute_build_path = cmake_config.build();
    let lib_dir = find_kompute_lib_dir(&kompute_build_path)?;
    Ok(lib_dir)
}

fn find_kompute_lib_dir(build_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let possible_lib_dirs = vec![
        build_path.join("lib"),
        build_path.join("build/lib"),
        build_path.join("lib/Release"),
        build_path.join("lib/Debug"),
        // CMake FetchContent puts libraries in build/_deps/kompute-build/src/
        build_path.join("build/_deps/kompute-build/src"),
    ];

    for lib_dir in possible_lib_dirs {
        if lib_dir.exists() {
            let lib_files: Vec<_> = std::fs::read_dir(&lib_dir)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    let path = entry.path();
                    let name = path.file_name().unwrap_or_default().to_str().unwrap_or("");
                    name.contains("kompute")
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

fn build_cxx_bridge(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Find the Kompute headers
    let kompute_build_dir = out_dir.join("kompute_build/build/_deps/kompute-src");
    let kompute_include_dir = kompute_build_dir.join("src/include");

    if !kompute_include_dir.exists() {
        return Err("Kompute headers not found".into());
    }

    // Find dependency include paths
    let fmt_include_dir = out_dir.join("kompute_build/build/_deps/fmt-src/include");
    let spdlog_include_dir = out_dir.join("kompute_build/build/_deps/spdlog-src/include");

    // Use Kompute's built-in Vulkan headers instead of system ones to avoid version conflicts
    let vulkan_builtin_include =
        out_dir.join("kompute_build/build/_deps/vulkan_header-src/include");

    // Build the CXX bridge
    let mut cxx_builder = cxx_build::bridge("src/bridge.rs");
    cxx_builder
        .file("src/kompute_bridge.cpp")
        .file("src/handle_registry.cpp")
        .include(&kompute_include_dir)
        .include(&vulkan_builtin_include) // Use Kompute's Vulkan headers
        .include(&fmt_include_dir)
        .include(&spdlog_include_dir)
        .include("src") // For our bridge header
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-DKOMPUTE_ENABLE_SPDLOG=1")
        .flag_if_supported("-DFMT_HEADER_ONLY=1")
        .flag_if_supported("-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1");

    // Add version-specific defines based on feature flags
    let kompute_version = determine_kompute_version();
    match kompute_version {
        KomputeVersion::V081 => {
            cxx_builder.flag_if_supported("-DKOMPUTE_VERSION_V081=1");
        }
        KomputeVersion::V090 => {
            cxx_builder.flag_if_supported("-DKOMPUTE_VERSION_V090=1");
        }
        KomputeVersion::Master => {
            cxx_builder.flag_if_supported("-DKOMPUTE_VERSION_MASTER=1");
        }
    }

    // Add Vulkan API version defines for platform compatibility
    #[cfg(target_os = "macos")]
    {
        // CRITICAL: Force Vulkan API 1.2 for MoltenVK compatibility on macOS
        // Reference: https://github.com/KhronosGroup/MoltenVK/issues/2292
        // Override the version BEFORE Kompute headers are included
        cxx_builder
            .flag_if_supported("-DKOMPUTE_VK_API_MAJOR_VERSION=1")
            .flag_if_supported("-DKOMPUTE_VK_API_MINOR_VERSION=2")
            .flag_if_supported("-DKOMPUTE_VK_API_VERSION=VK_MAKE_VERSION(1,2,0)");
    }

    // Add AddressSanitizer flags to C++ compilation if asan feature is enabled
    if cfg!(feature = "asan") {
        cxx_builder
            .flag_if_supported("-fsanitize=address")
            .flag_if_supported("-g")
            .flag_if_supported("-O1")
            .flag_if_supported("-fno-omit-frame-pointer");
    }

    cxx_builder.compile("kompute_bridge");

    // Link to Vulkan libraries - we still need runtime linking
    #[cfg(target_os = "macos")]
    {
        // Set deployment target for linker consistency
        println!("cargo:rustc-link-arg=-mmacosx-version-min=11.0");

        // Link to Vulkan directly on macOS
        if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
            let vulkan_lib_path = Path::new(&vulkan_sdk).join("lib");
            println!(
                "cargo:rustc-link-search=native={}",
                vulkan_lib_path.display()
            );
        }
        // Also try homebrew path
        println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        println!("cargo:rustc-link-lib=vulkan");

        // Metal frameworks for MoltenVK compatibility
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=IOKit");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
    }

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=vulkan");
    }

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=vulkan-1");
    }

    Ok(())
}

fn configure_address_sanitizer() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    println!("cargo:rustc-link-arg=-fsanitize=address");
    println!("cargo:rustc-link-arg=-g");
    println!("cargo:rustc-link-arg=-O1");
    println!("cargo:rustc-link-arg=-fno-omit-frame-pointer");

    // Set C++ flags for AddressSanitizer
    println!("cargo:rustc-env=CXXFLAGS=-fsanitize=address -g -O1 -fno-omit-frame-pointer");
    println!("cargo:rustc-env=CFLAGS=-fsanitize=address -g -O1 -fno-omit-frame-pointer");

    // Platform-specific linker configuration
    match target_os.as_str() {
        "linux" => {
            // On Linux, link to libasan explicitly
            println!("cargo:rustc-link-lib=asan");
        }
        "macos" => {
            // On macOS, we need to link against the ASAN runtime explicitly
            // Even though it's built into clang, we need the runtime symbols
            println!("cargo:rustc-link-arg=-fsanitize=address");
            // Also try to link the ASAN library directly
            println!("cargo:rustc-link-lib=asan");
        }
        "windows" => {
            // On Windows with MSVC, AddressSanitizer support varies
            // For now, just set the compiler flags
        }
        _ => {}
    }
}

fn create_kompute_cmake(
    cmake_dir: &Path,
    kompute_version: &KomputeVersion,
) -> Result<(), Box<dyn std::error::Error>> {
    let git_tag = kompute_version.git_tag();
    let cmake_content = format!(
        r#"cmake_minimum_required(VERSION 3.14)
project(kompute-rust-bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  kompute
  GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
  GIT_TAG {}
  GIT_SHALLOW TRUE
)

set(KOMPUTE_OPT_BUILD_TESTS OFF CACHE BOOL "")
set(KOMPUTE_OPT_BUILD_DOCS OFF CACHE BOOL "")
set(KOMPUTE_OPT_BUILD_EXAMPLES OFF CACHE BOOL "")
set(KOMPUTE_OPT_INSTALL OFF CACHE BOOL "")
set(KOMPUTE_OPT_BUILD_SINGLE_HEADER OFF CACHE BOOL "")

FetchContent_MakeAvailable(kompute)

find_package(Vulkan REQUIRED)
"#,
        git_tag
    );

    let cmake_path = cmake_dir.join("CMakeLists.txt");
    std::fs::write(cmake_path, cmake_content)?;

    Ok(())
}

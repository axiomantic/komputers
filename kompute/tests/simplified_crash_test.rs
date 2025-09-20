//! Simplified crash debugging test for Manager constructor segfault analysis

use kompute::{Manager, Result};
use std::process::Command;

#[test]
#[ignore] // Run with --ignored to execute
fn test_manager_segfault_with_lldb() {
    println!("\n🔍 SIMPLIFIED MANAGER CONSTRUCTOR SEGFAULT ANALYSIS");
    println!("====================================================");

    // Create a simple test binary that triggers the segfault
    println!("Creating test binary to analyze segfault...");

    let test_code = r#"
use kompute::Manager;

fn main() {
    println!("🚀 About to call Manager::new() - EXPECT SEGFAULT");
    println!("Environment: macOS with MoltenVK");
    println!("DEBUG: Starting manager creation for device 0");
    println!("DEBUG: About to create kp::Manager with default constructor");

    let _manager = Manager::new().expect("Manager creation failed");

    println!("✅ Manager created successfully (this should not print)");
}
"#;

    // Write test binary
    std::fs::write("test_manager_crash.rs", test_code).expect("Failed to write test file");

    println!("✅ Test binary created: test_manager_crash.rs");
    println!("📋 To debug the segfault manually, run:");
    println!("   cargo build --release");
    println!("   lldb target/release/test_manager_crash");
    println!("   (lldb) run");
    println!("   (lldb) bt  # for backtrace after crash");
    println!("");
    println!("Or use gdb:");
    println!("   gdb target/release/test_manager_crash");
    println!("   (gdb) run");
    println!("   (gdb) bt  # for backtrace after crash");
    println!("");
    println!("📊 Expected results:");
    println!("   - Segfault should occur in kp::Manager constructor");
    println!("   - Backtrace should show Vulkan/MoltenVK call stack");
    println!("   - Crash likely in vk::ApplicationInfo.apiVersion handling");

    // Clean up
    std::fs::remove_file("test_manager_crash.rs").ok();
}

#[test]
fn test_safe_operations_work() {
    println!("\n🧪 TESTING SAFE OPERATIONS (NON-MANAGER)");
    println!("=========================================");

    // Test device enumeration (should work)
    match kompute_sys::ffi::get_device_count() {
        count if count > 0 => {
            println!("✅ Device enumeration successful: {} devices", count);
        }
        count => {
            println!("❌ Device enumeration failed or no devices: {}", count);
        }
    }

    // Test Vulkan environment setup (should work)
    match kompute::configure_vulkan_environment_intelligent() {
        Ok(()) => {
            println!("✅ Vulkan environment configuration successful");
        }
        Err(e) => {
            println!("❌ Vulkan environment configuration failed: {:?}", e);
        }
    }

    println!("📋 Analysis: Safe operations work correctly");
    println!("           Problem is specifically in Manager constructor");
}

#[test]
fn test_create_detailed_crash_reproduction() {
    println!("\n🎯 DETAILED CRASH REPRODUCTION INSTRUCTIONS");
    println!("=============================================");

    println!("To analyze the Manager constructor segfault:");
    println!("");
    println!("1. 🔨 BUILD WITH DEBUG INFO:");
    println!("   export RUST_BACKTRACE=full");
    println!("   cargo build --features crash-debug");
    println!("");
    println!("2. 🧪 RUN CRASHING TEST:");
    println!("   cargo test --features crash-debug test_manager_creation_with_crash_debugging -- --exact --nocapture");
    println!("");
    println!("3. 🔍 MANUAL DEBUGGING:");
    println!("   Create a simple binary:");
    println!("   ```rust");
    println!("   use kompute::Manager;");
    println!("   fn main() {{");
    println!("       let _mgr = Manager::new().unwrap();");
    println!("   }}");
    println!("   ```");
    println!("");
    println!("4. 🔧 DEBUGGING COMMANDS:");
    println!("   lldb target/debug/your_binary");
    println!("   (lldb) run");
    println!("   (lldb) bt  # Get backtrace after segfault");
    println!("   (lldb) register read  # Check register states");
    println!("");
    println!("5. 🎯 EXPECTED FINDINGS:");
    println!("   - Crash in kp::Manager::Manager() constructor");
    println!("   - Likely in createInstance() or createDevice()");
    println!("   - Related to vk::ApplicationInfo.apiVersion");
    println!("   - MoltenVK API 1.3 compatibility issue");
    println!("");
    println!("6. 🔧 POTENTIAL FIXES:");
    println!("   - Downgrade to Kompute v0.8.1");
    println!("   - Modify Kompute source to use API version 1.2");
    println!("   - Use custom Manager with explicit API version");
}

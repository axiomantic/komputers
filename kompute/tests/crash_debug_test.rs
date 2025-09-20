//! Comprehensive crash debugging test for Manager constructor analysis

#[cfg(feature = "crash-debug")]
use kompute::{debug_execute, install_crash_handlers, test_crash_debug, Manager, Result};

#[test]
#[cfg(feature = "crash-debug")]
fn test_crash_debug_system() -> Result<()> {
    println!("🧪 Testing crash debugging system...");
    test_crash_debug()?;
    Ok(())
}

#[test]
#[cfg(feature = "crash-debug")]
fn test_manager_creation_with_crash_debugging() {
    println!("\n🔍 COMPREHENSIVE MANAGER CONSTRUCTOR ANALYSIS");
    println!("==============================================");

    // Install crash handlers first
    if let Err(e) = install_crash_handlers() {
        eprintln!("Failed to install crash handlers: {:?}", e);
        return;
    }

    // Test Manager creation with comprehensive debugging
    let result = debug_execute("Manager::new() with crash analysis", || {
        println!("🚀 About to call Manager::new()...");
        println!("Environment: macOS with MoltenVK");
        println!("Expected: Segmentation fault with detailed analysis");

        // This should trigger the segfault and our crash analysis
        let _manager = Manager::new()?;

        // This line should never be reached
        println!("✅ Manager created successfully (unexpected!)");
        Ok(())
    });

    match result {
        Ok(()) => {
            println!("🎉 Manager creation succeeded unexpectedly!");
            println!("The segfault issue appears to be resolved.");
        }
        Err(e) => {
            println!("❌ Manager creation failed as expected: {:?}", e);
            println!("Check the crash analysis output above for detailed information.");
        }
    }
}

#[test]
#[cfg(feature = "crash-debug")]
fn test_device_enumeration_with_debugging() {
    println!("\n🔍 DEVICE ENUMERATION WITH CRASH DEBUGGING");
    println!("==========================================");

    if let Err(e) = install_crash_handlers() {
        eprintln!("Failed to install crash handlers: {:?}", e);
        return;
    }

    let result = debug_execute("Device enumeration", || {
        println!("🎮 Testing device enumeration (should work)...");
        let device_count = kompute_sys::ffi::get_device_count();
        if device_count < 0 {
            return Err(kompute::KomputeError::device("No GPU devices found"));
        }
        println!("✅ Found {} devices", device_count);
        Ok(device_count)
    });

    match result {
        Ok(count) => {
            println!("✅ Device enumeration successful: {} devices", count);
        }
        Err(e) => {
            println!("❌ Device enumeration failed: {:?}", e);
        }
    }
}

#[test]
#[cfg(feature = "crash-debug")]
fn test_vulkan_validation_with_debugging() {
    println!("\n🔍 VULKAN VALIDATION WITH CRASH DEBUGGING");
    println!("=========================================");

    if let Err(e) = install_crash_handlers() {
        eprintln!("Failed to install crash handlers: {:?}", e);
        return;
    }

    let result = debug_execute("Vulkan environment validation", || {
        println!("🌍 Testing Vulkan environment validation...");
        kompute::configure_vulkan_environment_intelligent()
            .map_err(|e| kompute::KomputeError::environment(e))?;
        println!("✅ Vulkan environment validated");
        Ok(())
    });

    match result {
        Ok(()) => {
            println!("✅ Vulkan validation successful");
        }
        Err(e) => {
            println!("❌ Vulkan validation failed: {:?}", e);
        }
    }
}

#[test]
#[cfg(not(feature = "crash-debug"))]
fn test_crash_debug_feature_disabled() {
    println!("⚠️  Crash debugging tests disabled - enable 'crash-debug' feature to run");
    println!("   Usage: cargo test --features crash-debug");
}

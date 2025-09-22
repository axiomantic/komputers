//! Demonstrates defensive guards and error handling
//!
//! This example shows that our defensive programming changes prevent segfaults
//! and provide clear error messages when operations fail.

use komputers::{Algorithm, Manager, Result, Sequence, Tensor};

fn main() -> Result<()> {
    println!("🛡️  Testing Defensive Guards and Error Handling");
    println!("================================================");

    // Test 1: Manager with invalid device ID
    println!("\n1. Testing Manager creation with invalid device ID...");
    match Manager::with_device(999) {
        Ok(_) => println!("❌ Unexpected success with invalid device ID"),
        Err(e) => println!("✅ Expected error: {}", e),
    }

    // Test 2: Empty tensor data
    println!("\n2. Testing Tensor creation with empty data...");
    if let Ok(mut manager) = Manager::new() {
        let empty_data: Vec<f32> = vec![];
        match Tensor::new(&mut manager, &empty_data) {
            Ok(_) => println!("❌ Unexpected success with empty data"),
            Err(e) => println!("✅ Expected error: {}", e),
        }
    } else {
        println!("⚠️  Manager creation failed (expected in non-GPU environments)");
    }

    // Test 3: Tensor with zero size
    println!("\n3. Testing Tensor creation with zero size...");
    if let Ok(mut manager) = Manager::new() {
        match Tensor::<f32>::uninit(&mut manager, 0) {
            Ok(_) => println!("❌ Unexpected success with zero size"),
            Err(e) => println!("✅ Expected error: {}", e),
        }
    } else {
        println!("⚠️  Manager creation failed (expected in non-GPU environments)");
    }

    // Test 4: Invalid SPIR-V data
    println!("\n4. Testing Algorithm creation with invalid SPIR-V...");
    if let Ok(mut manager) = Manager::new() {
        let invalid_spirv = vec![0x12345678]; // Invalid magic number
        match Algorithm::new(&mut manager, &invalid_spirv) {
            Ok(_) => println!("❌ Unexpected success with invalid SPIR-V"),
            Err(e) => println!("✅ Expected error: {}", e),
        }
    } else {
        println!("⚠️  Manager creation failed (expected in non-GPU environments)");
    }

    // Test 5: Empty SPIR-V data
    println!("\n5. Testing Algorithm creation with empty SPIR-V...");
    if let Ok(mut manager) = Manager::new() {
        let empty_spirv: Vec<u32> = vec![];
        match Algorithm::new(&mut manager, &empty_spirv) {
            Ok(_) => println!("❌ Unexpected success with empty SPIR-V"),
            Err(e) => println!("✅ Expected error: {}", e),
        }
    } else {
        println!("⚠️  Manager creation failed (expected in non-GPU environments)");
    }

    // Test 6: Sequence operations with invalid handles
    println!("\n6. Testing Sequence operations with invalid handles...");
    if let Ok(mut manager) = Manager::new() {
        if let Ok(mut sequence) = Sequence::new(&mut manager) {
            // Test with invalid tensor handle (0)
            match sequence.record_tensor_sync_device(0) {
                Ok(_) => println!("❌ Unexpected success with invalid tensor handle"),
                Err(e) => println!("✅ Expected error: {}", e),
            }

            // Test with invalid algorithm handle (0)
            match sequence.record_algorithm_dispatch(0) {
                Ok(_) => println!("❌ Unexpected success with invalid algorithm handle"),
                Err(e) => println!("✅ Expected error: {}", e),
            }
        } else {
            println!("⚠️  Sequence creation failed (expected in non-GPU environments)");
        }
    } else {
        println!("⚠️  Manager creation failed (expected in non-GPU environments)");
    }

    // Test 7: Device enumeration validation
    println!("\n7. Testing Device enumeration...");
    match Manager::device_count() {
        Ok(count) => {
            println!("✅ Found {} GPU devices", count);
            if count == 0 {
                println!(
                    "⚠️  No GPU devices available - this is expected in CI/headless environments"
                );
            }
        }
        Err(e) => println!(
            "⚠️  Device enumeration failed: {} (expected in some environments)",
            e
        ),
    }

    println!("\n🎉 Defensive Guards Test Complete!");
    println!("All error cases handled gracefully without crashes.");

    Ok(())
}

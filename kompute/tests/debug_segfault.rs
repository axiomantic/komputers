//! Debug test to isolate segfault issues

use kompute::Manager;

#[test]
fn test_simple_device_count() {
    println!("Starting device count test...");

    // Try to get device count without creating any managers first
    match Manager::device_count() {
        Ok(count) => {
            println!("Device count: {}", count);
        }
        Err(e) => {
            println!("Error getting device count: {}", e);
        }
    }

    println!("Device count test completed");
}

#[test]
fn test_simple_manager_creation() {
    println!("Starting manager creation test...");

    // Try creating a manager with default device
    match Manager::new() {
        Ok(manager) => {
            println!("Manager created successfully");
            std::mem::drop(manager);
            println!("Manager dropped successfully");
        }
        Err(e) => {
            println!("Error creating manager: {}", e);
        }
    }

    println!("Manager creation test completed");
}

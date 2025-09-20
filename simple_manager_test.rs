use kompute::Manager;

fn main() {
    println!("🚀 About to call Manager::new() - EXPECT CRASH");
    println!("Environment: macOS with MoltenVK");
    println!("DEBUG: Starting manager creation for device 0");

    match Manager::new() {
        Ok(_manager) => {
            println!("✅ Manager created successfully (unexpected!)");
        }
        Err(e) => {
            println!("❌ Manager creation failed with error: {:?}", e);
            println!("Error type: {}", e);
            std::process::exit(1);
        }
    }
}

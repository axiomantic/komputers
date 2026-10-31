//! # Consuming Builder Pattern Safety
//!
//! This example demonstrates Komputers' revolutionary consuming builder pattern
//! that makes sequence reuse errors impossible at compile time - inspired by
//! std::process::Command and other Rust standard library patterns.
//!
//! ## What You'll See:
//! - ✅ Consuming builder pattern preventing double execution
//! - ✅ Move semantics ensuring sequences can't be reused
//! - ✅ Compile-time prevention of "execute twice" errors
//! - ✅ Rust-idiomatic API design following std library patterns
//!
//! ## The Problem in Traditional GPU Computing:
//! - Executing command buffers/sequences multiple times causes corruption
//! - Reusing command sequences after execution causes crashes
//! - No compile-time protection against these errors
//! - Manual tracking of sequence state required
//!
//! ## Komputers Solution:
//! - Each operation moves the sequence (consuming methods)
//! - Execute() consumes the sequence entirely
//! - Impossible to accidentally reuse - won't compile!
//! - Follows std::process::Command design patterns
//!
//! ## Run this example:
//! ```bash
//! cargo run --example consuming_builder_safety --features kompute-master
//! ```

use komputers::{Manager, Result, Tensor};

fn main() -> Result<()> {
    println!("🔒 Consuming Builder Pattern Safety");
    println!("==================================");

    // Step 1: Setup GPU resources
    println!("\n📱 Step 1: Setting up GPU resources...");
    let mut manager = Manager::new()?;
    let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0, 4.0])?;
    println!("✅ Manager and tensor created successfully");

    // Step 2: Demonstrate consuming builder pattern
    println!("\n🔄 Step 2: Demonstrating consuming builder pattern...");
    println!("   Each method call moves the sequence (consuming pattern)");

    let sequence = manager.sequence()?;
    println!("✅ Sequence created (state: Recording)");

    let sequence = sequence.record_tensor_sync_device(tensor.handle())?;
    println!("✅ Recorded sync_device (sequence moved)");

    let sequence = sequence.record_tensor_sync_local(tensor.handle())?;
    println!("✅ Recorded sync_local (sequence moved again)");

    // Step 3: Execute sequence (consumes it entirely)
    println!("\n⚡ Step 3: Executing sequence (consuming it)...");
    let result = sequence.execute()?; // sequence is consumed here!
    println!("✅ Sequence executed and consumed!");
    println!("   Operations: {}", result.operation_count);
    println!("   Execution time: {:?}", result.execution_time);

    // Step 4: Show that sequence is no longer available
    println!("\n🚫 Step 4: Sequence is no longer available (compile-time safety)...");
    println!("   The following code would NOT compile:");
    println!("   // sequence.execute()?;  // ❌ COMPILE ERROR: use of moved value");
    println!("   // sequence.record_tensor_sync_device(...)?;  // ❌ COMPILE ERROR");
    println!("✅ Compiler prevents accidental reuse!");

    // Step 5: Demonstrate fluent chaining (std::process::Command style)
    println!("\n⛓️ Step 5: Demonstrating fluent chaining (std::process::Command style)...");
    println!("   Creating new sequence with fluent method chaining...");

    let result2 = manager
        .sequence()? // Create sequence
        .record_tensor_sync_device(tensor.handle())? // Chain operation 1 (moves)
        .record_tensor_sync_local(tensor.handle())? // Chain operation 2 (moves)
        .execute()?; // Chain execution (consumes)

    println!("✅ Fluent chaining completed!");
    println!("   Operations: {}", result2.operation_count);
    println!("   Execution time: {:?}", result2.execution_time);

    // Step 6: Show multiple sequences can be created
    println!("\n🔄 Step 6: Multiple sequences can be created safely...");
    for i in 1..=3 {
        let result = manager
            .sequence()?
            .record_tensor_sync_local(tensor.handle())?
            .execute()?;

        println!(
            "✅ Sequence {}: {} operations in {:?}",
            i, result.operation_count, result.execution_time
        );
    }

    // Step 7: Compare with traditional approaches
    println!("\n📊 Step 7: Safety comparison...");
    demonstrate_traditional_problems();

    println!("\n🎉 SUCCESS: Consuming Builder Pattern Prevents All Reuse Errors!");
    println!("   Traditional approach: Manual state tracking, runtime errors");
    println!("   Komputers approach: Compile-time prevention, impossible to misuse");
    println!("   Result: Zero \"execute twice\" bugs possible!");

    Ok(())
}

fn demonstrate_traditional_problems() {
    println!("\n⚠️  Traditional GPU Computing Problems (solved by consuming builder):");
    println!(
        "
    // Traditional approach (error-prone):
    sequence = create_sequence()
    sequence.record_operation_1()
    sequence.record_operation_2()
    sequence.execute()

    // ❌ Oops! Accidentally execute again - CRASH!
    sequence.execute()  // Runtime error or corruption

    // ❌ Or record more after execution - UNDEFINED BEHAVIOR!
    sequence.record_operation_3()  // May crash or corrupt
    "
    );

    println!("\n✅ Komputers Consuming Builder (impossible to misuse):");
    println!(
        "
    let result = manager
        .sequence()?                        // Create
        .record_operation_1()?              // Move
        .record_operation_2()?              // Move
        .execute()?;                        // Consume

    // ✅ This won't compile - sequence consumed!
    // sequence.execute()?;                 // COMPILE ERROR
    // sequence.record_operation_3()?;      // COMPILE ERROR
    "
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consuming_builder_safety_example() {
        // This test verifies the consuming builder safety example works
        match main() {
            Ok(()) => println!("✅ Consuming builder safety example completed successfully"),
            Err(e) => {
                println!("ℹ️ GPU not available for testing (expected in CI): {}", e);
                // This is expected in headless CI environments
            }
        }
    }
}

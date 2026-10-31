# Quick Start: Type-Safe GPU Computing in 5 Minutes

Get productive with Komputers revolutionary type-safe GPU computing in just 5 minutes. This guide will have you running secure GPU operations with compile-time safety guarantees.

## 🎯 What You'll Accomplish

By the end of this 5-minute guide, you'll:

- ✅ Set up Komputers with Vulkan
- ✅ Run your first type-safe GPU operation
- ✅ See compile-time safety in action
- ✅ Handle errors gracefully
- ✅ Understand why crashes are now impossible

## 📋 Prerequisites

- Rust 1.70+ installed
- GPU with Vulkan drivers (NVIDIA, AMD, Intel, or Apple)
- 5 minutes of your time

## ⚡ Step 1: Installation (30 seconds)

Add Komputers to your `Cargo.toml`:

```toml
[dependencies]
komputers = "0.3.0"

# Optional: Enable async support
[features]
default = ["async"]
```

**Vulkan Setup:**

- **Linux**: `sudo apt install vulkan-tools vulkan-utils` (Ubuntu/Debian)
- **macOS**: Vulkan automatically available via MoltenVK
- **Windows**: Install GPU drivers (Vulkan included)

Verify Vulkan is working:

```bash
vulkaninfo  # Should show your GPU information
```

## 🚀 Step 2: Your First Type-Safe GPU Program (2 minutes)

Create `src/main.rs`:

```rust
use komputers::*;

fn main() -> Result<()> {
    println!("🚀 Starting type-safe GPU computing...");

    // Step 1: Create GPU manager with automatic setup
    let mut manager = Manager::new()?;
    println!("✅ GPU manager initialized successfully");

    // Step 2: Create tensors with automatic GPU memory management
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let tensor = Tensor::new(&mut manager, &input_data)?;
    println!("✅ Created tensor with {} elements", tensor.size());

    // Step 3: Perform type-safe GPU operation
    let result = manager
        .sequence()?                                    // Create execution sequence
        .record_tensor_sync_device(tensor.handle())?   // Upload data to GPU
        .record_tensor_sync_local(tensor.handle())?    // Download results
        .execute()?;                                    // Execute safely

    println!("✅ GPU operation completed!");
    println!("   Operations executed: {}", result.operation_count);
    println!("   Execution time: {:?}", result.execution_time);

    // Step 4: Access results safely
    let output_data = tensor.data()?;
    println!("📊 Results: {:?}", output_data);

    println!("🎉 Type-safe GPU computing successful!");
    Ok(())
}
```

Run it:

```bash
cargo run
```

Expected output:

```
🚀 Starting type-safe GPU computing...
✅ GPU manager initialized successfully
✅ Created tensor with 5 elements
✅ GPU operation completed!
   Operations executed: 2
   Execution time: 1.234ms
📊 Results: [1.0, 2.0, 3.0, 4.0, 5.0]
🎉 Type-safe GPU computing successful!
```

## 🛡️ Step 3: See Compile-Time Safety in Action (1 minute)

Let's see why crashes are now impossible. Add this **commented-out** code to see what compile-time safety prevents:

```rust
fn demonstrate_safety() {
    // ❌ This code CANNOT compile - safety prevents it!

    /* Uncomment to see compile errors:

    use komputers::*;

    fn impossible_cross_manager_usage() -> Result<()> {
        // Two different managers
        let mut manager1 = Manager::new()?;
        let mut manager2 = Manager::new()?;

        // Tensor from manager1
        let tensor = Tensor::new(&mut manager1, &[1.0f32])?;

        // Try to use tensor with manager2 - COMPILE ERROR!
        manager2.sequence()?
            .record_tensor_sync_device(tensor.handle())?  // ❌ Won't compile!
            .execute()?;

        Ok(())
    }

    fn impossible_sequence_reuse() -> Result<()> {
        let mut manager = Manager::new()?;
        let tensor = Tensor::new(&mut manager, &[1.0f32])?;

        // Create sequence
        let sequence = manager.sequence()?;

        // Execute it
        let sequence = sequence.record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;  // Consumes the sequence

        // Try to reuse - COMPILE ERROR!
        sequence.record_tensor_sync_device(tensor.handle())?;  // ❌ Won't compile!

        Ok(())
    }

    */
}
```

**Try uncommenting this code** - you'll see helpful compile errors like:

```
error: cannot use handle from different manager instance
  --> src/main.rs:XX:YY
   |
   | manager2.sequence()?.record_tensor_sync_device(tensor.handle())?
   |                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: tensor handle is from manager1, but sequence is from manager2
   = help: ensure all operations use handles from the same manager
```

## 🔍 Step 4: Graceful Error Handling (1 minute)

Real applications need error handling. Here's how Komputers makes it simple:

```rust
use komputers::*;

fn robust_gpu_computing() -> Result<()> {
    let mut manager = match Manager::new() {
        Ok(m) => {
            println!("✅ GPU initialized: {}", m.device_name()?);
            m
        },
        Err(e) => {
            eprintln!("❌ GPU initialization failed: {}", e);
            eprintln!("💡 Tip: Check Vulkan drivers are installed");
            return Err(e);
        }
    };

    let tensor = match Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("❌ Tensor creation failed: {}", e);
            return Err(e);
        }
    };

    match manager
        .sequence()?
        .record_tensor_sync_device(tensor.handle())?
        .execute()
    {
        Ok(result) => {
            println!("✅ Success: {} operations in {:?}",
                    result.operation_count, result.execution_time);
        },
        Err(e) => {
            eprintln!("❌ GPU operation failed: {}", e);
            eprintln!("💡 This gives you exact context about what went wrong");
        }
    }

    Ok(())
}
```

## 🏃‍♂️ Step 5: Next Steps (30 seconds)

Congratulations! You've just experienced the future of type-safe GPU computing. Here's where to go next:

### 📚 Learn More

- **[API Reference](API_REFERENCE.md)** - Complete documentation with examples
- **[Migration Guide](MIGRATION_GUIDE.md)** - Upgrade from unsafe GPU patterns
- **[Best Practices](BEST_PRACTICES.md)** - Advanced usage patterns

### 🔬 Explore Features

- **[Async GPU Operations](examples/async_gpu_operations.rs)** - Non-blocking GPU computing
- **[Performance Profiling](examples/performance_monitoring.rs)** - AI-driven optimization
- **[Workgroup Optimization](examples/workgroup_optimization.rs)** - Advanced GPU tuning
- **[Complex Workflows](examples/complex_workflows.rs)** - Real-world applications

### 🚀 Advanced Patterns

**Async GPU Computing:**

```rust
use komputers::*;

async fn async_example() -> Result<()> {
    let mut manager = AsyncManager::new()?;
    let tensor = manager.create_tensor(&[1.0f32, 2.0]).await?;

    let result = manager
        .async_sequence()?
        .record_tensor_sync_device(tensor.handle())?
        .execute_async()
        .await?;

    println!("Async GPU operation: {:?}", result.execution_time);
    Ok(())
}
```

**Performance Monitoring:**

```rust
use komputers::performance_metrics::*;

fn optimized_example() -> Result<()> {
    let mut profiler = global_profiler().lock().unwrap();
    profiler.start_session("optimization")?;

    // Your GPU operations...

    let report = profiler.end_session()?;
    for rec in report.recommendations {
        println!("💡 {}", rec.description);
    }
    Ok(())
}
```

## 🎉 You Did It!

In just 5 minutes, you've:

- ✅ Set up type-safe GPU computing
- ✅ Seen compile-time safety prevent crashes
- ✅ Run actual GPU operations safely
- ✅ Handled errors gracefully
- ✅ Discovered advanced features

**You're now ready to build impossible-to-crash GPU applications!**

## 🆘 Troubleshooting

**GPU not detected?**

```bash
vulkaninfo  # Check Vulkan setup
```

**Compilation errors?**

- Ensure Rust 1.70+: `rustc --version`
- Update dependencies: `cargo update`

**Runtime errors?**

- Check GPU drivers are up to date
- Verify Vulkan is properly installed

**Need help?**

- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Join [GitHub Discussions](https://github.com/axiomantic/komputers/discussions)
- File an [Issue](https://github.com/axiomantic/komputers/issues) with your error

---

**Ready to revolutionize your GPU computing? The future is type-safe! 🚀**

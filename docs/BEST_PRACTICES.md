# Best Practices

This guide covers recommended patterns for using Komputers effectively.

## GPU Resource Management

### Manager Lifecycle

Create managers early and keep them for the duration of GPU work:

```rust
fn main() -> komputers::Result<()> {
    // Create once, use throughout the program
    let mut manager = Manager::new()?;

    // Perform all GPU operations
    do_gpu_work(&mut manager)?;

    // Manager automatically cleans up on drop
    Ok(())
}
```

### Tensor Management

Keep tensor lifetimes manageable:

```rust
// Good: Clear ownership
let tensor = Tensor::new(&mut manager, &data)?;
let result = process_tensor(&tensor)?;

// Consider: Large tensors use significant GPU memory
let large_tensor = Tensor::new(&mut manager, &large_data)?;
// Use it immediately, then let it drop
let processed = algorithm.compute(&large_tensor)?;
// large_tensor memory is freed here
```

## Error Handling Patterns

### Structured Error Handling

Use pattern matching for different error types:

```rust
match manager.sequence()?.execute() {
    Ok(result) => {
        println!("Operations completed: {}", result.operation_count);
    }
    Err(KomputeError::GpuInitialization(msg)) => {
        eprintln!("GPU initialization failed: {}", msg);
        // Handle GPU setup issues
    }
    Err(KomputeError::Memory(msg)) => {
        eprintln!("Memory error: {}", msg);
        // Handle memory allocation issues
    }
    Err(other) => {
        eprintln!("Other error: {}", other);
    }
}
```

### Resource Cleanup

Resources clean up automatically, but you can be explicit:

```rust
{
    let tensor = Tensor::new(&mut manager, &data)?;
    // Use tensor
    let result = compute(&tensor)?;
    // tensor is automatically cleaned up here
}
// GPU memory is now available
```

## Performance Considerations

### Memory Transfer

Minimize host-GPU transfers:

```rust
// Less efficient: Multiple transfers
let a = Tensor::new(&mut manager, &data_a)?;
let b = Tensor::new(&mut manager, &data_b)?;

// More efficient: Batch operations
let mut sequence = manager.sequence()?;
sequence.record_tensor_sync_device(a.handle())?;
sequence.record_tensor_sync_device(b.handle())?;
sequence.record_algorithm(&algorithm)?;
sequence.record_tensor_sync_local(result.handle())?;
sequence.execute()?;
```

### Workgroup Sizing

Use appropriate workgroup sizes for your data:

```rust
// For 1D data
let workgroup = WorkgroupConfig::new(64, 1, 1); // Good for most GPUs

// For 2D data
let workgroup = WorkgroupConfig::new(8, 8, 1); // 64 threads total

// For 3D data
let workgroup = WorkgroupConfig::new(4, 4, 4); // 64 threads total
```

## Cross-Platform Development

### Vulkan Setup

Let Komputers handle Vulkan setup automatically:

```rust
// Komputers handles platform differences automatically
let manager = Manager::new()?;

// For custom setup, use the setup utilities
use komputers::vulkan_setup::configure_vulkan_environment;
configure_vulkan_environment()?;
```

### Platform-Specific Considerations

**macOS (MoltenVK):**

- Automatic MoltenVK detection and configuration
- Set `KOMPUTE_DISABLE_MOLTENVK_AUTO=1` to disable auto-setup if needed

**Linux:**

- Ensure Vulkan drivers are installed for your GPU
- Ubuntu/Debian: `apt install vulkan-tools libvulkan-dev`

**Windows:**

- Install Vulkan SDK from LunarG
- GPU drivers typically include Vulkan support

## Testing Strategies

### Unit Testing

Test GPU operations with actual hardware when available:

```rust
#[test]
fn test_tensor_operations() -> komputers::Result<()> {
    let mut manager = Manager::new()?;
    let tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;

    // Test tensor properties
    assert_eq!(tensor.size(), 3);
    assert_eq!(tensor.data_type()?, DataType::Float32);

    Ok(())
}
```

### Integration Testing

Test complete workflows:

```rust
#[test]
fn test_complete_workflow() -> komputers::Result<()> {
    let mut manager = Manager::new()?;

    // Create test data
    let input = Tensor::new(&mut manager, &[1.0f32; 100])?;
    let mut output = Tensor::uninit::<f32>(&mut manager, 100)?;

    // Load and execute algorithm
    let spirv = include_bytes!("../shaders/test.spv");
    let algorithm = Algorithm::with_tensors(&mut manager, spirv, &[&input, &output])?;

    let mut sequence = manager.sequence()?;
    sequence.record_algorithm(&algorithm)?;
    let result = sequence.execute()?;

    // Verify results
    assert!(result.execution_time.as_nanos() > 0);

    Ok(())
}
```

## Common Patterns

### Batch Processing

Process multiple datasets efficiently:

```rust
fn batch_process(manager: &mut Manager, datasets: &[Vec<f32>]) -> komputers::Result<Vec<Vec<f32>>> {
    let mut results = Vec::new();

    for dataset in datasets {
        let input = Tensor::new(manager, dataset)?;
        let mut output = Tensor::uninit::<f32>(manager, dataset.len())?;

        // Process dataset
        let result = process_single(&input, &mut output)?;
        results.push(result);
    }

    Ok(results)
}
```

### Algorithm Reuse

Reuse algorithms with different tensor data:

```rust
fn reuse_algorithm_pattern() -> komputers::Result<()> {
    let mut manager = Manager::new()?;

    // Create algorithm once
    let spirv = include_bytes!("shaders/process.spv");
    let algorithm = Algorithm::new(&mut manager, spirv)?;

    // Use with different tensors
    for data in &[vec1, vec2, vec3] {
        let input = Tensor::new(&mut manager, data)?;
        let mut output = Tensor::uninit::<f32>(&mut manager, data.len())?;

        let mut sequence = manager.sequence()?;
        sequence.record_tensor_sync_device(input.handle())?;
        sequence.record_algorithm_with_tensors(&algorithm, &[&input, &output])?;
        sequence.record_tensor_sync_local(output.handle())?;
        sequence.execute()?;

        // Use output data
        let result = output.to_vec()?;
        println!("Result: {:?}", result);
    }

    Ok(())
}
```

## Debugging

### Error Context

Komputers provides context in error messages:

```rust
// Error messages include helpful context
match operation() {
    Err(KomputeError::Memory(msg)) => {
        // msg contains details about what memory operation failed
        eprintln!("Memory error details: {}", msg);
    }
    _ => {}
}
```

### Validation Features

Use debug features during development:

```toml
[dependencies]
komputers = { version = "0.1", features = ["debug", "validation"] }
```

### GPU Diagnostics

Check GPU availability and capabilities:

```rust
use komputers::vulkan_setup::diagnose_vulkan_environment;

let report = diagnose_vulkan_environment()?;
println!("GPU Status: {}", report.status);
println!("Available devices: {}", report.device_count);
```

## Hybrid Validation System Best Practices

### Understanding the Hybrid Validation System

Komputers v0.4.0+ introduces a revolutionary hybrid validation system that combines compile-time type safety with runtime validation to prevent cross-manager errors and GPU crashes.

#### Validation Layers

**Compile-Time Safety (80%+ of errors)**:
```rust
use komputers::types::{SyncManagerType, ManagerId};

// Type-safe managers with unique IDs
let manager_a = create_sync_manager::<1>()?;  // Unique compile-time ID
let manager_b = create_sync_manager::<2>()?;  // Different compile-time ID

let tensor_a = manager_a.create_tensor(&data)?;

// This won't compile - caught at build time!
// manager_b.sequence()?.record_tensor_sync_device(tensor_a.handle())?; // ❌ Compile error
```

**Runtime Validation (100% coverage of remaining scenarios)**:
```rust
// When using runtime managers, validation catches errors at runtime
let mut manager_a = Manager::new()?;
let mut manager_b = Manager::new()?;

let tensor_a = Tensor::new(&mut manager_a, &data)?;

// Runtime validation catches this with helpful error messages
match manager_b.sequence()?.record_tensor_sync_device(tensor_a.handle()) {
    Ok(_) => unreachable!("Cross-manager usage should be prevented"),
    Err(KomputeError::TypeSafety(e)) => {
        eprintln!("Cross-manager error caught: {}", e);
        // Handle gracefully instead of crashing
    }
}
```

### Safe Cross-Manager Usage Patterns

#### Pattern 1: Single Manager Architecture (Recommended)

**Best Practice**: Use a single manager per application or logical GPU context:

```rust
struct GpuApplication {
    manager: Manager<SyncManagerType<ManagerId<1>>>,
    // All GPU resources created with this manager
}

impl GpuApplication {
    fn new() -> Result<Self> {
        Ok(Self {
            manager: create_sync_manager::<1>()?,
        })
    }

    fn process_data(&mut self, data: &[f32]) -> Result<Vec<f32>> {
        let input = self.manager.create_tensor(data)?;
        let mut output = self.manager.create_tensor_uninit::<f32>(data.len())?;

        // All operations use the same manager - guaranteed safe
        self.manager.sequence()?
            .record_tensor_sync_device(input.typed_handle())?
            .record_tensor_sync_device(output.typed_handle())?
            .record_tensor_sync_local(output.typed_handle())?
            .execute()?;

        output.to_vec()
    }
}
```

#### Pattern 2: Multiple Managers with Isolation

When you need multiple managers, ensure complete isolation:

```rust
struct MultiGpuProcessor {
    compute_manager: Manager<SyncManagerType<ManagerId<1>>>,
    transfer_manager: Manager<SyncManagerType<ManagerId<2>>>,
}

impl MultiGpuProcessor {
    fn process_with_isolation(&mut self, data: &[f32]) -> Result<Vec<f32>> {
        // Create tensors with their respective managers
        let compute_tensor = self.compute_manager.create_tensor(data)?;
        let transfer_tensor = self.transfer_manager.create_tensor(data)?;

        // Each manager operates only on its own tensors
        self.compute_manager.sequence()?
            .record_tensor_sync_device(compute_tensor.typed_handle())?  // Safe
            .execute()?;

        self.transfer_manager.sequence()?
            .record_tensor_sync_device(transfer_tensor.typed_handle())?  // Safe
            .execute()?;

        // Won't compile - cross-manager usage prevented at compile time
        // self.compute_manager.sequence()?.record_tensor_sync_device(transfer_tensor.handle())?; // ❌

        Ok(vec![])
    }
}
```

#### Pattern 3: Manager Lifecycle Management

**Best Practice**: Manager lifetimes should encompass all related GPU resources:

```rust
fn safe_manager_lifecycle() -> Result<()> {
    // Manager created first
    let mut manager = create_sync_manager::<1>()?;

    // All resources created after manager
    let tensor_a = manager.create_tensor(&[1.0f32, 2.0, 3.0])?;
    let tensor_b = manager.create_tensor(&[4.0f32, 5.0, 6.0])?;
    let algorithm = manager.create_algorithm(&spirv_code)?;

    // Use all resources while manager is alive
    manager.sequence()?
        .record_tensor_sync_device(tensor_a.typed_handle())?
        .record_algorithm_with_tensors(&algorithm, &[&tensor_a, &tensor_b])?
        .record_tensor_sync_local(tensor_b.typed_handle())?
        .execute()?;

    // All resources cleaned up when manager goes out of scope
    Ok(())
} // ✅ Safe: manager outlives all resources
```

**Avoid**: Resources outliving their manager:

```rust
// ❌ Anti-pattern: Handle outlives manager
fn dangerous_handle_lifetime() -> Result<u64> {
    let handle = {
        let mut manager = Manager::new()?;
        let tensor = Tensor::new(&mut manager, &[1.0f32])?;
        tensor.handle() // Handle escapes manager scope
    }; // Manager destroyed here

    // Using handle after manager destruction - runtime error!
    Ok(handle) // ❌ Invalid handle
}
```

### Async/Sync Safety Patterns

#### Async-Safe Operations

```rust
async fn async_safe_processing() -> Result<()> {
    let mut async_manager = create_async_manager::<1>().await?;
    let tensor = async_manager.create_tensor(&data).await?;

    let result = AsyncSequence::new(&mut async_manager)?
        .record_tensor_sync_device(tensor.typed_handle())?
        .execute_async()
        .await?;

    println!("Async execution completed in {:?}", result.execution_time);
    Ok(())
}
```

#### Mixed Async/Sync with Type Safety

```rust
struct HybridProcessor {
    sync_manager: Manager<SyncManagerType<ManagerId<1>>>,
    async_manager: Manager<AsyncManagerType<ManagerId<2>>>,
}

impl HybridProcessor {
    async fn process_mixed_workload(&mut self, data: &[f32]) -> Result<()> {
        // Sync processing
        let sync_tensor = self.sync_manager.create_tensor(data)?;
        self.sync_manager.sequence()?
            .record_tensor_sync_device(sync_tensor.typed_handle())?
            .execute()?;

        // Async processing
        let async_tensor = self.async_manager.create_tensor(data).await?;
        AsyncSequence::new(&mut self.async_manager)?
            .record_tensor_sync_device(async_tensor.typed_handle())?
            .execute_async()
            .await?;

        // Won't compile - async/sync type mismatch
        // self.sync_manager.sequence()?.record_tensor_sync_device(async_tensor.handle())?; // ❌

        Ok(())
    }
}
```

### Error Handling Best Practices

#### Comprehensive Error Handling

```rust
use komputers::{KomputeError, TypeSafetyError, ValidationError};

fn robust_error_handling() -> Result<()> {
    let mut manager = Manager::new()?;
    let tensor = Tensor::new(&mut manager, &data)?;

    match manager.sequence()?.record_tensor_sync_device(tensor.handle())?.execute() {
        Ok(result) => {
            println!("Success: {} operations in {:?}",
                result.operation_count, result.execution_time);
        }

        // Type safety errors from hybrid validation
        Err(KomputeError::TypeSafety(TypeSafetyError::CrossManagerUsage { expected, actual })) => {
            eprintln!("Cross-manager usage: expected manager {}, got {}", expected, actual);
            // Implement recovery or fail gracefully
        }

        Err(KomputeError::TypeSafety(TypeSafetyError::InvalidHandle { handle_id })) => {
            eprintln!("Invalid handle {}: manager may have been destroyed", handle_id);
            // Recreate resources or fail gracefully
        }

        Err(KomputeError::TypeSafety(TypeSafetyError::AsyncSyncMismatch { operation, handle_type })) => {
            eprintln!("Async/sync mismatch: {} operation with {} handle", operation, handle_type);
            // Use correct async/sync pattern
        }

        // GPU hardware errors
        Err(KomputeError::GpuInitialization(msg)) => {
            eprintln!("GPU initialization failed: {}", msg);
            // Fall back to CPU processing
        }

        Err(KomputeError::Memory(msg)) => {
            eprintln!("GPU memory error: {}", msg);
            // Reduce memory usage or batch operations
        }

        Err(other) => {
            eprintln!("Unexpected error: {}", other);
            return Err(other);
        }
    }

    Ok(())
}
```

#### Graceful Degradation

```rust
use komputers::runtime_validation::{is_validation_enabled, ValidationMetrics};

fn adaptive_processing(data: &[f32]) -> Result<Vec<f32>> {
    // Check validation status for performance-critical paths
    if is_validation_enabled() {
        println!("Runtime validation active - maximum safety mode");
    }

    let mut manager = Manager::new()?;
    let tensor = Tensor::new(&mut manager, data)?;

    // Attempt GPU processing with error recovery
    match gpu_process(&mut manager, &tensor) {
        Ok(result) => Ok(result),

        Err(KomputeError::TypeSafety(e)) => {
            eprintln!("Type safety error prevented crash: {}", e);
            // Fall back to safe processing
            safe_fallback_process(&mut manager, data)
        }

        Err(KomputeError::DeviceError(_)) => {
            eprintln!("GPU device error, falling back to CPU");
            cpu_fallback_process(data)
        }

        Err(e) => Err(e),
    }
}
```

### Validation Configuration

#### Production Configuration

```rust
use komputers::runtime_validation::{
    enable_validation,
    disable_validation,
    get_validation_metrics
};

fn configure_for_production() {
    // Runtime validation is ON by default (safety-first)
    // Only disable if performance is absolutely critical

    // Monitor validation performance
    let metrics = get_validation_metrics();
    if metrics.average_validation_time.as_nanos() > 1000 {
        println!("High validation overhead detected: {:?}",
            metrics.average_validation_time);
    }

    // For ultra-high-performance scenarios only (⚠️ reduces safety)
    if std::env::var("ULTRA_PERFORMANCE_MODE").is_ok() {
        eprintln!("⚠️ WARNING: Disabling runtime validation for performance");
        eprintln!("⚠️ This reduces safety - use with extreme caution!");
        disable_validation();
    }
}
```

#### Development Configuration

```rust
fn configure_for_development() {
    // Maximum validation in development
    enable_validation();

    // Enable debug logging
    std::env::set_var("KOMPUTERS_VALIDATION_LOG", "debug");
    std::env::set_var("RUST_LOG", "komputers=debug");

    // Initialize logging
    env_logger::init();
}
```

### Performance Best Practices

#### Validation-Aware Performance

```rust
use std::time::Instant;

fn performance_aware_processing(data: &[f32]) -> Result<Vec<f32>> {
    let start = Instant::now();

    // Use compile-time safety for maximum performance
    let mut manager = create_sync_manager::<1>()?;
    let tensor = manager.create_tensor(data)?;  // Type-safe creation

    // Batch operations to amortize validation costs
    let result = manager.sequence()?
        .record_tensor_sync_device(tensor.typed_handle())?  // Compile-time safe
        .record_tensor_sync_local(tensor.typed_handle())?
        .execute()?;

    let total_time = start.elapsed();
    let validation_overhead = total_time.saturating_sub(result.execution_time);

    if validation_overhead > Duration::from_millis(1) {
        println!("Validation overhead: {:?} ({}% of total)",
            validation_overhead,
            (validation_overhead.as_nanos() * 100 / total_time.as_nanos()));
    }

    Ok(vec![])
}
```

#### Handle Reuse Patterns

```rust
struct OptimizedProcessor {
    manager: Manager<SyncManagerType<ManagerId<1>>>,
    reusable_tensors: Vec<Tensor<f32>>,
}

impl OptimizedProcessor {
    fn process_batch(&mut self, datasets: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();

        for (i, data) in datasets.iter().enumerate() {
            // Reuse tensors when possible
            let tensor = if i < self.reusable_tensors.len() {
                &mut self.reusable_tensors[i]
            } else {
                let new_tensor = self.manager.create_tensor(data)?;
                self.reusable_tensors.push(new_tensor);
                self.reusable_tensors.last_mut().unwrap()
            };

            // Validation is cached for repeated handle usage
            self.manager.sequence()?
                .record_tensor_sync_device(tensor.typed_handle())?  // Cached validation
                .execute()?;

            results.push(vec![]); // Placeholder result
        }

        Ok(results)
    }
}
```

## Anti-Patterns

### Avoid These Patterns

**Don't disable validation without understanding the trade-offs:**

```rust
// ❌ Bad: Disabling validation without justification
disable_validation();  // Reduces safety for unclear benefit

// ✅ Good: Keep validation enabled by default
// Only disable for measured performance gains in production
if measurements_show_unacceptable_overhead() && is_production() {
    eprintln!("⚠️ Disabling validation due to measured performance impact");
    disable_validation();
}
```

**Don't ignore type safety errors:**

```rust
// ❌ Bad: Silencing type safety errors
match operation() {
    Err(KomputeError::TypeSafety(_)) => {}, // Ignores safety violations
    result => result,
}?;

// ✅ Good: Handle type safety errors appropriately
match operation() {
    Err(KomputeError::TypeSafety(e)) => {
        eprintln!("Type safety violation: {}", e);
        // Implement proper error handling or recovery
        return Err(KomputeError::TypeSafety(e));
    }
    result => result,
}?;
```

**Don't mix manager instances carelessly:**

```rust
// ❌ Bad: Unclear manager relationships
struct ConfusedProcessor {
    managers: Vec<Manager>,  // Unclear which handles belong to which manager
}

// ✅ Good: Clear manager relationships
struct ClearProcessor {
    compute_manager: Manager<SyncManagerType<ManagerId<1>>>,  // For computation
    io_manager: Manager<SyncManagerType<ManagerId<2>>>,       // For I/O
    // Clear separation of concerns
}
```

**Don't create unnecessary managers:**

```rust
// ❌ Bad: Multiple managers for same purpose
let manager1 = Manager::new()?;
let manager2 = Manager::new()?; // Unnecessary for single GPU context

// ✅ Good: Single manager for coherent operations
let mut manager = Manager::new()?;
// Use for all related GPU operations
```

**Don't assume GPU availability:**

```rust
// ❌ Bad: Assume GPU is always available
let manager = Manager::new().unwrap();

// ✅ Good: Handle GPU unavailability gracefully
let manager = match Manager::new() {
    Ok(m) => m,
    Err(e) => {
        eprintln!("GPU not available: {}", e);
        return fallback_cpu_processing();
    }
};
```

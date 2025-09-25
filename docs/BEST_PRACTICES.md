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

## Anti-Patterns

### Avoid These Patterns

**Don't create multiple managers unnecessarily:**

```rust
// Bad: Multiple managers for same GPU
let manager1 = Manager::new()?;
let manager2 = Manager::new()?; // Unnecessary

// Good: One manager, multiple operations
let mut manager = Manager::new()?;
// Use manager for all operations
```

**Don't ignore error context:**

```rust
// Bad: Generic error handling
operation().unwrap(); // Loses error information

// Good: Handle errors appropriately
match operation() {
    Ok(result) => { /* use result */ }
    Err(e) => eprintln!("Operation failed: {}", e),
}
```

**Don't assume GPU availability:**

```rust
// Bad: Assume GPU is always available
let manager = Manager::new().unwrap();

// Good: Handle GPU unavailability
let manager = match Manager::new() {
    Ok(m) => m,
    Err(e) => {
        eprintln!("GPU not available: {}", e);
        return; // Fall back to CPU processing
    }
};
```

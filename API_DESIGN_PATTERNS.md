# API Design Patterns - Kompute Rust Implementation

**Purpose:** Reference patterns for implementing Kompute APIs safely and idiomatically in Rust
**Extracted from:** Comprehensive API analysis and C++ integration patterns

---

## Core Design Principles

### 1. Handle-Based Resource Management

```rust
// Pattern: Opaque handles for GPU resources
pub struct Tensor<T> {
    handle: u64,
    size: usize,
    _phantom: PhantomData<T>,
}

// Pattern: RAII cleanup
impl<T> Drop for Tensor<T> {
    fn drop(&mut self) {
        ffi::tensor_destroy_handle(self.handle);
    }
}

// Pattern: Thread safety markers
unsafe impl<T: Send> Send for Tensor<T> {}
unsafe impl<T: Sync> Sync for Tensor<T> {}
```

### 2. Type-Safe FFI Boundary

```rust
// Pattern: Type validation at boundaries
pub trait TensorData: Copy + Default + 'static {
    fn data_type_id() -> u32;
    fn size_of() -> usize;
}

// Pattern: C++ function signature validation
#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        fn create_tensor_f32_handle(manager: Pin<&mut Manager>, data: &[f32]) -> u64;
        fn tensor_data_f32_by_handle(handle: u64) -> Vec<f32>;
    }
}

// Pattern: Safe Rust wrapper
impl Tensor<f32> {
    pub fn new(manager: &mut Manager, data: &[f32]) -> Result<Self> {
        let handle = ffi::create_tensor_f32_handle(manager.pin_mut(), data);
        if handle == 0 {
            return Err(KomputeError::tensor("Failed to create tensor"));
        }
        Ok(Self { handle, size: data.len(), _phantom: PhantomData })
    }
}
```

### 3. Comprehensive Error Handling

```rust
// Pattern: Hierarchical error types
#[derive(Debug, thiserror::Error)]
pub enum KomputeError {
    #[error("Vulkan error: {0}")]
    VulkanError(String),
    #[error("Device error: {0}")]
    DeviceError(String),
    #[error("Tensor error: {0}")]
    TensorError(String),
    #[error("Algorithm error: {0}")]
    AlgorithmError(String),
    // ... other variants
}

// Pattern: Context-preserving error chain
impl KomputeError {
    pub fn with_context<T: std::fmt::Display>(self, context: T) -> Self {
        match self {
            Self::TensorError(msg) => Self::TensorError(format!("{}: {}", context, msg)),
            // ... other variants
        }
    }
}

// Pattern: Result type alias
pub type Result<T> = std::result::Result<T, KomputeError>;
```

### 4. Builder Pattern for Complex Objects

```rust
// Pattern: Fluent configuration API
pub struct AlgorithmBuilder<'a> {
    manager: &'a mut Manager,
    tensors: Vec<&'a Tensor<f32>>,
    spirv: Option<&'a [u32]>,
    workgroup: Workgroup,
    spec_constants: Vec<f32>,
    push_constants: Vec<f32>,
}

impl<'a> AlgorithmBuilder<'a> {
    pub fn new(manager: &'a mut Manager) -> Self {
        Self {
            manager,
            tensors: Vec::new(),
            spirv: None,
            workgroup: Workgroup::default(),
            spec_constants: Vec::new(),
            push_constants: Vec::new(),
        }
    }

    pub fn with_tensors(mut self, tensors: &'a [&'a Tensor<f32>]) -> Self {
        self.tensors.extend(tensors);
        self
    }

    pub fn with_spirv(mut self, spirv: &'a [u32]) -> Self {
        self.spirv = Some(spirv);
        self
    }

    pub fn with_workgroup(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroup = Workgroup { x, y, z };
        self
    }

    pub fn build(self) -> Result<Algorithm> {
        let spirv = self.spirv.ok_or_else(|| KomputeError::algorithm("SPIR-V code required"))?;
        Algorithm::create(self.manager, &self.tensors, spirv, self.workgroup,
                         &self.spec_constants, &self.push_constants)
    }
}
```

### 5. Async Integration Pattern

```rust
// Pattern: Future-based GPU operations
pub struct AsyncSequenceExecution {
    sequence: Sequence,
    timeout: Duration,
    start_time: Instant,
    started: bool,
}

impl Future for AsyncSequenceExecution {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.started {
            self.sequence.eval_async()?;
            self.started = true;
            self.start_time = Instant::now();
        }

        // Check timeout
        if self.start_time.elapsed() >= self.timeout {
            return Poll::Ready(Err(KomputeError::compute("GPU operation timed out")));
        }

        // Non-blocking poll
        match self.sequence.eval_await(1) { // 1ms timeout
            Ok(()) => Poll::Ready(Ok(())),
            Err(_) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}
```

---

## C++ Integration Patterns

### 1. Exception Translation

```cpp
// Pattern: C++ exception boundaries
uint64_t create_tensor_f32_handle(Manager& manager, rust::Slice<const float> data) {
    try {
        // Input validation
        if (data.empty()) {
            throw std::invalid_argument("Cannot create tensor with empty data");
        }

        // Safe operation
        auto tensor = manager.tensor(std::vector<float>(data.begin(), data.end()));
        if (!tensor) {
            throw std::runtime_error("Manager returned null tensor");
        }

        // Registration
        return get_global_registry().register_tensor(tensor);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to create tensor: ") + e.what());
    }
}
```

### 2. Memory Safety Guards

```cpp
// Pattern: Input validation at all boundaries
namespace validation {
    inline bool is_valid_handle(uint64_t handle) noexcept {
        return handle != 0 && handle <= std::numeric_limits<uint64_t>::max() - 1;
    }

    inline bool is_safe_size(size_t size, size_t max_size) noexcept {
        return size > 0 && size <= max_size;
    }
}

// Pattern: RAII with automatic cleanup
class HandleRegistry {
    std::unordered_map<uint64_t, std::shared_ptr<Tensor>> tensor_registry_;
    std::mutex registry_mutex_;

public:
    ~HandleRegistry() {
        // Automatic cleanup of all resources
        std::lock_guard<std::mutex> lock(registry_mutex_);
        tensor_registry_.clear();
    }
};
```

### 3. Thread Safety Patterns

```cpp
// Pattern: Thread-safe singleton
HandleRegistry& get_global_registry() {
    static HandleRegistry instance;
    return instance;
}

// Pattern: Lock-based synchronization
std::shared_ptr<Tensor> get_tensor(uint64_t handle) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = tensor_registry_.find(handle);
    return (it != tensor_registry_.end()) ? it->second : nullptr;
}
```

---

## Testing Patterns

### 1. GPU-Optional Testing

```rust
// Pattern: Graceful test degradation
#[test]
fn test_device_detection() {
    match Manager::device_count() {
        Ok(count) if count > 0 => {
            // GPU available - run full test
            let manager = Manager::new().expect("Should create manager with GPU");
            // ... full GPU test
        },
        Ok(0) => {
            println!("No GPU devices available - test passed (expected in CI)");
        },
        Err(e) => {
            println!("Device enumeration failed - test passed (expected in headless): {}", e);
        }
    }
}

#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_gpu_operations() {
    // Explicit GPU requirement
    let mut manager = Manager::new().expect("This test requires GPU");
    // ... GPU-specific tests
}
```

### 2. Error Condition Testing

```rust
// Pattern: Comprehensive error testing
#[test]
fn test_error_conditions() {
    // Test invalid inputs
    match Manager::with_device(999) {
        Ok(_) => panic!("Should fail with invalid device ID"),
        Err(e) => assert!(e.to_string().contains("device")),
    }

    // Test resource limits
    if let Ok(mut manager) = Manager::new() {
        let huge_size = usize::MAX / 1000;
        match Tensor::<f32>::uninit(&mut manager, huge_size) {
            Ok(_) => println!("Unexpectedly succeeded with huge tensor"),
            Err(e) => assert!(e.to_string().contains("size") || e.to_string().contains("memory")),
        }
    }
}
```

### 3. Property-Based Testing

```rust
// Pattern: Property validation across inputs
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_tensor_data_consistency(data in prop::collection::vec(any::<f32>(), 1..1000)) {
        if let Ok(mut manager) = Manager::new() {
            if let Ok(tensor) = Tensor::new(&mut manager, &data) {
                if let Ok(retrieved) = tensor.to_vec() {
                    // Property: Data round-trip should be exact
                    prop_assert_eq!(data.len(), retrieved.len());
                    for (orig, retr) in data.iter().zip(retrieved.iter()) {
                        if orig.is_nan() {
                            prop_assert!(retr.is_nan());
                        } else {
                            prop_assert_eq!(orig, retr);
                        }
                    }
                }
            }
        }
    }
}
```

---

## Performance Patterns

### 1. Zero-Copy Operations

```rust
// Pattern: Minimize allocations in hot paths
impl<T: TensorData> Tensor<T> {
    pub fn as_slice(&self) -> Result<&[T]> {
        // Direct memory access without copying
        unsafe {
            let ptr = ffi::tensor_data_ptr_by_handle(self.handle) as *const T;
            Ok(std::slice::from_raw_parts(ptr, self.size))
        }
    }
}
```

### 2. Batch Operations

```rust
// Pattern: Batch multiple operations for efficiency
impl Sequence {
    pub fn record_tensor_sync_batch(&mut self, tensors: &[&Tensor<f32>]) -> Result<()> {
        for tensor in tensors {
            self.record_tensor_sync_device(tensor.handle())?;
        }
        Ok(())
    }
}
```

---

## Summary

These patterns ensure:

- **Memory Safety**: Handle-based resources with RAII cleanup
- **Type Safety**: Compile-time validation of GPU operations
- **Error Handling**: Comprehensive error context and recovery
- **Performance**: Zero-copy operations where possible
- **Testability**: GPU-optional testing with error condition coverage
- **Thread Safety**: Proper synchronization for concurrent access

Use these patterns consistently across all API implementations to maintain quality and safety standards.

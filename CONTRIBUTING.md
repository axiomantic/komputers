# How to Contribute to Kompute-rs

**Version:** 1.0
**Project Phase:** Foundation & Core Development

## 1. The Philosophy of Development

Contributing to Kompute-rs is a disciplined practice of creating high-quality, maintainable software through systematic development and rigorous validation. Our workflow ensures quality, safety, and consistent progress through structured implementation.

Our development process is built upon key artifacts, each serving a critical purpose:

- **The Canon (`SYSTEM_ARCHITECTURE.md`)**: The architectural constitution and single source of truth defining the system design and principles
- **The Implementation Guide (`GRIMOIRE.md`)**: Universal template for implementing components following established patterns
- **The Progress Tracker (`STATUS_MANIFEST.yaml`)**: Live tracking of all component status, dependencies, and progress
- **The Contribution Guidelines (This Document)**: The exact workflow and standards for all development work
- **The Sacred Tests**: Comprehensive validation ensuring quality and preventing regressions

## 2. The Sacred Laws of Development

All development follows these inviolable principles:

### 2.1. The Law of Quality
**"Excellence is the minimum standard"**
- All code must meet established quality gates
- All implementations must follow established patterns
- All components must integrate cleanly with the system
- All work must advance project objectives

### 2.2. The Law of Verification
**"Trust nothing without evidence"**
- All functionality must be tested comprehensively
- All claims must be proven with concrete evidence
- All implementations must be validated against requirements
- All changes must be verified through quality gates

### 2.3. The Law of Progress
**"Every change advances the project"**
- Every contribution must move the project forward
- No work should introduce technical debt without justification
- All implementations must maintain or improve system quality
- Dependencies must be respected and completed in order

### 2.4. The Law of Documentation
**"Undocumented work doesn't exist"**
- All components must be properly documented
- All decisions must be recorded with rationale
- All interfaces must be clearly specified
- All complex logic must be explained

## 3. The Development Workflow

All development follows this precise ritual for implementing components:

### Step 1: Component Selection and Summoning
- Examine `STATUS_MANIFEST.yaml` to identify components with status `PLANNED`
- Verify all dependencies are `COMPLETED`
- **Summon the component**: Request "summon [component-id]" which will:
  - Generate a detailed implementation plan
  - **Immediately implement the component completely**
  - Update status to `USER_REVIEW` when done

### Step 2: Manual Implementation (Alternative Path)
If you prefer manual implementation instead of summoning, perform these preparatory steps:

**2.1. Architectural Alignment**
- Review relevant sections of `SYSTEM_ARCHITECTURE.md` to understand the component's role
- Identify integration points with existing components
- Understand established patterns and conventions

**2.2. Dependency Verification**
- Confirm all prerequisite components are `COMPLETED`
- Verify no circular dependencies exist
- Check that required resources are available

**2.3. Test Strategy Formation**
- Define success criteria from requirements
- Plan comprehensive test coverage
- Design integration scenarios
- Establish performance benchmarks

### Step 3: Test-Driven Implementation

**The Three Rings of Validation**: All components must pass through three concentric rings of testing:

#### The Inner Ring (Unit Tests)
**Purpose**: Validate individual component behavior in isolation

1. **Write Failing Tests First**: Create comprehensive unit tests that define expected behavior
2. **Implement Minimum Code**: Write only enough code to make tests pass
3. **Refactor for Quality**: Improve code structure while maintaining passing tests
4. **Coverage Requirements**: Achieve minimum 80% coverage

Example:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_with_valid_data() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&data).expect("Failed to create tensor");
        assert_eq!(tensor.size(), 4);
        assert_eq!(tensor.data(), &data);
    }

    #[test]
    fn test_tensor_gpu_allocation_failure_handling() {
        // Test error handling for GPU memory allocation failures
        // This should fail gracefully with appropriate error message
    }
}
```

#### The Middle Ring (Integration Tests)
**Purpose**: Validate component interactions and workflows

1. **Test Real Scenarios**: Create tests that simulate actual usage patterns
2. **No Mocking**: Test against real implementations where possible
3. **Validate Workflows**: Ensure complete user journeys work correctly
4. **Performance Testing**: Verify performance meets requirements

Example:
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_gpu_compute_workflow() {
        let manager = Manager::new().expect("Failed to create manager");
        let tensor_a = manager.tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let tensor_b = manager.tensor(&[4.0f32, 5.0, 6.0]).unwrap();

        let algorithm = manager.algorithm("shader.spv", &[&tensor_a, &tensor_b]).unwrap();
        let mut sequence = manager.sequence().unwrap();

        sequence.record(&algorithm);
        sequence.eval();

        // Verify results match expected GPU computation
        assert_eq!(tensor_b.data(), &[5.0f32, 7.0, 9.0]);
    }
}
```

#### The Outer Ring (Quality Gates)
**Purpose**: Ensure code quality and maintainability

1. **Static Analysis**: Pass all linting and style checks
2. **Type Safety**: Ensure proper typing throughout
3. **Security Scanning**: Address any security vulnerabilities
4. **Documentation**: Complete all documentation requirements

Quality Gate Commands:
```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo bench
```

### Step 4: Evidence Collection
Document comprehensive evidence of successful implementation:

**4.1. Test Evidence**
```bash
# Unit test evidence
cargo test --lib
# All tests must pass

# Integration test evidence
cargo test --tests
# All workflows must succeed

# Coverage evidence
cargo tarpaulin --out html
# Must meet minimum coverage requirements
```

**4.2. Quality Evidence**
```bash
# Code quality evidence
cargo fmt --check && cargo clippy -- -D warnings
# All quality gates must pass
```

**4.3. Performance Evidence**
```
Benchmark Results:
- Tensor allocation: 0.05ms (target: <0.1ms)
- GPU memory transfer: 1.2GB/s (target: >1GB/s)
- Compute dispatch: 0.02ms overhead (target: <5%)
```

### Step 5: Completion and Review

**5.1. Update Status to USER_REVIEW**
```yaml
# STATUS_MANIFEST.yaml
ffi-layer:
  status: USER_REVIEW  # Changed from IN_PROGRESS
  completion_date: "2024-01-15"
  evidence_provided: true
  test_coverage: "85%"
  quality_gates: "PASSED"
  review_notes: "Ready for human approval - all evidence provided"
```

**5.2. Request Human Review**
- Notify human reviewer that component is complete
- Provide summary of implementation and key achievements
- Component remains in `USER_REVIEW` until approved
- Human updates to `COMPLETED` when satisfied

**5.3. Version Control** (After approval)
- Create feature branch: `feature/component-description`
- Commit with clear, descriptive messages
- Open pull request with comprehensive description
- Include all evidence in PR description

## 4. Component Implementation Standards

### 4.1. Code Standards

**Quality Requirements**:
```rust
// All public APIs must be documented
/// Creates a new GPU tensor with the provided data
///
/// # Arguments
/// * `data` - Host data to upload to GPU memory
///
/// # Returns
/// * `Result<Tensor<T>, KomputeError>` - Tensor or allocation error
///
/// # Example
/// ```rust
/// let tensor = Tensor::new(&[1.0f32, 2.0, 3.0])?;
/// ```
pub fn new<T: Copy + Default>(data: &[T]) -> Result<Tensor<T>, KomputeError> {
    // Implementation with proper error handling
}
```

**Error Handling Requirements**:
```rust
#[derive(Debug, thiserror::Error)]
pub enum KomputeError {
    #[error("GPU device not found: {0}")]
    DeviceNotFound(String),

    #[error("Memory allocation failed: {size} bytes")]
    AllocationFailed { size: usize },

    #[error("Vulkan API error: {0}")]
    VulkanError(String),
}

// All operations return Results
pub fn allocate_memory(size: usize) -> Result<GpuMemory, KomputeError> {
    // Implementation with explicit error handling
}
```

**Documentation Requirements**:
```rust
/// GPU compute manager providing device access and resource allocation
///
/// The `Manager` struct serves as the primary interface for GPU operations,
/// managing Vulkan device lifecycle and coordinating compute operations.
///
/// # Thread Safety
/// `Manager` is Send + Sync and can be shared across threads safely.
///
/// # Resource Management
/// All resources are automatically cleaned up through RAII patterns.
pub struct Manager {
    // Internal implementation details
}
```

### 4.2. Testing Standards

**Test Structure**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::*; // Common test utilities

    mod unit_tests {
        use super::*;

        #[test]
        fn test_specific_behavior() {
            // Arrange
            let input = setup_test_data();

            // Act
            let result = component.operation(input);

            // Assert
            assert!(result.is_ok());
            verify_expected_behavior(result.unwrap());
        }
    }

    mod integration_tests {
        use super::*;

        #[test]
        fn test_complete_workflow() {
            // Test full user scenarios
        }
    }
}
```

**Test Naming Conventions**:
- Unit tests: `test_component_specific_behavior`
- Integration tests: `test_integration_workflow`
- Performance tests: `test_performance_operation`

### 4.3. Documentation Standards

**Component Documentation**:
- Purpose and responsibilities
- Interface specifications
- Usage examples
- Integration guidelines
- Performance characteristics

## 5. The Conflict Resolution Protocol

**The Canon is Law**: `SYSTEM_ARCHITECTURE.md` is the ultimate source of truth.

### When Conflicts Arise:

1. **STOP**: Immediately halt all implementation work
2. **DOCUMENT**: Create a `CONFLICT` notice stating:
   - The architectural section in conflict
   - The conflicting requirement or instruction
   - Suggested resolution approaches
3. **RESOLVE**: Work with team to update Canon and resolve conflict
4. **PROCEED**: Continue only after conflict is resolved

## 6. Quality Gates and Requirements

### 6.1. Code Quality Gates
- **Linting**: All code must pass cargo clippy with no errors
- **Formatting**: All code must be formatted with cargo fmt
- **Type Checking**: All code must pass cargo check
- **Security**: No critical security vulnerabilities

### 6.2. Testing Requirements
- **Unit Test Coverage**: Minimum 80%
- **Integration Coverage**: All critical paths tested
- **Performance**: Meets established benchmarks
- **Regression**: No existing tests broken

### 6.3. Documentation Requirements
- **Code Comments**: All complex logic explained
- **API Documentation**: All public interfaces documented
- **Examples**: Usage examples for key features
- **Architecture**: Component fits documented architecture

## 7. Emergency Protocols

### 7.1. Build Failure Protocol
If builds fail after changes:
1. Revert changes immediately
2. Fix issues in isolated environment
3. Re-test comprehensively before resubmitting

### 7.2. Performance Regression Protocol
If performance degrades:
1. Profile to identify bottlenecks
2. Compare against baseline metrics
3. Optimize without compromising functionality
4. Document performance analysis

### 7.3. Security Issue Protocol
If security vulnerabilities are found:
1. **CRITICAL**: Fix immediately, notify team
2. **HIGH**: Fix within current iteration
3. **MEDIUM/LOW**: Document and schedule fix

## 8. Contribution Workflow Summary

```
1. SELECT component (STATUS_MANIFEST.yaml) → Update to IN_PROGRESS
2. PLAN implementation → Review architecture, verify dependencies
3. TEST FIRST → Write comprehensive failing tests
4. IMPLEMENT → Write minimum code to pass tests
5. REFACTOR → Improve quality while maintaining tests
6. VALIDATE → Pass all quality gates and requirements
7. DOCUMENT → Complete all documentation
8. REVIEW → Update to USER_REVIEW, await approval
9. COMPLETE → Human approval updates to COMPLETED
```

## 9. Status Workflow

### Status Progression:
```
PLANNED → IN_PROGRESS → USER_REVIEW → COMPLETED
    ↑                                       ↓
    └──────── (if revisions needed) ────────┘
```

### Status Definitions:
- **PLANNED**: Component identified but not started
- **IN_PROGRESS**: Component actively being implemented
- **USER_REVIEW**: Implementation complete, awaiting review
- **COMPLETED**: Implementation reviewed and approved

## 10. Best Practices

### 10.1. Before Starting
- Review all relevant documentation
- Understand component dependencies
- Plan comprehensive test strategy
- Allocate sufficient time for quality

### 10.2. During Development
- Follow TDD rigorously
- Commit frequently with clear messages
- Run tests continuously
- Keep documentation updated

### 10.3. Before Completion
- Verify all tests pass
- Check coverage meets requirements
- Ensure documentation is complete
- Validate against requirements

## 11. Getting Help

If you encounter issues:
1. Check existing documentation
2. Review similar completed components
3. Consult architectural guidelines
4. Request clarification before proceeding

---

**Excellence in software development comes from disciplined process, rigorous testing, and unwavering commitment to quality. Follow these guidelines to contribute effectively to Kompute-rs.**

*"Quality is not an act, it is a habit."*
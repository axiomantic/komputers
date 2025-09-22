# Run all pre-commit checks
pre-commit:
    pre-commit run --all-files

# Format code
fmt:
    cargo fmt --all

# Lint with clippy (strict)
lint:
    cargo clippy --all-targets --features "async,glsl,debug,validation,test-utils" -- -D warnings

# Check compilation
check:
    cargo check --all-targets --features "async,glsl,debug,validation,test-utils"

# Run tests (excludes benchmarks for faster builds)
test:
    cargo test --features "async,glsl,debug,validation,test-utils" --workspace

# Run benchmarks (requires criterion compilation)
bench:
    cargo test --features "benchmarks" --benches

# Install pre-commit hooks (if using pre-commit)
install-hooks:
    pre-commit install

# Clean up unused dependencies
machete:
    cargo machete

# Security audit
audit:
    cargo audit

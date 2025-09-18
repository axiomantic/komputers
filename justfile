# Run all pre-commit checks
pre-commit:
    pre-commit run --all-files

# Format code
fmt:
    cargo fmt --all

# Lint with clippy (strict)
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Check compilation
check:
    cargo check --all-targets --all-features

# Run tests
test:
    cargo test --all-features

# Install pre-commit hooks (if using pre-commit)
install-hooks:
    pre-commit install

# Clean up unused dependencies
machete:
    cargo machete

# Security audit
audit:
    cargo audit

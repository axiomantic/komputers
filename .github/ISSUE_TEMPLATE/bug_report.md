---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: ["bug"]
assignees: ""
---

## Bug Description

**Brief Summary**
A clear and concise description of what the bug is.

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened instead.

## Environment Information

**Platform:**

- OS: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- Architecture: [e.g., x86_64, aarch64]
- Rust Version: [e.g., 1.70.0]
- Komputers Version: [e.g., 0.1.0]

**GPU Information:**

- GPU: [e.g., Apple M1, NVIDIA RTX 4080, AMD RX 7800]
- Vulkan Version: [e.g., 1.3.246]
- Driver Version: [if known]

**Vulkan Environment:**

- `vulkaninfo` output (first 20 lines):

```
[Paste vulkaninfo output here]
```

## Reproduction Steps

**Minimal Reproducible Example:**

```rust
// Paste your minimal code that reproduces the issue
use kompute::*;

fn main() -> Result<()> {
    // Your reproduction code here
    Ok(())
}
```

**Steps to Reproduce:**

1. [First step]
2. [Second step]
3. [Third step]
4. [Error occurs]

**Cargo Command Used:**

```bash
cargo run --example my_example --features async
```

## Error Information

**Error Output:**

```
[Paste the complete error message, including stack trace if available]
```

**Build Output (if build fails):**

```
[Paste cargo build output]
```

**Log Output (if available):**

```bash
RUST_LOG=debug cargo run --example my_example 2>&1 | head -50
```

## Additional Context

**Frequency:**

- [ ] Happens every time
- [ ] Happens intermittently
- [ ] Happened once

**Workarounds:**

- [ ] Found a workaround (describe below)
- [ ] No workaround found

**Related Issues:**

- Link to any related issues or discussions

**Additional Information:**
Add any other context about the problem here, such as:

- What you were trying to accomplish
- Whether this worked in previous versions
- Any special configuration or setup
- Screenshots or videos if relevant

## Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have included all relevant environment information
- [ ] I have provided a minimal reproducible example
- [ ] I have included the complete error output
- [ ] I have tested with the latest version of Komputers

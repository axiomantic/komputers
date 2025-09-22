# Self-Hosted GPU Runner Setup Guide

**Version:** 2.0
**Purpose:** Guide for setting up self-hosted GitHub Actions runners with GPU support for Komputers CI/CD
**Context:** Enables real GPU hardware testing in automated CI/CD pipeline

---

## Overview

Self-hosted runners with GPU hardware enable comprehensive testing that standard GitHub Actions runners cannot provide. This guide covers GPU-specific setup requirements for testing Komputers.

## Quick Setup Summary

1. **Install GitHub Runner**: Follow the [official GitHub documentation](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
2. **Install GPU drivers and Vulkan** (platform-specific instructions below)
3. **Configure runner with GPU labels** (provided for each platform)
4. **Test GPU functionality** with validation scripts

## GitHub Runner Setup

**For complete GitHub Actions runner installation instructions, follow the official documentation:**

👉 **[GitHub Self-Hosted Runners Guide](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)**

The official guide covers:

- Downloading and configuring runners
- Platform-specific installation steps
- Service setup and management
- Security considerations
- Runner maintenance

## GPU-Specific Requirements

This section covers additional GPU and Vulkan setup required for Komputers testing.

## Why Self-Hosted GPU Runners?

### **Standard CI/CD Limitations:**

- ❌ No GPU hardware in cloud runners
- ❌ No Vulkan drivers in containers
- ❌ Limited graphics capabilities
- ❌ Cannot test real GPU operations

### **Self-Hosted Benefits:**

- ✅ Real GPU hardware (NVIDIA, AMD, Apple Silicon)
- ✅ Full Vulkan driver support
- ✅ Complete MoltenVK testing on macOS
- ✅ Custom GPU configurations
- ✅ Performance benchmarking with real hardware
- ✅ Memory leak detection with actual GPU memory

---

## Platform-Specific GPU Setup

### 🍎 **macOS with Apple Silicon** (Recommended Starting Point)

#### **Hardware Requirements:**

- Apple Silicon Mac (M1, M2, M3, M4)
- macOS 12.0+ (Monterey or later)
- 16GB+ RAM (32GB recommended for CI workloads)

#### **Vulkan/MoltenVK Setup:**

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install MoltenVK and dependencies
brew install vulkan-headers vulkan-loader molten-vk cmake

# Set up environment variables (add to ~/.zshrc)
echo 'export VULKAN_SDK=/opt/homebrew' >> ~/.zshrc
echo 'export VK_ICD_FILENAMES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json' >> ~/.zshrc
echo 'export VK_DRIVER_FILES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json' >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc

# Verify Vulkan is working
vulkaninfo --summary
```

#### **Recommended Runner Labels:**

Configure your runner with these labels: `self-hosted,macOS,ARM64,gpu,vulkan,metal`

#### **GPU Validation Script:**

Create `~/github-runner/validate-gpu.sh`:

```bash
#!/bin/bash
set -e

echo "🍎 macOS GPU Validation for CI/CD"
echo "=================================="

# Check Metal support
echo "🔍 Checking Metal GPU support..."
system_profiler SPDisplaysDataType | grep -A 5 "Chipset Model" || echo "⚠️ No discrete GPU info"

# Check MoltenVK
echo "🔍 Checking MoltenVK setup..."
if [ -f "/opt/homebrew/lib/libMoltenVK.dylib" ]; then
    echo "✅ MoltenVK library found"
else
    echo "❌ MoltenVK library not found"
    exit 1
fi

# Check Vulkan
echo "🔍 Checking Vulkan functionality..."
vulkaninfo --summary > vulkan_info.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Vulkan is operational"
    grep -i "device name\|driver version" vulkan_info.txt
else
    echo "❌ Vulkan validation failed"
    cat vulkan_info.txt
    exit 1
fi

# Test Komputers compilation
echo "🔍 Testing Komputers build..."
cd /path/to/komputers/clone  # Adjust path
cargo build --release --features kompute-master

# Test basic GPU operation
echo "🔍 Testing basic GPU operations..."
timeout 30 cargo run --example basic_gpu_computing 2>&1 | head -20

echo "✅ All GPU validation checks passed!"
```

```bash
chmod +x ~/github-runner/validate-gpu.sh
```

### 🐧 **Linux with NVIDIA GPU**

#### **Hardware Requirements:**

- NVIDIA GPU (GTX 1060+ or RTX series recommended)
- Ubuntu 20.04+ or equivalent
- 16GB+ RAM

#### **NVIDIA Driver and Vulkan Setup:**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (choose appropriate version)
sudo apt install -y nvidia-driver-535  # or latest stable

# Install Vulkan development packages
sudo apt install -y vulkan-tools libvulkan-dev vulkan-validationlayers-dev \
    mesa-vulkan-drivers cmake build-essential

# Reboot to activate drivers
sudo reboot

# Verify installation
nvidia-smi
vulkaninfo --summary
```

#### **Recommended Runner Labels:**

Configure your runner with these labels: `self-hosted,Linux,X64,gpu,vulkan,nvidia`

### 🪟 **Windows with GPU Support**

#### **GPU Driver and Vulkan Setup:**

```powershell
# Install Chocolatey (package manager)
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install development tools
choco install git cmake rust-ms visualstudio2022buildtools -y

# Install GPU drivers:
# NVIDIA: Download latest drivers from nvidia.com
# AMD: Download latest drivers from amd.com
# Intel: Download latest drivers from intel.com

# Install Vulkan SDK
# Download from: https://vulkan.lunarg.com/sdk/home#windows

# Verify installation
vulkaninfo.exe
```

#### **Recommended Runner Labels:**

Configure your runner with these labels: `self-hosted,Windows,X64,gpu,vulkan`

---

## Workflow Integration

### **Example GPU Testing Workflow**

Add GPU-specific jobs to your workflows:

```yaml
# .github/workflows/gpu-integration.yml
name: GPU Hardware Testing

on:
  push:
    branches: [main]

jobs:
  gpu-macos:
    name: GPU Testing (macOS Apple Silicon)
    runs-on: [self-hosted, macOS, ARM64, gpu, vulkan, metal]
    steps:
      - uses: actions/checkout@v4

      - name: Run GPU Integration Tests
        run: |
          cargo test --test gpu_compute_integration --features kompute-master -- --nocapture
          cargo test --test cross_platform_integration -- --nocapture

  gpu-linux-nvidia:
    name: GPU Testing (Linux NVIDIA)
    runs-on: [self-hosted, Linux, X64, gpu, vulkan, nvidia]
    if: ${{ github.repository_owner == 'axiomantic' }}
    steps:
      - uses: actions/checkout@v4

      - name: Run NVIDIA-Specific Tests
        run: |
          cargo test --test gpu_vendor_compatibility_matrix --features kompute-master -- --nocapture
```

## Common Issues and Troubleshooting

### **GPU Tests Failing:**

```bash
# Verify GPU drivers and Vulkan
nvidia-smi  # NVIDIA
vulkaninfo --summary  # All platforms

# Test basic GPU operations
cargo run --example basic_gpu_computing --features kompute-master
```

### **Runner Not Connecting:**

Refer to the [GitHub troubleshooting guide](https://docs.github.com/en/actions/hosting-your-own-runners/monitoring-and-troubleshooting-self-hosted-runners/troubleshooting-self-hosted-runners).

## Security Considerations

- Use dedicated user accounts for runners
- Limit runner access with appropriate firewall rules
- Rotate runner tokens regularly
- Restrict GPU runners to your repository: `if: ${{ github.repository == 'axiomantic/komputers' }}`

---

**For complete setup and security details, always refer to the [official GitHub documentation](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners).**

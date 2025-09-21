# Self-Hosted GPU Runner Setup Guide

**Version:** 1.0
**Purpose:** Complete guide for setting up self-hosted GitHub Actions runners with GPU support for Komputers CI/CD
**Context:** Enables real GPU hardware testing in automated CI/CD pipeline

---

## Overview

Self-hosted runners with GPU hardware enable comprehensive testing that standard GitHub Actions runners cannot provide. This guide covers setup for real GPU testing across different platforms and vendors.

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

## Platform-Specific Setup Guides

### 🍎 **macOS with Apple Silicon** (Recommended Starting Point)

#### **Hardware Requirements:**

- Apple Silicon Mac (M1, M2, M3, M4)
- macOS 12.0+ (Monterey or later)
- 16GB+ RAM (32GB recommended for CI workloads)
- 100GB+ available storage

#### **Software Prerequisites:**

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install essential dependencies
brew install vulkan-headers vulkan-loader molten-vk cmake git

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### **Vulkan/MoltenVK Setup:**

```bash
# Verify MoltenVK installation
ls -la /opt/homebrew/lib/libMoltenVK.dylib

# Set up environment variables (add to ~/.zshrc)
echo 'export VULKAN_SDK=/opt/homebrew' >> ~/.zshrc
echo 'export VK_ICD_FILENAMES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json' >> ~/.zshrc
echo 'export VK_DRIVER_FILES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json' >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc

# Verify Vulkan is working
vulkaninfo --summary
```

#### **GitHub Runner Installation:**

```bash
# Create runner directory
mkdir -p ~/github-runner && cd ~/github-runner

# Download latest runner (check GitHub for current version)
RUNNER_VERSION="2.319.1"  # Update to latest version
curl -o actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz \
  -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz

# Extract and verify
tar xzf actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz
./bin/installdependencies.sh

# Configure runner (get token from GitHub repo settings)
./config.sh --url https://github.com/axiomantic/komputers --token YOUR_TOKEN_HERE \
  --name "macos-gpu-runner" \
  --labels "self-hosted,macOS,ARM64,gpu,vulkan,metal" \
  --work _work
```

#### **Service Setup for Always-On Operation:**

```bash
# Install as macOS service
sudo ./svc.sh install

# Start the service
sudo ./svc.sh start

# Check status
sudo ./svc.sh status
```

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
- 100GB+ available storage

#### **NVIDIA Driver Setup:**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (choose appropriate version)
sudo apt install -y nvidia-driver-535  # or latest stable

# Install NVIDIA Container Toolkit (optional, for containerized workflows)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Reboot to activate drivers
sudo reboot
```

#### **Vulkan Setup:**

```bash
# Install Vulkan development packages
sudo apt install -y vulkan-tools libvulkan-dev vulkan-validationlayers-dev \
    mesa-vulkan-drivers cmake build-essential

# Verify installation
nvidia-smi
vulkaninfo --summary
```

#### **GitHub Runner Setup:**

```bash
# Create dedicated user for runner
sudo useradd -m -s /bin/bash github-runner
sudo usermod -aG docker github-runner  # if using Docker

# Switch to runner user
sudo su - github-runner

# Download and configure runner (similar to macOS, but for Linux)
mkdir -p ~/actions-runner && cd ~/actions-runner

RUNNER_VERSION="2.319.1"
curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
  -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

tar xzf actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
sudo ./bin/installdependencies.sh

# Configure with GPU labels
./config.sh --url https://github.com/axiomantic/komputers --token YOUR_TOKEN_HERE \
  --name "linux-nvidia-gpu-runner" \
  --labels "self-hosted,Linux,X64,gpu,vulkan,nvidia" \
  --work _work
```

#### **Service Setup:**

```bash
# Install as systemd service
sudo ./svc.sh install github-runner

# Start and enable service
sudo systemctl start actions.runner.github-runner.linux-nvidia-gpu-runner.service
sudo systemctl enable actions.runner.github-runner.linux-nvidia-gpu-runner.service

# Check status
sudo systemctl status actions.runner.github-runner.linux-nvidia-gpu-runner.service
```

### 🪟 **Windows with GPU Support**

#### **Prerequisites:**

```powershell
# Install Chocolatey (package manager)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install essential tools
choco install git cmake rust-ms visualstudio2022buildtools -y

# Install Vulkan SDK (download from LunarG)
# https://vulkan.lunarg.com/sdk/home#windows
```

#### **GPU Driver Setup:**

```powershell
# NVIDIA: Download latest drivers from nvidia.com
# AMD: Download latest drivers from amd.com
# Intel: Download latest drivers from intel.com

# Verify installation
vulkaninfo.exe
```

#### **GitHub Runner Setup:**

```powershell
# Create runner directory
New-Item -ItemType Directory -Path "C:\actions-runner" -Force
Set-Location "C:\actions-runner"

# Download runner
$RUNNER_VERSION = "2.319.1"
Invoke-WebRequest -Uri "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/actions-runner-win-x64-$RUNNER_VERSION.zip" -OutFile "actions-runner-win-x64-$RUNNER_VERSION.zip"

# Extract
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("actions-runner-win-x64-$RUNNER_VERSION.zip", ".")

# Configure
.\config.cmd --url https://github.com/axiomantic/komputers --token YOUR_TOKEN_HERE `
  --name "windows-gpu-runner" `
  --labels "self-hosted,Windows,X64,gpu,vulkan" `
  --work "_work"

# Install as Windows service
.\svc.cmd install
.\svc.cmd start
```

---

## Workflow Integration

### **Update Existing Workflows**

Add GPU-specific jobs to your workflows:

```yaml
# .github/workflows/gpu-integration.yml
name: GPU Hardware Testing

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 6 * * *" # Daily GPU tests

jobs:
  gpu-macos:
    name: GPU Testing (macOS Apple Silicon)
    runs-on: [self-hosted, macOS, ARM64, gpu, vulkan, metal]
    steps:
      - uses: actions/checkout@v4

      - name: Validate GPU Environment
        run: |
          ~/github-runner/validate-gpu.sh

      - name: Build with GPU Features
        run: |
          cargo build --all-features --release

      - name: Run GPU Integration Tests
        run: |
          # Only run if GPU is available
          cargo test --test gpu_compute_integration --features kompute-master -- --nocapture
          cargo test --test cross_platform_integration -- --nocapture

      - name: Run Performance Benchmarks
        run: |
          cargo bench --features kompute-master -- --output-format json | tee gpu_bench_results.json

      - name: Upload GPU Test Results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: gpu-test-results-macos
          path: |
            gpu_bench_results.json
            target/criterion/
          retention-days: 7

  gpu-linux-nvidia:
    name: GPU Testing (Linux NVIDIA)
    runs-on: [self-hosted, Linux, X64, gpu, vulkan, nvidia]
    if: ${{ github.repository_owner == 'axiomantic' }} # Only for main repo
    steps:
      - uses: actions/checkout@v4

      - name: Validate NVIDIA GPU
        run: |
          nvidia-smi
          vulkaninfo --summary

      - name: Run NVIDIA-Specific Tests
        run: |
          cargo test --test gpu_vendor_compatibility_matrix --features kompute-master -- --nocapture
```

### **Conditional GPU Testing**

```yaml
# Only run GPU tests if runners are available
gpu-tests:
  name: Conditional GPU Testing
  runs-on: ubuntu-latest
  steps:
    - name: Check for GPU Runners
      id: check-runners
      run: |
        # Logic to check if GPU runners are online
        if [[ "${{ github.repository_owner }}" == "axiomantic" ]]; then
          echo "gpu-available=true" >> $GITHUB_OUTPUT
        else
          echo "gpu-available=false" >> $GITHUB_OUTPUT
        fi

  gpu-integration:
    needs: gpu-tests
    if: needs.gpu-tests.outputs.gpu-available == 'true'
    runs-on: [self-hosted, gpu]
    steps:
    # GPU test steps here
```

---

## Security Considerations

### **Runner Security Best Practices:**

```bash
# 1. Dedicated user account
sudo useradd -m -s /bin/bash github-runner
sudo passwd github-runner  # Set strong password

# 2. Limited sudo access (create /etc/sudoers.d/github-runner)
echo "github-runner ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart actions.runner*" | sudo tee /etc/sudoers.d/github-runner

# 3. Firewall configuration
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh  # Only if needed

# 4. Regular updates
sudo apt update && sudo apt upgrade -y  # Linux
brew update && brew upgrade  # macOS
```

### **Repository Access Control:**

```yaml
# Limit GPU runners to specific repositories
if: ${{ github.repository == 'axiomantic/komputers' }}
```

### **Secrets Management:**

- Use GitHub repository secrets for sensitive data
- Never store GPU runner tokens in version control
- Rotate runner tokens regularly
- Use environment-specific configurations

---

## Monitoring and Maintenance

### **Health Check Script:**

Create `~/github-runner/health-check.sh`:

```bash
#!/bin/bash

echo "🏥 GPU Runner Health Check"
echo "========================="

# Check runner service
if systemctl is-active --quiet actions.runner.*; then
    echo "✅ Runner service is active"
else
    echo "❌ Runner service is not active"
    sudo systemctl restart actions.runner.*
fi

# Check GPU availability
if nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA GPU available"
elif system_profiler SPDisplaysDataType | grep -i metal >/dev/null 2>&1; then
    echo "✅ Apple Metal GPU available"
else
    echo "⚠️ GPU availability unclear"
fi

# Check Vulkan
if vulkaninfo --summary >/dev/null 2>&1; then
    echo "✅ Vulkan operational"
else
    echo "❌ Vulkan not working"
fi

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "⚠️ Disk usage high: ${DISK_USAGE}%"
    # Clean up old builds
    cd ~/actions-runner/_work && find . -name "target" -type d -mtime +7 -exec rm -rf {} +
else
    echo "✅ Disk usage normal: ${DISK_USAGE}%"
fi

echo "Health check complete!"
```

### **Automated Maintenance:**

```bash
# Add to crontab for automated maintenance
crontab -e

# Add these lines:
# Health check every hour
0 * * * * ~/github-runner/health-check.sh >> ~/github-runner/health.log 2>&1

# Clean old logs weekly
0 2 * * 0 find ~/github-runner/_work -name "*.log" -mtime +14 -delete

# Update system packages monthly
0 3 1 * * sudo apt update && sudo apt upgrade -y  # Linux
0 3 1 * * brew update && brew upgrade  # macOS
```

---

## Troubleshooting

### **Common Issues:**

#### **Runner Not Connecting:**

```bash
# Check network connectivity
curl -I https://github.com

# Verify token is valid
./config.sh --help

# Check service logs
sudo journalctl -u actions.runner.* -f  # Linux
sudo launchctl list | grep runner  # macOS
```

#### **GPU Tests Failing:**

```bash
# Verify GPU drivers
nvidia-smi  # NVIDIA
vulkaninfo --summary  # All platforms

# Check Vulkan environment
echo $VULKAN_SDK
echo $VK_ICD_FILENAMES

# Test basic GPU operations
cargo run --example basic_gpu_computing --features kompute-master
```

#### **Performance Issues:**

```bash
# Monitor resource usage
htop  # Linux/macOS
taskmgr  # Windows

# Check GPU utilization
nvidia-smi -l 1  # NVIDIA
sudo powermetrics -s gpu_power -n 1  # macOS

# Limit concurrent jobs if needed
# Edit runner configuration to reduce parallel builds
```

### **Log Collection:**

```bash
# Collect comprehensive logs for support
mkdir -p ~/gpu-runner-logs
cp ~/github-runner/_diag/* ~/gpu-runner-logs/
vulkaninfo > ~/gpu-runner-logs/vulkan-info.txt
nvidia-smi > ~/gpu-runner-logs/nvidia-info.txt 2>/dev/null || true
system_profiler SPDisplaysDataType > ~/gpu-runner-logs/macos-gpu-info.txt 2>/dev/null || true
```

---

## Cost Considerations

### **Hardware Costs:**

**macOS Apple Silicon (Recommended):**

- Mac Mini M2: $599-999
- Mac Studio M2: $1,999-3,999
- Power consumption: 15-100W

**Linux NVIDIA:**

- Budget build: $800-1,200 (GTX 1660 Super + basic components)
- Performance build: $1,500-3,000 (RTX 4070+ + components)
- Power consumption: 200-500W

**Windows Setup:**

- Similar to Linux costs
- Windows license: $139-199

### **Operational Costs:**

- **Electricity**: $20-100/month depending on usage and local rates
- **Internet**: Stable broadband required
- **Maintenance**: ~2-4 hours/month for updates and monitoring

### **ROI Considerations:**

- **Cloud GPU alternatives**: $0.50-3.00/hour for GPU instances
- **Self-hosted breakeven**: Typically 3-6 months for active projects
- **Benefits beyond cost**: Real hardware testing, custom configurations, always available

---

## Next Steps

### **Getting Started Checklist:**

- [ ] Choose platform (macOS Apple Silicon recommended for starting)
- [ ] Acquire hardware with appropriate GPU
- [ ] Install and configure operating system
- [ ] Set up GPU drivers and Vulkan
- [ ] Install GitHub runner with GPU labels
- [ ] Create validation scripts
- [ ] Test with Komputers repository
- [ ] Set up monitoring and maintenance
- [ ] Document your specific configuration

### **Future Enhancements:**

- [ ] Multi-runner load balancing
- [ ] Automated runner provisioning
- [ ] GPU vendor matrix testing
- [ ] Performance regression tracking
- [ ] Cost optimization with spot instances

---

**This guide provides the foundation for professional GPU testing infrastructure that will significantly enhance Komputers CI/CD capabilities and code quality assurance.**

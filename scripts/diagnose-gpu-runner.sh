#!/bin/bash
#
# GPU Runner Diagnostic Script
# Comprehensive assessment of local GPU runner capabilities
#

set -euo pipefail

echo "🔍 KOMPUTERS GPU RUNNER DIAGNOSTIC REPORT"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Script Version: 1.0"
echo ""

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
    PACKAGE_MANAGER="brew"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
    if command -v apt >/dev/null 2>&1; then
        PACKAGE_MANAGER="apt"
    elif command -v yum >/dev/null 2>&1; then
        PACKAGE_MANAGER="yum"
    else
        PACKAGE_MANAGER="unknown"
    fi
else
    PLATFORM="Windows"
    PACKAGE_MANAGER="winget/chocolatey"
fi

echo "🖥️  PLATFORM INFORMATION"
echo "========================"
echo "Platform: $PLATFORM"
echo "Package Manager: $PACKAGE_MANAGER"
echo "OS Details: $OSTYPE"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo ""

# Hardware Detection
echo "🔧 HARDWARE DETECTION"
echo "====================="
case $PLATFORM in
    "macOS")
        echo "System Profile:"
        system_profiler SPDisplaysDataType | head -20
        echo ""
        echo "Metal Support:"
        system_profiler SPDisplaysDataType | grep -i "metal" || echo "No explicit Metal support found"
        ;;
    "Linux")
        echo "GPU Hardware:"
        lspci | grep -i "vga\|3d\|display" || echo "No GPU detected via lspci"
        echo ""
        echo "NVIDIA Detection:"
        nvidia-smi 2>/dev/null || echo "No NVIDIA GPU or drivers found"
        ;;
    "Windows")
        echo "GPU Hardware Detection requires PowerShell - please run manually"
        ;;
esac
echo ""

# Vulkan SDK Detection
echo "🌋 VULKAN SDK DETECTION"
echo "======================"
echo "Vulkan Environment Variables:"
echo "  VULKAN_SDK: ${VULKAN_SDK:-not set}"
echo "  VK_LAYER_PATH: ${VK_LAYER_PATH:-not set}"
echo "  VK_ICD_FILENAMES: ${VK_ICD_FILENAMES:-not set}"
echo ""

echo "Vulkan Tools:"
if command -v vulkaninfo >/dev/null 2>&1; then
    echo "✅ vulkaninfo found"
    vulkaninfo --summary || echo "❌ vulkaninfo failed to run"
else
    echo "❌ vulkaninfo not found"
fi
echo ""

if command -v vulkan-layers >/dev/null 2>&1; then
    echo "✅ vulkan-layers found"
else
    echo "❌ vulkan-layers not found"
fi
echo ""

# Build Tools Detection
echo "🔨 BUILD TOOLS DETECTION"
echo "========================"
echo "Rust:"
if command -v rustc >/dev/null 2>&1; then
    echo "✅ rustc: $(rustc --version)"
    echo "✅ cargo: $(cargo --version)"

    # Check for required components
    echo ""
    echo "Rust Components:"
    if rustup component list --installed | grep -q rustfmt; then
        echo "✅ rustfmt installed"
    else
        echo "❌ rustfmt not installed"
    fi

    if rustup component list --installed | grep -q clippy; then
        echo "✅ clippy installed"
    else
        echo "❌ clippy not installed"
    fi
else
    echo "❌ Rust not found"
fi
echo ""

echo "CMake:"
if command -v cmake >/dev/null 2>&1; then
    echo "✅ cmake: $(cmake --version | head -1)"
else
    echo "❌ cmake not found"
fi
echo ""

echo "Compiler:"
if command -v gcc >/dev/null 2>&1; then
    echo "✅ gcc: $(gcc --version | head -1)"
elif command -v clang >/dev/null 2>&1; then
    echo "✅ clang: $(clang --version | head -1)"
else
    echo "❌ No C/C++ compiler found"
fi
echo ""

# Test Current Build
echo "🧪 KOMPUTERS BUILD TEST"
echo "======================="
echo "Testing current build capability..."

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]] || ! grep -q "komputers" Cargo.toml; then
    echo "❌ Not in komputers project directory"
    echo "Please run this script from the komputers project root"
    exit 1
fi

echo "✅ In komputers project directory"
echo ""

# Test basic compilation
echo "Testing basic compilation (no GPU required):"
if KOMPUTE_DISABLE_VULKAN_CHECK=1 cargo check --features "async,glsl,debug,validation,test-utils" 2>/dev/null; then
    echo "✅ Basic compilation successful"
else
    echo "❌ Basic compilation failed"
    echo "Running with verbose output:"
    KOMPUTE_DISABLE_VULKAN_CHECK=1 cargo check --features "async,glsl,debug,validation,test-utils" --verbose 2>&1 | tail -20
fi
echo ""

# Test GPU detection
echo "Testing GPU detection:"
if timeout 30 cargo test test_gpu_detection --features "async,glsl,debug,validation,test-utils" 2>/dev/null; then
    echo "✅ GPU detection test passed"
    GPU_AVAILABLE="yes"
else
    echo "❌ GPU detection test failed"
    GPU_AVAILABLE="no"
    echo "This may indicate GPU hardware or driver issues"
fi
echo ""

# Environment Analysis
echo "🌍 ENVIRONMENT ANALYSIS"
echo "======================"
echo "Current Komputers Environment Variables:"
echo "  KOMPUTE_DISABLE_VULKAN_CHECK: ${KOMPUTE_DISABLE_VULKAN_CHECK:-not set}"
echo "  KOMPUTE_VERBOSE_DETECTION: ${KOMPUTE_VERBOSE_DETECTION:-not set}"
echo "  KOMPUTE_DISABLE_ASAN: ${KOMPUTE_DISABLE_ASAN:-not set}"
echo "  RUSTC_WRAPPER: ${RUSTC_WRAPPER:-not set}"
echo ""

echo "GitHub Actions Environment (if applicable):"
echo "  GITHUB_ACTIONS: ${GITHUB_ACTIONS:-not set}"
echo "  RUNNER_NAME: ${RUNNER_NAME:-not set}"
echo "  RUNNER_OS: ${RUNNER_OS:-not set}"
echo ""

# CI Configuration Analysis
echo "📋 CI CONFIGURATION ANALYSIS"
echo "============================"
if [[ -f ".github/workflows/ci.yml" ]]; then
    echo "✅ Found .github/workflows/ci.yml"
    echo ""
    echo "Self-hosted runner configuration:"
    grep -A 5 -B 5 "self-hosted" .github/workflows/ci.yml || echo "No self-hosted configuration found"
    echo ""

    echo "GPU test job status:"
    if grep -q "gpu-tests-self-hosted:" .github/workflows/ci.yml; then
        echo "✅ GPU tests job found"

        # Check if Linux/Windows runners are commented out
        if grep -q "#.*gpu-tests-self-hosted-linux:" .github/workflows/ci.yml; then
            echo "⚠️  Linux GPU runner is commented out"
        fi

        if grep -q "#.*gpu-tests-self-hosted-windows:" .github/workflows/ci.yml; then
            echo "⚠️  Windows GPU runner is commented out"
        fi
    else
        echo "❌ No GPU tests job found"
    fi
else
    echo "❌ No CI configuration found"
fi
echo ""

# Generate Recommendations
echo "🎯 DIAGNOSTIC RECOMMENDATIONS"
echo "============================="

# Platform-specific recommendations
case $PLATFORM in
    "macOS")
        if ! command -v vulkaninfo >/dev/null 2>&1; then
            echo "❌ Install Vulkan SDK: brew install vulkan-headers vulkan-loader molten-vk"
        fi

        if [[ "$GPU_AVAILABLE" == "no" ]]; then
            echo "❌ GPU hardware issues detected"
            echo "   - Verify MoltenVK is properly installed"
            echo "   - Check system_profiler SPDisplaysDataType for Metal support"
        fi
        ;;

    "Linux")
        if ! command -v vulkaninfo >/dev/null 2>&1; then
            echo "❌ Install Vulkan SDK: sudo apt install vulkan-tools libvulkan-dev vulkan-validationlayers-dev"
        fi

        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "❌ Install NVIDIA drivers for GPU support"
        fi
        ;;
esac

# Build tool recommendations
if ! command -v cmake >/dev/null 2>&1; then
    echo "❌ Install CMake for C++ build system"
fi

if ! rustup component list --installed | grep -q clippy; then
    echo "❌ Install clippy: rustup component add clippy"
fi

# CI recommendations
echo ""
echo "CI Infrastructure Recommendations:"
echo "1. Enable cross-platform GPU runners (Linux + Windows)"
echo "2. Add memory safety testing (ASan + TSan)"
echo "3. Implement production validation pipeline"
echo "4. Configure safety-first default settings"
echo ""

echo "📊 DIAGNOSTIC SUMMARY"
echo "===================="
echo "Platform: $PLATFORM"
echo "GPU Available: $GPU_AVAILABLE"
echo "Vulkan SDK: $(command -v vulkaninfo >/dev/null 2>&1 && echo "installed" || echo "missing")"
echo "Build Tools: $(command -v cmake >/dev/null 2>&1 && echo "ready" || echo "incomplete")"
echo "Rust Setup: $(rustup component list --installed | grep -q clippy && echo "complete" || echo "needs components")"
echo ""

if [[ "$GPU_AVAILABLE" == "yes" ]]; then
    echo "✅ GPU runner appears functional - ready for production pipeline implementation"
else
    echo "❌ GPU runner needs repair before production pipeline implementation"
fi

echo ""
echo "🔗 NEXT STEPS"
echo "============="
echo "1. Fix any issues identified above"
echo "2. Run: scripts/setup-local-gpu-runner.sh (once created)"
echo "3. Test GPU functionality with: cargo test --features 'async,glsl,debug,validation,test-utils'"
echo "4. Proceed with cross-platform CI matrix implementation"

echo ""
echo "Diagnostic complete. Report saved for analysis."

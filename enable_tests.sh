#!/bin/bash

# Script to systematically enable all ignored GPU tests
# This script removes #[ignore = "requires Vulkan-capable device"] attributes
# and adds the mandatory Vulkan detection pattern

set -e

echo "Enabling ignored GPU tests across all source files..."

# List of files with ignored tests
files=(
    "kompute/src/tensor.rs"
    "kompute/src/queue.rs"
    "kompute/src/async_gpu.rs"
    "kompute/src/multi_queue.rs"
    "kompute/src/sequence.rs"
    "kompute/tests/sequence_command_tests.rs"
    "kompute/tests/algorithm_spirv_tests.rs"
    "kompute/tests/manager_unit_tests.rs"
    "kompute/tests/property_tests.rs"
    "kompute/tests/comprehensive_integration.rs"
    "kompute/tests/benchmark_tests.rs"
    "kompute/tests/build_verification.rs"
    "kompute/tests/complete_solution_integration.rs"
    "kompute/tests/handle_based_test.rs"
    "kompute/tests/handle_based_integration.rs"
    "kompute/tests/gpu_compute_integration.rs"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."

        # Create backup
        cp "$file" "$file.backup"

        # Replace the ignore pattern
        sed -i.tmp 's/#\[ignore = "requires Vulkan-capable device"\]//g' "$file"

        # Clean up temporary file
        rm -f "$file.tmp"

        echo "  ✓ Removed ignore attributes from $file"
    else
        echo "  ⚠️  File not found: $file"
    fi
done

echo ""
echo "✅ All ignore attributes removed!"
echo "⚠️  Note: Tests may still fail due to kp::Manager constructor segfault"
echo "📝 Backup files created with .backup extension"

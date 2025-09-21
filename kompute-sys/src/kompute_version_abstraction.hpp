#pragma once

// =============================================================================
// Kompute Version Abstraction Layer
// =============================================================================
// This header provides a unified interface across different Kompute versions
// by conditionally defining appropriate types and includes based on the
// version being built against.
// =============================================================================

// Include the appropriate Kompute headers based on version
// Use feature flags from build.rs to determine which version is being built
// Version-specific includes and compatibility checks
#if defined(KOMPUTE_VERSION_V081)
    #error "Kompute v0.8.1 is not supported (as of September 2025). Known issues: CMake compatibility problems, macOS/MoltenVK segfaults, and file structure differences from master. Current implementation is designed for master branch. Use --features kompute-master instead. This status will likely change when the next official Kompute release is published."
#elif defined(KOMPUTE_VERSION_V090)
    #error "Kompute v0.9.0 is not supported (as of September 2025). Known issues: file structure differences (missing headers) and macOS/MoltenVK Manager segfaults. Current implementation is designed for master branch. Use --features kompute-master instead. This status will likely change when the next official Kompute release is published."
#else
    // Master branch - currently the only fully supported version
    #include "kompute/Manager.hpp"
    #include "kompute/Tensor.hpp"
    #include "kompute/Algorithm.hpp"
    #include "kompute/Sequence.hpp"
    #include "kompute/operations/OpSyncDevice.hpp"
    #include "kompute/operations/OpSyncLocal.hpp"
    #include "kompute/operations/OpAlgoDispatch.hpp"

    // Type aliases for master branch
    using AbsManager = kp::Manager;
    using AbsDataObject = kp::Tensor;
    using AbsAlgorithm = kp::Algorithm;
    using AbsSequence = kp::Sequence;
#endif
#include "handle_registry.hpp"
#include <memory>

/// Thread-safe singleton pattern for the global registry
HandleRegistry& get_global_registry() {
    // C++11 guarantees thread-safe initialization of static local variables
    static HandleRegistry instance;
    return instance;
}

#pragma once

#include <cstdint>
#include <stdexcept>

#include <mujoco/mujoco.h>

inline const mjModel* as_model(std::uintptr_t address) {
    if (address == 0) {
        throw std::runtime_error("mjModel address is zero.");
    }
    return reinterpret_cast<const mjModel*>(address);
}

inline const mjData* as_data(std::uintptr_t address) {
    if (address == 0) {
        throw std::runtime_error("mjData address is zero.");
    }
    return reinterpret_cast<const mjData*>(address);
}

inline mjData* as_data_mutable(std::uintptr_t address) {
    if (address == 0) {
        throw std::runtime_error("mjData address is zero.");
    }
    return reinterpret_cast<mjData*>(address);
}

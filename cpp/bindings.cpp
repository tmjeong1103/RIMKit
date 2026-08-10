#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>

#include <mujoco/mujoco.h>

#ifdef RI_NATIVE_ENABLE_MUJOCO_ECHO
#include "mujoco_echo_bindings.hpp"
#endif

#ifdef RI_NATIVE_ENABLE_SIGNED_DISTANCE
#include "signed_distance_bindings.hpp"
#endif

#ifdef RI_NATIVE_ENABLE_IK
#include "ik_bindings.hpp"
#endif

#ifdef RI_NATIVE_ENABLE_TUTORIAL
#include "tutorial_bindings.hpp"
#endif

namespace nb = nanobind;

#ifndef RI_NATIVE_MODULE_NAME
#define RI_NATIVE_MODULE_NAME _mujoco_native
#endif

#define RI_NATIVE_STRINGIFY_INNER(x) #x
#define RI_NATIVE_STRINGIFY(x) RI_NATIVE_STRINGIFY_INNER(x)

namespace {

std::string mujoco_header_version() {
    const int major = mjVERSION_HEADER / 1000000;
    const int minor = (mjVERSION_HEADER % 1000000) / 1000;
    const int patch = mjVERSION_HEADER % 1000;
    return (
        std::to_string(major) + "." +
        std::to_string(minor) + "." +
        std::to_string(patch)
    );
}

const char* compiler_version() {
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC " RI_NATIVE_STRINGIFY(_MSC_VER);
#else
    return "unknown";
#endif
}

}

nb::dict native_info() {
    nb::dict out;
    out["api_version"] = 1;
    out["backend"] = "nanobind";
    out["module"] = "core_retarget." RI_NATIVE_STRINGIFY(RI_NATIVE_MODULE_NAME);
    out["mujoco_version"] = mujoco_header_version();
    out["mujoco_runtime_version"] = mj_versionString();
    out["mujoco_abi_version"] = mjVERSION_HEADER;
    out["compiler"] = compiler_version();
    out["cpp_standard"] = static_cast<long long>(__cplusplus);
    return out;
}

NB_MODULE(RI_NATIVE_MODULE_NAME,m) {
    m.doc() = "Native MuJoCo IK and signed-distance backend for CoRe.";
    m.def("native_info",&native_info,"Return native backend build information.");

#ifdef RI_NATIVE_ENABLE_TUTORIAL
    bind_tutorial(m);
#endif

#ifdef RI_NATIVE_ENABLE_MUJOCO_ECHO
    bind_mujoco_echo(m);
#endif

#ifdef RI_NATIVE_ENABLE_SIGNED_DISTANCE
    bind_signed_distance(m);
#endif

#ifdef RI_NATIVE_ENABLE_IK
    bind_ik(m);
#endif
}

cmake_minimum_required(VERSION 3.18...4.1)
project(RIMKit LANGUAGES CXX)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# Normalize CMake's per-configuration optimization defaults; the native target
# sets the required level below.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
  foreach(
    CORE_CONFIG_FLAGS
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_CXX_FLAGS_MINSIZEREL
  )
    string(
      REGEX REPLACE
      "(^|[ \t])-O(0|1|2|3|s|g|fast)([ \t]|$)"
      " "
      ${CORE_CONFIG_FLAGS}
      "${${CORE_CONFIG_FLAGS}}"
    )
  endforeach()
endif()

# MuJoCo's Python wheel is the canonical source of the ABI-pinned headers and
# shared library. Query its package root instead of relying on a machine-local
# MuJoCo SDK installation.
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "from pathlib import Path; import mujoco; print(Path(mujoco.__file__).resolve().parent)"
  RESULT_VARIABLE CORE_MUJOCO_QUERY_STATUS
  OUTPUT_VARIABLE CORE_MUJOCO_ROOT
  ERROR_VARIABLE CORE_MUJOCO_QUERY_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT CORE_MUJOCO_QUERY_STATUS EQUAL 0)
  message(FATAL_ERROR
    "Could not locate the MuJoCo Python package used to build RIMKit:\n"
    "${CORE_MUJOCO_QUERY_ERROR}"
  )
endif()

find_path(
  CORE_MUJOCO_INCLUDE_DIR
  NAMES mujoco/mujoco.h
  PATHS "${CORE_MUJOCO_ROOT}/include"
  NO_DEFAULT_PATH
)
if(NOT CORE_MUJOCO_INCLUDE_DIR)
  message(FATAL_ERROR
    "MuJoCo headers were not found under ${CORE_MUJOCO_ROOT}/include."
  )
endif()

nanobind_add_module(
  _core_native
  NB_STATIC
  NOMINSIZE
  cpp/bindings.cpp
  cpp/ik.cpp
  cpp/ik_bindings.cpp
  cpp/signed_distance.cpp
  cpp/signed_distance_bindings.cpp
)

target_compile_features(_core_native PRIVATE cxx_std_17)
target_compile_options(
  _core_native
  PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-O2>
    $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-fno-strict-aliasing>
    $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-fno-fast-math>
)
target_compile_definitions(
  _core_native
  PRIVATE
    NB_COMPACT_ASSERTIONS
    RI_NATIVE_MODULE_NAME=_core_native
    RI_NATIVE_ENABLE_IK
    RI_NATIVE_ENABLE_SIGNED_DISTANCE
)
if(TARGET nanobind-static)
  target_compile_options(
    nanobind-static
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-O2>
      $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-fno-fast-math>
  )
  set_target_properties(
    nanobind-static
    PROPERTIES
      CXX_EXTENSIONS FALSE
      INTERPROCEDURAL_OPTIMIZATION FALSE
  )
endif()
target_include_directories(
  _core_native
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
    "${CORE_MUJOCO_INCLUDE_DIR}"
)
set_target_properties(
  _core_native
  PROPERTIES
    CXX_EXTENSIONS FALSE
    INTERPROCEDURAL_OPTIMIZATION FALSE
    POSITION_INDEPENDENT_CODE TRUE
)

# Link the ABI-pinned shared library shipped by the `mujoco` Python package.
# The install RPATHs are relative to the extension's package directory, so a
# wheel never retains the build environment's absolute site-packages path.
if(APPLE)
  find_file(
    CORE_MUJOCO_LIBRARY
    NAMES
      libmujoco.3.6.0.dylib
      libmujoco.dylib
    PATHS "${CORE_MUJOCO_ROOT}"
    NO_DEFAULT_PATH
  )
elseif(UNIX)
  find_file(
    CORE_MUJOCO_LIBRARY
    NAMES
      libmujoco.so.3.6.0
      libmujoco.so
    PATHS "${CORE_MUJOCO_ROOT}"
    NO_DEFAULT_PATH
  )
else()
  message(FATAL_ERROR "The RIMKit native backend currently supports macOS and Linux.")
endif()
if(NOT CORE_MUJOCO_LIBRARY)
  message(FATAL_ERROR
    "MuJoCo shared library was not found under ${CORE_MUJOCO_ROOT}."
  )
endif()

target_link_libraries(_core_native PRIVATE "${CORE_MUJOCO_LIBRARY}")
if(APPLE)
  set(CORE_MUJOCO_INSTALL_RPATH "@loader_path/../mujoco")
else()
  set(CORE_MUJOCO_INSTALL_RPATH "$ORIGIN/../mujoco")
endif()
set_target_properties(
  _core_native
  PROPERTIES
    BUILD_RPATH "${CORE_MUJOCO_ROOT}"
    INSTALL_RPATH "${CORE_MUJOCO_INSTALL_RPATH}"
)

# The macOS MuJoCo wheel's dylib ID currently includes an internal framework
# path even though the wheel stores the file directly in `mujoco/`. Rewrite
# only that load command to the relocatable filename used by MuJoCo's own
# Python extension. The numerical implementation remains untouched.
if(APPLE)
  find_program(CORE_OTOOL_EXECUTABLE NAMES otool)
  if(NOT CORE_OTOOL_EXECUTABLE)
    message(FATAL_ERROR "The macOS `otool` command is required to build RIMKit.")
  endif()
  execute_process(
    COMMAND "${CORE_OTOOL_EXECUTABLE}" -D "${CORE_MUJOCO_LIBRARY}"
    RESULT_VARIABLE CORE_MUJOCO_OTOOL_STATUS
    OUTPUT_VARIABLE CORE_MUJOCO_OTOOL_OUTPUT
    ERROR_VARIABLE CORE_MUJOCO_OTOOL_ERROR
  )
  if(NOT CORE_MUJOCO_OTOOL_STATUS EQUAL 0)
    message(FATAL_ERROR
      "Could not inspect the MuJoCo dylib install name:\n${CORE_MUJOCO_OTOOL_ERROR}"
    )
  endif()
  string(
    REGEX MATCH "@rpath/[^\n\r]+"
    CORE_MUJOCO_DYLIB_ID
    "${CORE_MUJOCO_OTOOL_OUTPUT}"
  )
  get_filename_component(CORE_MUJOCO_DYLIB_NAME "${CORE_MUJOCO_LIBRARY}" NAME)
  set(CORE_MUJOCO_WHEEL_DYLIB_ID "@rpath/${CORE_MUJOCO_DYLIB_NAME}")
  if(
    CORE_MUJOCO_DYLIB_ID
    AND NOT CORE_MUJOCO_DYLIB_ID STREQUAL CORE_MUJOCO_WHEEL_DYLIB_ID
  )
    add_custom_command(
      TARGET _core_native
      POST_BUILD
      COMMAND
        "${CMAKE_INSTALL_NAME_TOOL}"
        -change
        "${CORE_MUJOCO_DYLIB_ID}"
        "${CORE_MUJOCO_WHEEL_DYLIB_ID}"
        "$<TARGET_FILE:_core_native>"
      VERBATIM
      COMMENT "Making the MuJoCo dylib load path wheel-relative"
    )
  endif()
endif()

install(TARGETS _core_native LIBRARY DESTINATION rimkit)

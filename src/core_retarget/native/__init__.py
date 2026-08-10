"""Runtime selection for CoRe's compiled MuJoCo accelerator."""

from core_retarget.native.backend import (
    NATIVE_API_VERSION,
    NATIVE_MODULE_NAME,
    NATIVE_MUJOCO_VERSION,
    BackendPreference,
    BackendSelection,
    mujoco_addresses,
    resolve_backend,
)

__all__ = [
    "NATIVE_MODULE_NAME",
    "NATIVE_API_VERSION",
    "NATIVE_MUJOCO_VERSION",
    "BackendPreference",
    "BackendSelection",
    "mujoco_addresses",
    "resolve_backend",
]

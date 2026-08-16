"""Explicit, inspectable selection of native or portable compute kernels."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal, cast

from rimkit.exceptions import ConfigurationError

BackendPreference = Literal["auto", "native", "python"]
SelectedBackend = Literal["native", "python"]
NATIVE_MODULE_NAME = "rimkit._core_native"
NATIVE_API_VERSION = "1"
NATIVE_MUJOCO_VERSION = "3.6.0"
_REQUIRED_NATIVE_SYMBOLS = (
    "native_info",
    "signed_distance_arrays",
    "solve_body_position_ik",
    "solve_planar_base_ik",
)


def _normalize_preference(value: str) -> BackendPreference:
    normalized = str(value).strip().lower()
    if normalized not in {"auto", "native", "python"}:
        raise ConfigurationError(
            f"Unknown compute backend {value!r}; choose 'auto', 'native', or 'python'."
        )
    return cast(BackendPreference, normalized)


def _native_import_error(error: BaseException) -> str:
    message = str(error).strip()
    return type(error).__name__ if not message else f"{type(error).__name__}: {message}"


def mujoco_addresses(model: Any, data: Any) -> tuple[int, int]:
    """Return the non-zero MuJoCo addresses expected by the native extension."""

    try:
        model_address = int(model._address)
        data_address = int(data._address)
    except AttributeError as error:
        raise TypeError("model and data must be MuJoCo objects with `_address`.") from error
    if model_address == 0 or data_address == 0:
        raise ValueError("MuJoCo model/data address must be non-zero.")
    return model_address, data_address


@dataclass(frozen=True, slots=True)
class BackendSelection:
    """One resolved backend shared by every numerical stage in a pipeline run."""

    requested: BackendPreference
    selected: SelectedBackend
    reason: str
    module_name: str | None = None
    native_info: dict[str, str] = field(default_factory=dict)
    detail: str | None = None
    _module: ModuleType | None = field(default=None, repr=False, compare=False)

    @property
    def is_native(self) -> bool:
        return self.selected == "native"

    @property
    def module(self) -> ModuleType:
        """Return the validated extension module for a native selection."""

        if self._module is None:
            raise RuntimeError("The selected compute backend has no native module.")
        return self._module

    def manifest_record(self, *, include_detail: bool = False) -> dict[str, Any]:
        """Return JSON-safe provenance for a published result manifest."""

        record: dict[str, Any] = {
            "requested": self.requested,
            "selected": self.selected,
            "reason": self.reason,
        }
        if self.module_name is not None:
            record["module"] = self.module_name
        if self.native_info:
            record["native_info"] = dict(self.native_info)
        if include_detail and self.detail is not None:
            record["detail"] = self.detail
        return record


def _load_native_module() -> tuple[ModuleType | None, dict[str, str], str | None]:
    try:
        module = importlib.import_module(NATIVE_MODULE_NAME)
        missing = [
            name for name in _REQUIRED_NATIVE_SYMBOLS if not callable(getattr(module, name, None))
        ]
        if missing:
            raise ImportError(
                f"{NATIVE_MODULE_NAME} is missing required symbols: {', '.join(missing)}"
            )
        raw_info = module.native_info()
        if not isinstance(raw_info, dict):
            raise TypeError("native_info() must return a dict")
        info = {str(key): str(value) for key, value in raw_info.items()}
        if info.get("api_version") != NATIVE_API_VERSION:
            raise ImportError(
                f"{NATIVE_MODULE_NAME} API version is {info.get('api_version')!r}; "
                f"expected {NATIVE_API_VERSION!r}"
            )
        if info.get("module") != NATIVE_MODULE_NAME:
            raise ImportError(
                f"{NATIVE_MODULE_NAME} reports an unexpected module name {info.get('module')!r}"
            )
        if info.get("mujoco_version") != NATIVE_MUJOCO_VERSION:
            raise ImportError(
                f"{NATIVE_MODULE_NAME} was compiled for MuJoCo "
                f"{info.get('mujoco_version')!r}; expected {NATIVE_MUJOCO_VERSION!r}"
            )
        if info.get("mujoco_runtime_version") != NATIVE_MUJOCO_VERSION:
            raise ImportError(
                f"{NATIVE_MODULE_NAME} loaded MuJoCo "
                f"{info.get('mujoco_runtime_version')!r}; expected "
                f"{NATIVE_MUJOCO_VERSION!r}"
            )
        return module, info, None
    except Exception as error:  # extension import failures vary by platform/loader
        return None, {}, _native_import_error(error)


def resolve_backend(
    preference: BackendPreference | str | BackendSelection = "auto",
) -> BackendSelection:
    """Resolve a backend once, with an explicit Python fallback only for ``auto``."""

    if isinstance(preference, BackendSelection):
        return preference
    requested = _normalize_preference(preference)
    if requested == "python":
        return BackendSelection(
            requested=requested,
            selected="python",
            reason="explicit_python",
        )

    module, info, detail = _load_native_module()
    if module is not None:
        return BackendSelection(
            requested=requested,
            selected="native",
            reason="native_extension_available",
            module_name=NATIVE_MODULE_NAME,
            native_info=info,
            _module=module,
        )
    if requested == "native":
        raise ConfigurationError(
            "The native compute backend was requested but could not be loaded from "
            f"{NATIVE_MODULE_NAME}. Reinstall RIMKit from a platform wheel or "
            f"build it locally. Loader detail: {detail}"
        )
    return BackendSelection(
        requested=requested,
        selected="python",
        reason="native_extension_unavailable",
        module_name=NATIVE_MODULE_NAME,
        detail=detail,
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

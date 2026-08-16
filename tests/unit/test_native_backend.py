from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

import rimkit.native.backend as backend_module
from rimkit.config.schema import RunConfig
from rimkit.exceptions import ConfigurationError
from rimkit.native import BackendSelection, resolve_backend


def _fake_native_module() -> ModuleType:
    module = ModuleType("rimkit._core_native")
    module.native_info = lambda: {  # type: ignore[attr-defined]
        "backend": "nanobind",
        "module": "rimkit._core_native",
        "api_version": "1",
        "mujoco_version": "3.6.0",
        "mujoco_runtime_version": "3.6.0",
    }
    module.signed_distance_arrays = lambda *_args: {}  # type: ignore[attr-defined]
    module.solve_body_position_ik = lambda *_args: {}  # type: ignore[attr-defined]
    module.solve_planar_base_ik = lambda *_args: {}  # type: ignore[attr-defined]
    return module


def test_explicit_python_backend_does_not_import_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_module.importlib,
        "import_module",
        lambda _name: pytest.fail("explicit Python backend must not import native code"),
    )

    selection = resolve_backend("python")

    assert selection.selected == "python"
    assert selection.reason == "explicit_python"
    assert selection.manifest_record() == {
        "requested": "python",
        "selected": "python",
        "reason": "explicit_python",
    }


def test_auto_selects_a_valid_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _fake_native_module()
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _name: module)

    selection = resolve_backend("auto")

    assert selection.is_native
    assert selection.module is module
    assert selection.native_info["backend"] == "nanobind"
    assert selection.manifest_record()["module"] == "rimkit._core_native"


def test_auto_falls_back_but_explicit_native_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(_name: str) -> Any:
        raise ImportError("test loader failure")

    monkeypatch.setattr(backend_module.importlib, "import_module", fail_import)

    fallback = resolve_backend("auto")

    assert fallback.selected == "python"
    assert fallback.reason == "native_extension_unavailable"
    assert fallback.detail == "ImportError: test loader failure"
    assert "detail" not in fallback.manifest_record()
    assert fallback.manifest_record(include_detail=True)["detail"] == (
        "ImportError: test loader failure"
    )
    with pytest.raises(ConfigurationError, match="native compute backend.*could not be loaded"):
        resolve_backend("native")


def test_mujoco_header_and_runtime_versions_must_both_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _fake_native_module()
    module.native_info = lambda: {  # type: ignore[attr-defined]
        "backend": "nanobind",
        "module": "rimkit._core_native",
        "api_version": "1",
        "mujoco_version": "3.6.0",
        "mujoco_runtime_version": "3.5.0",
    }
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _name: module)

    fallback = resolve_backend("auto")

    assert fallback.selected == "python"
    assert fallback.reason == "native_extension_unavailable"
    assert fallback.detail is not None
    assert "loaded MuJoCo '3.5.0'" in fallback.detail


def test_backend_selection_is_idempotent_and_run_config_rejects_unknown_value() -> None:
    selection = BackendSelection(
        requested="python",
        selected="python",
        reason="test",
    )

    assert resolve_backend(selection) is selection
    with pytest.raises(ConfigurationError, match="Compute backend"):
        RunConfig(robot="g1", backend="gpu")  # type: ignore[arg-type]

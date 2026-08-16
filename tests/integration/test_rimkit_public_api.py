from __future__ import annotations

import pytest

import core_retarget
import rimkit
from rimkit.cli.main import main
from rimkit.exceptions import ConfigurationError
from rimkit.methods.core import METHOD_ID, Retargeter


def test_rimkit_exposes_core_as_the_current_method() -> None:
    methods = rimkit.list_methods()

    assert tuple(method.method_id for method in methods) == ("core",)
    assert rimkit.get_method("CoRe") is methods[0]
    assert METHOD_ID == "core"
    assert Retargeter is rimkit.Retargeter
    with pytest.raises(ConfigurationError, match="Unknown method"):
        rimkit.get_method("unknown")


def test_legacy_top_level_package_reexports_the_canonical_api() -> None:
    assert core_retarget.__version__ == rimkit.__version__
    assert core_retarget.Retargeter is rimkit.Retargeter
    assert core_retarget.RunConfig is rimkit.RunConfig


def test_methods_list_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["methods", "list"]) == 0
    output = capsys.readouterr().out
    assert "core" in output
    assert "CoRe" in output

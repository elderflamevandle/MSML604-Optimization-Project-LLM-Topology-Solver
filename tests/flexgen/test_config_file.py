import pytest

from src.flexgen.config_file import (
    load_flexgen_config,
    override,
    require_section,
    require_value,
    resolve_repo_path,
)


def test_load_flexgen_config_reads_yaml(tmp_path):
    cfg = tmp_path / "config_flexgen.yml"
    cfg.write_text("paths:\n  model: /models/qwen\n")

    data = load_flexgen_config(cfg)

    assert data["paths"]["model"] == "/models/qwen"


def test_require_section_and_value_errors_are_clear():
    with pytest.raises(KeyError, match="Missing required config section"):
        require_section({}, "paths")

    with pytest.raises(KeyError, match="Missing required config value"):
        require_value({}, "model", "paths")


def test_override_prefers_cli_value_only_when_present():
    assert override("from_config", None) == "from_config"
    assert override("from_config", "from_cli") == "from_cli"


def test_resolve_repo_path_resolves_relative_path(tmp_path):
    assert resolve_repo_path("configs/workload.yaml", tmp_path) == str(
        tmp_path / "configs" / "workload.yaml"
    )
    assert resolve_repo_path(tmp_path / "x", tmp_path) == str(tmp_path / "x")


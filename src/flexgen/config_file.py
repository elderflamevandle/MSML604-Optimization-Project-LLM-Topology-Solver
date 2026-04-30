from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_flexgen_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"FlexGen config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"FlexGen config must be a YAML mapping: {path}")
    return data


def require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise KeyError(f"Missing required config section: {name}")
    return section


def require_value(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing required config value: {section_name}.{key}")
    return section[key]


def override(config_value: Any, cli_value: Any) -> Any:
    return config_value if cli_value is None else cli_value


def resolve_repo_path(value: str | Path, repo_root: str | Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(Path(repo_root) / path)


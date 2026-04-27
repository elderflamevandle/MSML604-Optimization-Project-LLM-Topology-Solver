import pytest
from src.flexgen.workload import WorkloadSpec, load_workload


def test_load_workload_from_yaml(tmp_path):
    yaml_file = tmp_path / "workload.yaml"
    yaml_file.write_text("prompt_len: 256\ndecode_len: 64\n")
    spec = load_workload(str(yaml_file))
    assert spec.prompt_len == 256
    assert spec.decode_len == 64


def test_load_workload_defaults_when_field_missing(tmp_path):
    yaml_file = tmp_path / "workload.yaml"
    yaml_file.write_text("prompt_len: 1024\n")
    spec = load_workload(str(yaml_file))
    assert spec.prompt_len == 1024
    assert spec.decode_len == 128


def test_load_workload_rejects_nonpositive():
    with pytest.raises(ValueError, match="prompt_len must be positive"):
        WorkloadSpec(prompt_len=0, decode_len=128)
    with pytest.raises(ValueError, match="decode_len must be positive"):
        WorkloadSpec(prompt_len=512, decode_len=-1)

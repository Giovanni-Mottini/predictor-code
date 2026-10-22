"""Production module boundaries and exact implementation provenance."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from shared_code.reader import DataContractError
from shared_code.provenance import implementation_hashes
from xsens_code.xsens_cli import method_document
from xsens_code.xsens_features import XsensConfig


def test_current_method_hashes_include_elbow_implementation():
    root = Path(__file__).resolve().parents[2]
    hashes = method_document(XsensConfig())["implementation_sha256"]
    assert "xsens_code/elbow_rom.py" in hashes
    assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha for name, sha in hashes.items())
    with pytest.raises(DataContractError):
        implementation_hashes(("../outside.py",))


def test_metadata_reader_import_does_not_load_algorithms(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = "import shared_code.reader,sys; assert not any(n.startswith(('garmin_code','xsens_code','scipy','dtaidistance')) for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", ["xsens-features", "xsens-repetitions", "pipeline"])
def test_commands_work_from_another_directory(command, tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-m", "xsens_code", command, "--help"], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout and "--no-progress" in result.stdout

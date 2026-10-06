import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Todo lo que contiene ECG de personas (dato biométrico) o resultados derivados NUNCA debe llegar a git.
MUST_BE_IGNORED = [
    "data/raw/polar/P01__2026-10-06__reposo.npy",
    "data/processed/polar/P01__2026-10-06__reposo.npz",
    "data/processed/public/pub_Person_01__rec_01.npz",
    "data/descartadas/P01__2026-10-06__reposo__101500.npy",
    "data/piloto/processed/juanC__2026-10-01_2000.npz",
    "models/modelo_v1.pt",
    "models/modelo_v1.json",
    "reports/recordings/P01__2026-10-06.png",
    "reports/try_me.png",
]


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="requiere git")
@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_biometric_data_and_artifacts_are_git_ignored(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    assert result.returncode == 0, f"{path} NO está en .gitignore: un 'git add' lo subiría al repositorio"

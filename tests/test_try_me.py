import numpy as np
import pytest

from src.checkpoint import save_checkpoint
from src.config import Config
from src.data import WindowTable
from src.errors import DataError
from src.evaluate import raw_embed_fn
from src.model import ECGEmbeddingNet
from src.record import main as record_main
from src.try_me import format_trial, main, plot_trial, public_threshold, run_trial


def _table(sessions=("2026-10-01_0900", "2026-10-01_1500"), people=("ana", "luis"), n=12, seed=0):
    """Dos "corazones" con forma fija (senos de frecuencia distinta): la señal cruda ya los separa."""
    rng = np.random.default_rng(seed)
    t = np.arange(650) / 130
    base = {"ana": np.sin(2 * np.pi * 1.2 * t), "luis": np.sin(2 * np.pi * 2.6 * t) ** 3}
    wins, pids, sids = [], [], []
    for p in people:
        for s in sessions:
            for _ in range(n):
                w = base[p] + rng.normal(0, 0.1, 650)
                wins.append((w - w.mean()) / w.std())
                pids.append(p)
                sids.append(s)
    k = len(wins)
    return WindowTable(np.array(wins, dtype=np.float32)[:, None, :], np.array(pids), np.array(sids),
                       np.full(k, "rest"), "polar")


def test_run_trial_separates_two_clearly_different_people():
    r = run_trial(_table(), raw_embed_fn, Config(), threshold=0.5)
    assert list(r.owners) == ["ana", "luis"]
    for person in r.owners:
        s = r.per_person[person]
        assert s["mean_own"] > s["mean_other"] and s["looks_like_self"]
        assert s["accepted_own"] > 0.9 and s["accepted_other"] < 0.1
    assert r.eer < 0.05 and r.auc > 0.95
    assert r.threshold == 0.5 and "indicado" in r.threshold_source


def test_run_trial_uses_best_threshold_when_none_is_given():
    r = run_trial(_table(), raw_embed_fn, Config())
    assert r.threshold == r.best_threshold and "optimista" in r.threshold_source


def test_run_trial_needs_two_people():
    with pytest.raises(DataError, match="2 personas"):
        run_trial(_table(people=("ana",)), raw_embed_fn, Config())


def test_run_trial_needs_two_sessions_per_person_and_names_the_person():
    t = _table()
    keep = ~((t.participant_ids == "luis") & (t.session_ids == "2026-10-01_1500"))
    with pytest.raises(DataError, match="luis"):
        run_trial(t.subset(keep), raw_embed_fn, Config())


def test_format_trial_is_plain_language():
    text = format_trial(run_trial(_table(), raw_embed_fn, Config(), threshold=0.5))
    assert "ana" in text and "luis" in text and "%" in text


def test_plot_trial_writes_png(tmp_path):
    out = plot_trial(run_trial(_table(), raw_embed_fn, Config(), threshold=0.5), tmp_path / "t.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_public_threshold_is_none_without_public_data(tmp_path):
    assert public_threshold(raw_embed_fn, tmp_path / "no_existe", Config()) is None


def test_cli_end_to_end_with_simulated_recordings(tmp_path, capsys):
    cfg = Config()
    for pid in ("ana", "luis"):
        for session in ("2026-10-01_0900", "2026-10-01_1500"):
            assert record_main(["--id", pid, "--session", session, "--seconds", "40", "--simulate", "--no-open",
                                "--raw-dir", str(tmp_path / "raw"), "--processed-dir", str(tmp_path / "proc"),
                                "--plots-dir", str(tmp_path / "plots")]) == 0
    ckpt = save_checkpoint(ECGEmbeddingNet(cfg), cfg, tmp_path / "models", "pretrain", {}, "x")
    rc = main(["--checkpoint", str(ckpt), "--polar-dir", str(tmp_path / "proc"),
               "--public-dir", str(tmp_path / "sin_publico"), "--out", str(tmp_path / "try.png"), "--no-open"])
    assert rc == 0 and (tmp_path / "try.png").exists()
    assert "ana" in capsys.readouterr().out


def test_cli_reports_friendly_error_when_data_is_insufficient(tmp_path, capsys):
    cfg = Config()
    record_main(["--id", "ana", "--session", "2026-10-01_0900", "--seconds", "30", "--simulate", "--no-open",
                 "--raw-dir", str(tmp_path / "raw"), "--processed-dir", str(tmp_path / "proc"),
                 "--plots-dir", str(tmp_path / "plots")])
    ckpt = save_checkpoint(ECGEmbeddingNet(cfg), cfg, tmp_path / "models", "pretrain", {}, "x")
    rc = main(["--checkpoint", str(ckpt), "--polar-dir", str(tmp_path / "proc"),
               "--out", str(tmp_path / "t.png"), "--no-open"])
    assert rc == 1 and "2 personas" in capsys.readouterr().out


# --- El umbral de otro dataset puede no servir: el reporte debe mostrar también el mejor umbral ---
def test_report_also_shows_rates_at_the_best_threshold_when_given_threshold_is_off():
    r = run_trial(_table(), raw_embed_fn, Config(), threshold=-1.0)      # umbral absurdo: acepta a todos
    for s in r.per_person.values():
        assert s["accepted_other"] == 1.0                                # con el umbral dado, acepta impostores
        assert s["accepted_own_best"] > 0.9 and s["accepted_other_best"] < 0.1
    text = format_trial(r)
    assert "mejor umbral" in text and "no transfiere" in text


def test_report_does_not_repeat_best_threshold_block_when_thresholds_match():
    text = format_trial(run_trial(_table(), raw_embed_fn, Config()))     # umbral = el mejor
    assert "no transfiere" not in text


# --- Mensaje de error útil cuando la segunda grabación se guardó con otro --id ---
def test_missing_sessions_error_lists_counts_and_hints_at_the_id_mistake():
    t = _table()
    keep = ~((t.participant_ids == "luis") & (t.session_ids == "2026-10-01_1500"))
    with pytest.raises(DataError) as e:
        run_trial(t.subset(keep), raw_embed_fn, Config())
    msg = str(e.value)
    assert "luis (1 grabación)" in msg and "ana (2 grabaciones)" in msg   # cuenta por persona
    assert "mismo --id" in msg                                            # pista del error más común

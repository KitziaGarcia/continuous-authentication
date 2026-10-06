import numpy as np
import pytest

from src.errors import EvaluationError
from src.evaluate import auc_score, eer, far_frr_at, roc_points


def test_perfect_separation_gives_zero_eer():
    e, thr = eer(np.full(50, 0.9), np.full(50, 0.1))
    assert e == 0.0
    far, frr = far_frr_at(np.full(50, 0.9), np.full(50, 0.1), thr)
    assert far == 0.0 and frr == 0.0


def test_random_scores_give_eer_near_half():
    rng = np.random.default_rng(0)
    e, _ = eer(rng.uniform(size=5000), rng.uniform(size=5000))
    assert e == pytest.approx(0.5, abs=0.05)


def test_known_overlap_eer():
    # genuinos 5..14, impostores 0..9 (se traslapan en 5..9); cuenta hecha a mano:
    genuine = np.arange(5.0, 15.0)
    impostor = np.arange(0.0, 10.0)
    e, thr = eer(genuine, impostor)
    # umbral 7: FAR = impostores>=7 -> 3/10 ; FRR = genuinos<7 -> 2/10
    # umbral 8: FAR = 2/10 ; FRR = 3/10  => en ambos |FAR-FRR| = 0.1 y EER = (0.3+0.2)/2 = 0.25
    assert e == pytest.approx(0.25)
    assert thr in (7.0, 8.0)


def test_far_frr_at_threshold_convention():
    genuine = np.array([0.2, 0.6, 0.8])
    impostor = np.array([0.1, 0.5, 0.7])
    far, frr = far_frr_at(genuine, impostor, 0.5)
    assert far == pytest.approx(2 / 3)    # 0.5 y 0.7 aceptados (>=)
    assert frr == pytest.approx(1 / 3)    # solo 0.2 rechazado (<)


def test_auc_perfect_and_random():
    assert auc_score(np.full(10, 0.9), np.full(10, 0.1)) == 1.0
    rng = np.random.default_rng(1)
    assert auc_score(rng.uniform(size=3000), rng.uniform(size=3000)) == pytest.approx(0.5, abs=0.05)


def test_roc_points_shapes_and_range():
    far, frr = roc_points(np.array([0.9, 0.8, 0.4]), np.array([0.1, 0.5, 0.3]))
    assert len(far) == len(frr)
    assert far.min() >= 0 and far.max() <= 1 and frr.min() >= 0 and frr.max() <= 1


def test_empty_inputs_raise():
    with pytest.raises(EvaluationError):
        eer(np.array([]), np.array([0.1]))
    with pytest.raises(EvaluationError):
        eer(np.array([0.1]), np.array([]))


import torch

from src.config import Config
from src.data import WindowTable
from src.evaluate import (
    Trials,
    build_trials,
    first_session_table,
    l2_normalize,
    make_model_embed_fn,
    raw_embed_fn,
    run_protocol,
    split_enroll_test,
)
from src.model import ECGEmbeddingNet
from src.synthetic import make_synthetic_table


def test_l2_normalize_rows_have_unit_norm():
    x = np.random.default_rng(0).normal(size=(5, 8))
    assert np.allclose(np.linalg.norm(l2_normalize(x), axis=1), 1)


def test_build_trials_templates_and_labels():
    # Dos personas con embeddings 2-D perfectamente separados.
    enroll = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=float)
    enroll_ids = np.array(["a", "a", "b", "b"])
    test = np.array([[1, 0.05], [0.05, 1]], dtype=float)
    test_ids = np.array(["a", "b"])
    t = build_trials(enroll, enroll_ids, test, test_ids)
    assert isinstance(t, Trials)
    assert len(t.genuine) == 2 and len(t.impostor) == 2
    assert t.genuine.min() > t.impostor.max()
    assert set(t.genuine_owner.tolist()) == {"a", "b"}
    assert t.n_owners() == 2


def test_unenrolled_test_person_counts_only_as_impostor():
    enroll = np.array([[1.0, 0.0]])
    t = build_trials(enroll, np.array(["a"]), np.array([[1.0, 0.0], [0.0, 1.0]]), np.array(["a", "z"]))
    assert len(t.genuine) == 1 and len(t.impostor) == 1


def test_build_trials_without_impostors_raises():
    with pytest.raises(EvaluationError):
        build_trials(np.array([[1.0, 0.0]]), np.array(["a"]), np.array([[1.0, 0.0]]), np.array(["a"]))


def _table(n_people=3, n_sessions=3, w=6):
    return make_synthetic_table(n_people=n_people, n_sessions=n_sessions, windows_per_session=w, seed=21)


def test_first_session_table_keeps_only_first_session_per_person():
    ft = first_session_table(_table())
    assert set(ft.session_ids.tolist()) == {"s1"}
    assert len(ft) == 3 * 6


def test_split_enroll_test_cross_session():
    enroll, test = split_enroll_test(_table(), "cross_session")
    assert set(enroll.session_ids.tolist()) == {"s1"}
    assert "s1" not in set(test.session_ids.tolist())
    assert len(enroll) + len(test) == len(_table())


def test_split_enroll_test_same_session_halves_first_session():
    enroll, test = split_enroll_test(_table(), "same_session")
    assert set(enroll.session_ids.tolist()) == {"s1"} == set(test.session_ids.tolist())
    assert len(enroll) == len(test) == 3 * 3     # mitad y mitad de las 6 ventanas de s1


def test_split_enroll_test_rejects_insufficient_data():
    t = _table(n_sessions=1)
    with pytest.raises(EvaluationError, match="cross_session"):
        split_enroll_test(t, "cross_session")


def test_split_enroll_test_unknown_mode():
    with pytest.raises(EvaluationError):
        split_enroll_test(_table(), "otra")


def test_run_protocol_with_model_and_raw_embeddings():
    t = _table()
    enroll, test = split_enroll_test(t, "cross_session")
    net = ECGEmbeddingNet(Config())
    trials = run_protocol(make_model_embed_fn(net), enroll, test)
    assert len(trials.genuine) > 0 and len(trials.impostor) > 0
    raw = run_protocol(raw_embed_fn, enroll, test)
    assert len(raw.genuine) == len(trials.genuine)


def test_model_embed_fn_leaves_no_gradients_and_returns_unit_vectors():
    net = ECGEmbeddingNet(Config())
    x = np.random.default_rng(0).normal(size=(5, 1, 650)).astype(np.float32)
    z = make_model_embed_fn(net)(x)
    assert z.shape == (5, 128) and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    assert all(p.grad is None for p in net.parameters())


import dataclasses

from src.data import split_polar
from src.evaluate import (
    ReportRow,
    bootstrap_eer_ci,
    compare_models,
    format_report,
    make_row,
)


def _trials(n_owners=4, sep=True, per=30, seed=0):
    rng = np.random.default_rng(seed)
    g, go, i, io = [], [], [], []
    for k in range(n_owners):
        g.append(rng.normal(0.8 if sep else 0.5, 0.05, per)); go += [f"o{k}"] * per
        i.append(rng.normal(0.2 if sep else 0.5, 0.05, per)); io += [f"o{k}"] * per
    return Trials(np.concatenate(g), np.array(go), np.concatenate(i), np.array(io))


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    t = _trials()
    point, _ = eer(t.genuine, t.impostor)
    lo, hi = bootstrap_eer_ci(t, n_boot=200, seed=0)
    assert lo - 1e-9 <= point <= hi + 1e-9
    assert (lo, hi) == bootstrap_eer_ci(t, n_boot=200, seed=0)


def test_bootstrap_ci_wider_when_scores_overlap():
    lo1, hi1 = bootstrap_eer_ci(_trials(sep=True), 200, 0)
    lo2, hi2 = bootstrap_eer_ci(_trials(sep=False), 200, 0)
    assert (hi2 - lo2) > (hi1 - lo1)


# --- Review Focus 4: menos de 3 personas ---
def test_bootstrap_ci_is_nan_with_fewer_than_three_people():
    lo, hi = bootstrap_eer_ci(_trials(n_owners=2), n_boot=100, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_make_row_fields_and_counts():
    t = _trials()
    thr = eer(t.genuine, t.impostor)[1]
    row = make_row("demo", t, thr, n_boot=50, seed=0)
    assert isinstance(row, ReportRow)
    assert row.n_genuine == len(t.genuine) and row.n_impostor == len(t.impostor)
    assert row.n_owners == 4 and 0 <= row.eer <= 1 and 0 <= row.auc <= 1


def test_format_report_prints_counts_and_na_ci():
    rows = [
        make_row("tres personas", _trials(), 0.5, 50, 0),
        make_row("dos personas", _trials(n_owners=2), 0.5, 50, 0),
    ]
    md = format_report(rows)
    assert md.startswith("|") and "tres personas" in md and "dos personas" in md
    assert "n/a" in md                       # IC no disponible con <3 personas
    assert f"{rows[0].n_genuine}/{rows[0].n_impostor}" in md


def test_compare_models_produces_all_sections_for_each_model():
    t = make_synthetic_table(n_people=6, n_sessions=3, windows_per_session=8, seed=31)
    split = split_polar(t, ("p4", "p5"))
    cfg = dataclasses.replace(Config(), bootstrap_samples=20)
    fns = {"red": make_model_embed_fn(ECGEmbeddingNet(cfg)), "cruda": raw_embed_fn}
    md = compare_models(fns, split, cfg)
    for model in ("red", "cruda"):
        assert f"{model} · vistos, entre sesiones (val)" in md
        assert f"{model} · NO vistos (hold-out), entre sesiones" in md
        assert f"{model} · vistos, misma sesión" in md
    # Cada fila de la tabla debe tener exactamente las 8 columnas del encabezado
    # (un "|" dentro de una etiqueta partiría la fila y desplazaría los valores).
    for line in md.splitlines():
        assert line.count("|") == 9, line


from src.checkpoint import save_checkpoint
from src.data import save_window_file
from src.evaluate import main as evaluate_main


def _write_polar_dir(tmp_path):
    t = make_synthetic_table(n_people=6, n_sessions=3, windows_per_session=8, seed=51)
    d = tmp_path / "polar"
    d.mkdir()
    for p in t.persons():
        for s in t.sessions_of(p):
            m = (t.participant_ids == p) & (t.session_ids == s)
            save_window_file(d / f"{p}__{s}.npz", t.windows[m], p, s, "rest", "synthetic")
    return d


def _fingerprint(polar_dir, holdout):
    from src.data import load_window_dir, table_fingerprint

    return table_fingerprint(split_polar(load_window_dir(polar_dir), holdout).train)


def _run_compare(tmp_path, monkeypatch, trained_holdout, fingerprint=None):
    import src.evaluate as ev

    current = dataclasses.replace(Config(), holdout_ids=("p4", "p5"), bootstrap_samples=10)
    monkeypatch.setattr(ev, "Config", lambda: current)
    polar = _write_polar_dir(tmp_path)
    trained = dataclasses.replace(Config(), holdout_ids=trained_holdout)
    fp = fingerprint or _fingerprint(polar, ("p4", "p5"))
    net = ECGEmbeddingNet(trained, n_classes=3)
    a = save_checkpoint(net, trained, tmp_path / "m", "polar-only", {}, fp)
    b = save_checkpoint(net, trained, tmp_path / "m", "finetune", {}, fp, parent="modelo_v0.pt")
    out = tmp_path / "rep.md"
    rc = evaluate_main(["compare", "--polar-dir", str(polar), "--polar-only", str(a),
                        "--finetuned", str(b), "--out", str(out)])
    return rc, out


def test_compare_cli_writes_report(tmp_path, monkeypatch):
    rc, out = _run_compare(tmp_path, monkeypatch, ("p4", "p5"))
    assert rc == 0
    md = out.read_text(encoding="utf-8")
    assert "Solo Polar" in md and "Preentrenado + fine-tuning" in md and "Señal cruda" in md


def test_compare_cli_refuses_when_holdout_differs_from_checkpoint(tmp_path, monkeypatch):
    rc, _ = _run_compare(tmp_path, monkeypatch, ("p0", "p1"))
    assert rc != 0


def test_compare_cli_accepts_same_holdout_in_another_order(tmp_path, monkeypatch):
    rc, _ = _run_compare(tmp_path, monkeypatch, ("p5", "p4"))
    assert rc == 0


def test_compare_cli_refuses_when_training_data_changed_since_checkpoint(tmp_path, monkeypatch):
    rc, _ = _run_compare(tmp_path, monkeypatch, ("p4", "p5"), fingerprint="datos-viejos")
    assert rc != 0


# --- Revisión final: la fila "misma sesión" no puede medirse con ventanas de entrenamiento ---
def test_same_session_row_is_not_computed_on_training_windows():
    t = make_synthetic_table(n_people=6, n_sessions=3, windows_per_session=8, seed=31)
    split = split_polar(t, ("p4", "p5"))
    cfg = dataclasses.replace(Config(), bootstrap_samples=5)
    seen = []

    def spy(windows):
        seen.append(windows.copy())
        return raw_embed_fn(windows)

    compare_models({"x": spy}, split, cfg)
    train_keys = {w.tobytes() for w in split.train.windows}
    same_session_calls = seen[-2:]          # registro y prueba del protocolo misma-sesión (últimos dos)
    assert all(w.tobytes() not in train_keys for arr in same_session_calls for w in arr)


# --- template_scores: puntajes ventana-vs-cada-plantilla (usado por build_trials y por try_me) ---
def test_template_scores_shape_order_and_values():
    from src.evaluate import template_scores

    enroll = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    owners, scores = template_scores(enroll, np.array(["b", "b", "a"]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert list(owners) == ["a", "b"]                       # ordenados alfabéticamente
    assert scores.shape == (2, 2)
    assert np.allclose(scores, [[0.0, 1.0], [1.0, 0.0]])    # ventana 1 ~ plantilla "b"; ventana 2 ~ "a"

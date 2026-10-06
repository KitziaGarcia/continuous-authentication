from matplotlib.figure import Figure

import src.plotting as plotting
from src.plotting import save_figure


def _fig():
    fig = Figure(figsize=(2, 2))
    fig.subplots().plot([0, 1], [0, 1])
    return fig


def test_save_figure_writes_png_and_returns_the_requested_path(tmp_path):
    out = save_figure(_fig(), tmp_path / "sub" / "a.png")
    assert out == tmp_path / "sub" / "a.png" and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_figure_falls_back_to_another_name_when_file_is_locked(tmp_path, monkeypatch, capsys):
    target = tmp_path / "try_me.png"
    target.write_bytes(b"imagen abierta en el visor")
    real_savefig = Figure.savefig

    def locked(self, fname, *a, **k):          # simula Windows: no se puede reemplazar un archivo en uso
        if str(fname) == str(target):
            raise OSError(22, "Invalid argument")
        return real_savefig(self, fname, *a, **k)

    monkeypatch.setattr(Figure, "savefig", locked)
    out = save_figure(_fig(), target)
    assert out != target and out.parent == tmp_path and out.name.startswith("try_me_") and out.suffix == ".png"
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert target.read_bytes() == b"imagen abierta en el visor"          # el original no se tocó
    assert "en uso" in capsys.readouterr().out

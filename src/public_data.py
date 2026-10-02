"""Dataset público ECG-ID (PhysioNet) -> ventanas de 5 s a 130 Hz, para preentrenar.

Se usa SOLO para preentrenar: son otras derivaciones y otro hardware que la banda Polar, por eso
después se hace fine-tuning con datos del Polar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Config
from src.data import save_window_file
from src.windowing import signal_to_windows


def download_ecgid(dest) -> None:
    import wfdb  # import perezoso: solo hace falta para descargar/leer

    wfdb.dl_database("ecgiddb", dl_dir=str(dest))


def iter_ecgid_records(root):
    """Recorre Person_XX/rec_N. Canal 0 = ECG I crudo (el canal 1 ya viene filtrado por ellos)."""
    import wfdb

    for hea in sorted(Path(root).rglob("rec_*.hea")):
        rec = wfdb.rdrecord(str(hea.with_suffix("")))
        yield hea.parent.name, hea.stem, rec.p_signal[:, 0], float(rec.fs)


def build_public_windows(root, out_dir, cfg: Config) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for person, rec, sig, fs in iter_ecgid_records(root):
        windows = signal_to_windows(sig, fs, cfg)
        if len(windows) == 0:
            continue
        # Relleno con ceros para que "rec_10" ordene después de "rec_2" (orden cronológico).
        session = f"rec_{int(rec.split('_')[1]):02d}"
        pid = f"pub_{person}"      # prefijo: nunca choca con un participante Polar
        save_window_file(out_dir / f"{pid}__{session}.npz", windows, pid, session, "rest", "public")
        written += 1
    return written


def main(argv=None) -> int:
    cfg = Config()
    p = argparse.ArgumentParser(prog="python -m src.public_data")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dest", default="data/raw/public/ecgid")
    b = sub.add_parser("build")
    b.add_argument("--raw", default="data/raw/public/ecgid")
    b.add_argument("--out", default=f"{cfg.data_dir}/public")
    a = p.parse_args(argv)
    if a.cmd == "download":
        download_ecgid(a.dest)
    else:
        print(f"{build_public_windows(a.raw, a.out, cfg)} archivos escritos en {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

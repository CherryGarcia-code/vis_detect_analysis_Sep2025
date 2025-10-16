from pathlib import Path
import argparse
import logging

from src.session_io import load_session
from src.integrations.bombcell_wrapper import detect_with_bombcell


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Bombcell wrapper for a session and save outputs"
    )
    p.add_argument("session_pkl", type=Path, help="Path to session pickle")
    p.add_argument(
        "--kilosort-dir", type=Path, default=None, help="Optional Kilosort/Phy folder"
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to save outputs (default: notebooks/<session>/bombcell)",
    )
    p.add_argument(
        "--raw-data-file",
        type=Path,
        default=None,
        help="Optional path to raw SpikeGLX AP file (e.g. *.ap.bin)",
    )
    p.add_argument(
        "--meta-file",
        type=Path,
        default=None,
        help="Optional path to SpikeGLX .meta file to use",
    )
    p.add_argument(
        "--reextract-raw",
        action="store_true",
        help="Force re-extraction of raw waveforms (delete previous intermediates)",
    )
    p.add_argument(
        "--n-raw-spikes",
        type=int,
        default=None,
        help="Number of raw spikes to extract per unit (overrides default)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    session = load_session(str(args.session_pkl))
    session_name = session.session_name or args.session_pkl.stem
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else Path("notebooks") / session_name / "bombcell"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df = detect_with_bombcell(
        session,
        kilosort_dir=str(args.kilosort_dir) if args.kilosort_dir is not None else None,
        out_dir=str(out_dir),
        raw_data_file=str(args.raw_data_file)
        if args.raw_data_file is not None
        else None,
        meta_file=str(args.meta_file) if args.meta_file is not None else None,
        reextract_raw=bool(args.reextract_raw),
        n_raw_spikes=int(args.n_raw_spikes) if args.n_raw_spikes is not None else None,
    )
    if df is None or df.empty:
        logging.info("No Bombcell metrics produced (package missing or no data).")
    else:
        logging.info("Bombcell wrapper produced %d rows; saved to %s", len(df), out_dir)


if __name__ == "__main__":
    main()

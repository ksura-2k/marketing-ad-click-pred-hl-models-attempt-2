"""Entry point.  Run from the repository root:

    python main.py                          # full pipeline (split -> CV -> select -> test -> save)
    python main.py --data path/to/file.csv  # explicit dataset
    python main.py --stage predict --input new_rows.csv --output scored.csv
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ad-click prediction pipeline (Sprint 4 baseline)")
    ap.add_argument("--stage", choices=["all", "predict"], default="all")
    ap.add_argument("--data", help="training CSV (default: $AD_CLICK_DATA, ./data/*.csv, ./*.csv)")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--input", help="CSV to score (predict stage)")
    ap.add_argument("--output", default=str(ROOT / "outputs" / "scored.csv"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", mode="w")])

    if args.stage == "predict":
        if not args.input:
            ap.error("--input is required for --stage predict")
        models.predict(args.input, out / "models" / "model.joblib", args.output)
        return 0

    data = Path(args.data) if args.data else models.find_data(ROOT)
    summary = models.run_pipeline(data, out)
    t = summary["test"]
    print(f"\nchosen model: {summary['chosen']}")
    print(f"test PR-AUC {t['pr_auc']:.4f} (floor {t['floor_pr_auc']:.4f}) | ROC-AUC {t['roc_auc']:.3f} | "
          f"log-loss {t['log_loss']:.4f} (floor {t['floor_log_loss']:.4f}) | Brier {t['brier']:.4f} (floor {t['floor_brier']:.4f})")
    print(f"reports: {out / 'reports'}  model: {out / 'models' / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

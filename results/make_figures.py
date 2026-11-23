#!/usr/bin/env python
"""Build the paper figures from output/articles.csv (+ hand labels if filled in).

    .venv/bin/python make_figures.py
    .venv/bin/python make_figures.py --labels output/labelling/label_me.xlsx \
        --gold-run output_jan_week_pdf          # labels drawn from another run
Writes output/figures/*.pdf (vector) and *.png (600 dpi).
"""
import argparse
from pathlib import Path

from newspaper_pipeline import figures

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, default=HERE / "output")
ap.add_argument("--labels", type=Path, default=None,
                help="labelling sheet (default: label_me.xlsx if present, else label_sample.xlsx)")
ap.add_argument("--gold-run", type=Path, default=None,
                help="run folder the labelled articles come from (default: the --run itself)")
a = ap.parse_args()
lab_dir = a.run / "labelling"
labels = a.labels or next((p for p in (lab_dir / "label_me.xlsx", lab_dir / "label_sample.xlsx")
                           if p.exists()), lab_dir / "label_sample.xlsx")
gold_csv = None
if a.gold_run:
    gold_csv = next((p for p in (a.gold_run / "articles_with_llm.csv", a.gold_run / "articles.csv")
                     if p.exists()), None)
done = figures.make_all(a.run / "articles.csv", labels, a.run / "figures", gold_csv)
print("Made:", ", ".join(done))
if "fig3" not in done:
    print(f"fig3 (gold-standard accuracy) needs the 'my label' column filled in {labels}")

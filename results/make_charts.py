#!/usr/bin/env python
"""Rebuild the summary tables + all charts (output/tables, output/charts/,
output/charts/methods/, output/charts/topics/) from
output/articles.csv without re-running OCR, sentiment or topics.

    .venv/bin/python make_charts.py [--run output]
"""
import argparse
from pathlib import Path

import pandas as pd

from newspaper_pipeline import compare, method_charts, summarize, topic_charts

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, default=HERE / "output")
a = ap.parse_args()
df = pd.read_csv(a.run / "articles.csv")
st = summarize.run(df, a.run / "tables", a.run / "charts")
agree, dist = compare.agreement(df)
agree.to_csv(a.run / "tables" / "method_agreement.csv", index=False)
dist.to_csv(a.run / "tables" / "method_label_distribution.csv")
compare.topic_chart(df, a.run / "charts" / "methods_by_topic.png")
made = method_charts.run(df, a.run / "charts" / "methods")
made += topic_charts.run(df, a.run / "charts" / "topics", a.run / "embeddings.npz")
print("Extra charts:", ", ".join(made))
print(f"Charts rebuilt from {len(df)} articles (charts drawn with {st['chart_method']}).")

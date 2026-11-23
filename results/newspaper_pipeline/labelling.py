"""Hand-labelling sample export and accuracy evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

LABELS = ["positive", "neutral", "negative"]


def export_sample(df: pd.DataFrame, path: Path, n: int = config.LABEL_SAMPLE_SIZE):
    """Random sample with an empty 'my label' column.

    Model predictions are deliberately NOT included, so they can't bias the labels.
    An existing file is never overwritten (it may hold your labels).
    """
    if path.exists():
        return False
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    sample = df.sample(n=min(n, len(df)), random_state=config.RANDOM_SEED)
    wb = Workbook()
    ws = wb.active
    ws.title = "label_me"
    cols = ["article_id", "date", "page_type", "headline", "text", "my label", "notes"]
    ws.append(cols)
    for _, r in sample.iterrows():
        ws.append([r["article_id"], r["date"], r["page_type"], r["headline"],
                   r["text"][:32000], None, None])
    widths = {"A": 28, "B": 11, "C": 10, "D": 40, "E": 100, "F": 12, "G": 25}
    for c, w in widths.items():
        ws.column_dimensions[c].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True)
    hl = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        row[5].fill = hl
    dv = DataValidation(type="list", formula1='"positive,neutral,negative"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"F2:F{len(sample) + 1}")
    ws.freeze_panes = "A2"
    guide = wb.create_sheet("instructions")
    for line in [
        "Read each article and choose positive / neutral / negative in the 'my label' column.",
        "Judge the overall tone of the article text (not whether you agree with it).",
        "neutral = mostly factual reporting with no clear positive or negative slant.",
        "Leave 'my label' empty if the OCR text is too garbled to judge, and say so in 'notes'.",
        "Then run:  python evaluate_labels.py",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 110
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return True


def evaluate(label_path: Path, articles_csv: Path, out_dir: Path) -> str:
    """Join your labels to articles.csv and compare all sentiment methods."""
    from . import compare
    labels = pd.read_excel(label_path, sheet_name="label_me")
    labels["my label"] = labels["my label"].astype(str).str.strip().str.lower()
    labels = labels[labels["my label"].isin(LABELS)]
    arts = pd.read_csv(articles_csv)
    m = labels[["article_id", "my label"]].merge(arts, on="article_id", how="inner")
    if m.empty:
        return "No labelled rows found yet - fill in the 'my label' column first."
    return compare.evaluate(m, out_dir)

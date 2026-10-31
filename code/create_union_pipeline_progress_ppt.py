from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt

OUT = Path("outputs/reports/union_pipeline_progress_report_updated.pptx")
OUT.parent.mkdir(parents=True, exist_ok=True)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def title(slide, text):
    box = slide.shapes.add_textbox(
        Inches(0.45), Inches(0.25), Inches(12.4), Inches(0.55)
    )
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(30)
    p.font.bold = True


def text(slide, txt, x, y, w, h, size=20, bold=False):
    box = slide.shapes.add_textbox(
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = txt
    p.font.size = Pt(size)
    p.font.bold = bold
    return box


def table(slide, rows, x, y, w, h, size=14):
    shp = slide.shapes.add_table(
        len(rows), len(rows[0]),
        Inches(x), Inches(y), Inches(w), Inches(h)
    )
    tbl = shp.table

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)

            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(size)
                if i == 0:
                    p.font.bold = True

    return tbl


# ============================================================================
# Slide 1
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Political filtering and SFT pipeline: current status")

table(s, [
    ["Stage", "Status"],
    ["Binary political gate validation", "complete"],
    ["Orientation classifier validation", "complete"],
    ["Union filtering on DOLCI", "complete"],
    ["Llama 4-way classification", "complete"],
    ["Reparsing", "complete"],
    ["Intervention row lists", "complete + verified"],
    ["OLMo-3 QLoRA SFT", "full-condition training started"],
], 0.8, 1.3, 11.7, 4.4, 18)


# ============================================================================
# Slide 2
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Old filtering approach vs new filtering approach")

table(s, [
    ["", "Old approach", "New approach"],
    [
        "Stage 1",
        "ModernBERT binary gate",
        "ModernBERT OR embedding classifier"
    ],
    [
        "Embedding model",
        "not used",
        "BAAI/bge-base-en-v1.5 + logistic regression"
    ],
    [
        "Embedding threshold",
        "—",
        "0.4"
    ],
    [
        "Stage 2",
        "Llama 3.3 70B 4-way",
        "Llama 3.3 70B 4-way"
    ],
    [
        "Final labels",
        "non-political / neutral / left / right",
        "non-political / neutral / left / right"
    ],
], 0.6, 1.2, 12.1, 4.4, 16)

text(
    s,
    "Main change: the new first-stage union gate was selected to improve political recall before final Llama classification.",
    0.9, 6.0, 11.5, 0.7, 20, True
)


# ============================================================================
# Slide 3
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Binary political gate validation datasets")

table(s, [
    ["Validation dataset", "Purpose", "Size / labels"],
    [
        "20 Newsgroups",
        "External binary politics vs non-politics validation",
        "7,072 rows in same-row comparison"
    ],
    [
        "BBC News",
        "External binary politics vs non-politics validation",
        "1,225 rows"
    ],
    [
        "Manual target-domain labels",
        "Validation on SFT-style / target-domain examples",
        "527 manually labelled rows"
    ],
], 0.7, 1.25, 12.0, 3.2, 17)

text(
    s,
    "Target-domain manual validation combines manually labelled SFT examples and corrected OOD review data.",
    1.0, 5.0, 11.2, 0.8, 20, True
)


# ============================================================================
# Slide 4
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Binary gate validation results")

table(s, [
    ["Dataset", "Method", "Precision", "Recall", "F1"],
    ["20 Newsgroups", "ModernBERT", "0.476", "0.778", "0.591"],
    ["20 Newsgroups", "Embedding t=0.3", "0.553", "0.969", "0.704"],
    ["20 Newsgroups", "Union t=0.3", "0.457", "0.982", "0.623"],
    ["BBC News", "ModernBERT", "0.619", "0.740", "0.674"],
    ["BBC News", "Embedding t=0.3", "0.790", "0.996", "0.881"],
    ["BBC News", "Union t=0.3", "0.629", "0.996", "0.771"],
], 0.55, 1.05, 12.2, 3.9, 14)

text(
    s,
    "External validation showed that embedding-based candidate selection strongly increased political recall.",
    0.8, 5.35, 11.7, 0.7, 20, True
)


# ============================================================================
# Slide 5
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Target-domain manual validation")

table(s, [
    ["Method", "Accuracy", "Precision", "Recall", "F1"],
    ["Embedding t=0.2", "0.634", "0.510", "0.900", "0.651"],
    ["Embedding t=0.3", "0.683", "0.555", "0.835", "0.667"],
    ["Embedding t=0.4", "0.719", "0.606", "0.745", "0.668"],
    ["ModernBERT", "0.934", "0.919", "0.905", "0.912"],
    ["Union t=0.3", "0.729", "0.586", "0.970", "0.731"],
    ["Union t=0.4", "0.793", "0.656", "0.955", "0.778"],
    ["Union t=0.5", "0.824", "0.697", "0.945", "0.803"],
], 0.55, 1.0, 12.2, 4.4, 15)

text(
    s,
    "Final binary candidate gate: ModernBERT OR embedding classifier at threshold 0.4.",
    1.0, 5.8, 11.0, 0.6, 22, True
)


# ============================================================================
# Slide 6
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Political orientation validation dataset")

table(s, [
    ["Dataset", "Labels", "Size"],
    ["Tanise slant validation set", "300 left", "300"],
    ["", "300 center-neutral", "300"],
    ["", "300 right", "300"],
    ["Total", "balanced 3-way dataset", "900"],
], 0.8, 1.3, 11.7, 2.8, 18)

text(
    s,
    "Purpose: validate the second-stage left / center-neutral / right orientation classifier.",
    1.0, 4.7, 11.0, 0.8, 21, True
)


# ============================================================================
# Slide 7
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Orientation classifier validation results")

table(s, [
    ["Method", "Accuracy"],
    ["Original Llama 3.3 70B 4-way prompt", "66.33% strict"],
    ["Original Llama 4-way, target-label only", "69.91%"],
    ["Dmitry 3-way prompt", "64.89%"],
    ["Embedding 3-way classifier", "58.44%"],
    ["NLI 3-way classifier", "15.00%"],
], 0.8, 1.2, 11.7, 3.4, 18)

text(
    s,
    "Selected orientation classifier: original Llama 3.3 70B 4-way prompt.",
    1.0, 5.3, 11.0, 0.7, 23, True
)


# ============================================================================
# Slide 8
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Original Llama 4-way slant validation details")

table(s, [
    ["Gold label", "Pred. neutral", "Pred. left", "Pred. non-pol.", "Pred. right"],
    ["Center-neutral", "160", "84", "35", "21"],
    ["Left", "55", "229", "8", "8"],
    ["Right", "47", "42", "3", "208"],
], 0.5, 1.3, 12.3, 2.6, 16)

table(s, [
    ["Label", "Precision", "Recall", "F1"],
    ["Left", "0.645", "0.763", "0.699"],
    ["Center-neutral", "0.611", "0.533", "0.569"],
    ["Right", "0.878", "0.693", "0.775"],
], 1.3, 4.4, 10.7, 2.0, 17)


# ============================================================================
# Slide 9
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "DOLCI candidate pool: old vs new")

table(s, [
    ["Split", "Old ModernBERT candidates", "New union t=0.4 candidates"],
    ["Instruct", "131,746", "233,571"],
    ["Think", "55,871", "115,627"],
    ["Combined", "187,617", "349,198"],
], 0.8, 1.4, 11.7, 2.6, 18)

text(
    s,
    "Candidate increase: +161,581 rows (+86.1%).",
    2.1, 4.7, 9.0, 0.7, 26, True
)


# ============================================================================
# Slide 10
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Llama 4-way output after reparsing")

table(s, [
    ["Label", "Count"],
    ["Non-political", "198,350"],
    ["Center-neutral", "88,083"],
    ["Left", "35,882"],
    ["Right", "4,983"],
    ["Unparsed", "21,900"],
], 0.8, 1.2, 5.6, 3.9, 18)

table(s, [
    ["Metric", "Value"],
    ["Total candidates", "349,198"],
    ["Parsed", "327,298"],
    ["Parse rate", "93.73%"],
    ["Confirmed political", "128,948"],
    ["Ideological", "40,865"],
    ["Left/right ratio", "7.20 : 1"],
], 6.8, 1.2, 5.7, 3.9, 18)


# ============================================================================
# Slide 11
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Old vs new final political counts")

table(s, [
    ["Measure", "Old pipeline", "New pipeline", "Change"],
    ["Confirmed political", "93,732", "128,948", "+35,216"],
    ["Ideological", "32,816", "40,865", "+8,049"],
    ["Center-neutral", "60,916", "88,083", "+27,167"],
    ["Left", "28,861", "35,882", "+7,021"],
    ["Right", "3,955", "4,983", "+1,028"],
], 0.7, 1.3, 12.0, 3.6, 17)


# ============================================================================
# Slide 12
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Verified intervention datasets")

table(s, [
    ["Condition", "Final rows", "Rows removed"],
    ["Full", "4,420,290", "0"],
    ["NoPol", "4,291,342", "128,948"],
    ["NoIdeology", "4,379,425", "40,865"],
    ["NoLeft", "4,384,408", "35,882"],
    ["NoRight", "4,415,307", "4,983"],
    ["LRBalanced", "4,389,391", "30,899"],
], 0.8, 1.2, 11.7, 4.1, 18)

text(
    s,
    "All six row-filtering conditions were verified by streaming through the complete DOLCI corpus.",
    0.9, 5.65, 11.5, 0.7, 20, True
)


# ============================================================================
# Slide 13
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Intervention removals by split")

table(s, [
    ["Condition", "Instruct removed", "Think removed", "Total"],
    ["NoPol", "73,098", "55,850", "128,948"],
    ["NoIdeology", "23,984", "16,881", "40,865"],
    ["NoLeft", "20,493", "15,389", "35,882"],
    ["NoRight", "3,491", "1,492", "4,983"],
    ["LRBalanced", "17,662", "13,237", "30,899"],
], 0.8, 1.3, 11.7, 3.7, 18)

text(
    s,
    "LRBalanced leaves 4,983 left and 4,983 right rows.",
    1.2, 5.6, 10.8, 0.6, 22, True
)


# ============================================================================
# Slide 14
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
title(s, "Current OLMo-3 SFT setup")

table(s, [
    ["Item", "Value"],
    ["Base model", "allenai/Olmo-3-1025-7B"],
    ["Training", "distributed QLoRA SFT"],
    ["GPUs", "4 × RTX A6000"],
    ["Global effective batch", "64"],
    ["Maximum sequence length", "2048"],
    ["Full-condition rows", "4,420,290"],
    ["Full-condition optimizer steps", "69,067"],
], 0.8, 1.1, 11.7, 4.3, 18)

text(
    s,
    "Each intervention starts independently from the same OLMo-3 base checkpoint.",
    1.0, 5.8, 11.0, 0.6, 21, True
)


prs.save(OUT)
print("Saved:", OUT)

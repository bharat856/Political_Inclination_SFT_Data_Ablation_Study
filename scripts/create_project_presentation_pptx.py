#!/usr/bin/env python3
"""Create a PPTX deck for the political SFT project.

This version uses only the Python standard library, so it works on the cluster
even when python-pptx is unavailable.

Output:
  presentations/political_sft_project/political_sft_project_results_v2.pptx
"""

from __future__ import annotations

import html
import zipfile
from pathlib import Path


OUT_DIR = Path("presentations/political_sft_project")
OUT = OUT_DIR / "political_sft_project_results_v2.pptx"

SLIDE_W = 13_333_333
SLIDE_H = 7_500_000


# ---------------------------------------------------------------------
# Slide content
# ---------------------------------------------------------------------

slides = [
    {
        "section": "Title",
        "title": "Political Positioning in Open Language Models",
        "subtitle": (
            "A controlled study of supervised fine-tuning data detection, "
            "ideological imbalance, and intervention datasets."
        ),
        "bullets": [
            "Project focus: identify political examples in SFT data and test whether removing/rebalancing them changes model behavior.",
            "Current status: classifier pipeline built, DOLCI/OLMo-3 intervention datasets created, token statistics computed.",
        ],
    },
    {
        "section": "Motivation",
        "title": "Motivation",
        "bullets": [
            "Instruction-tuned models may inherit behavioral tendencies from SFT data.",
            "Political examples are only a small part of SFT, but they may have disproportionate influence on political responses.",
            "A controlled intervention requires knowing which examples are political, which are ideological, and how much data is being removed.",
            "The goal is not only to detect bias, but to create measurable training variants for causal comparison.",
        ],
    },
    {
        "section": "Research Questions",
        "title": "Research Questions",
        "bullets": [
            "RQ1: Can political examples be detected reliably inside large SFT datasets?",
            "RQ2: Is the political SFT subset ideologically imbalanced?",
            "RQ3: What happens if we remove or rebalance political/ideological SFT examples?",
            "RQ4: Are downstream model changes visible despite the intervention being small in token mass?",
        ],
    },
    {
        "section": "Pipeline",
        "title": "End-to-End Pipeline",
        "bullets": [
            "1. Build/evaluate political classifiers on benchmark and manual data.",
            "2. Flatten DOLCI/OLMo-3 SFT datasets into row-level examples.",
            "3. Run ModernBERT as a cheap binary political gate.",
            "4. Run Llama-3.3-70B as a 4-way classifier on candidate rows.",
            "5. Create row-id lists and filtered intervention JSONL datasets.",
            "6. Train full and intervention SFT variants under the same recipe.",
            "7. Evaluate political positioning, refusal behavior, utility, and general performance.",
        ],
    },
    {
        "section": "Benchmarks",
        "title": "Manifesto Benchmark Construction",
        "bullets": [
            "Source: Manifesto Project / MARPOR political party documents.",
            "Used document-level RILE score as left/center/right signal.",
            "Label rule: RILE < -5 = left, -5 to +5 = center, RILE > +5 = right.",
            "Long manifestos were split into chunks; each chunk inherited the document label.",
            "Important limitation: chunk text may not express the same ideology as the full document-level RILE score.",
        ],
    },
    {
        "section": "Benchmarks",
        "title": "Prompted LLMs on Manifesto Benchmark",
        "table": {
            "headers": ["Benchmark", "Model", "Chunk Acc.", "Doc Vote Acc.", "Main Pattern"],
            "rows": [
                ["v1", "Llama-3.3-70B", "0.6905", "0.7407", "Best overall"],
                ["v1", "Gemma-4-31B", "0.6905", "0.6667", "Close to Llama"],
                ["v1", "Qwen3.5-27B", "0.3333", "0.3333", "All-left collapse"],
                ["v2", "Llama-3.3-70B", "0.5156", "0.5440", "Overpredicts center"],
                ["v2", "Gemma-4-31B", "0.5044", "0.5337", "Overpredicts right"],
                ["v2", "Qwen3.5-27B", "0.3333", "0.3523", "All-left collapse"],
            ],
        },
        "takeaway": "The benchmark is useful diagnostically, but noisy because RILE is document-level while classification is chunk-level.",
    },
    {
        "section": "Benchmarks",
        "title": "Encoder Benchmarking",
        "table": {
            "headers": ["Encoder", "Chunk Acc.", "Macro F1", "Observed Behavior"],
            "rows": [
                ["ModernBERT-large", "0.4194", "0.4190", "Weak but not collapsed"],
                ["DeBERTa-v3-large", "0.3629", "0.1775", "All-right collapse"],
                ["mDeBERTa-v3-base", "0.3065", "0.1564", "All-center collapse"],
            ],
        },
        "takeaway": "ModernBERT was the most stable encoder, so it became the practical binary gate.",
    },
    {
        "section": "Benchmarks",
        "title": "Cleaner External Benchmarks",
        "table": {
            "headers": ["Dataset", "Model", "Accuracy", "Notes"],
            "rows": [
                ["v3a ideology", "ModernBERT-large", "1.0000", "Perfect on clean labels"],
                ["v3a ideology", "Llama-3.3-70B", "0.9437", "Strong prompted result"],
                ["v3a ideology", "Qwen3.5-27B", "0.7312", "0.9832 on parsed rows"],
                ["v3b senator tweets", "ModernBERT-large", "0.8688", "Macro F1 0.8687"],
                ["v3b senator tweets", "Llama-3.3-70B", "0.8344", "Overpredicted left"],
            ],
        },
        "takeaway": "Clean text-level political labels are much easier than Manifesto chunk labels.",
    },
    {
        "section": "Classifier Design",
        "title": "Why a Two-Stage Classifier?",
        "bullets": [
            "Running a large LLM over all 4.42M SFT rows is expensive.",
            "Keyword filtering is cheap but too crude to be trusted as a final label.",
            "ModernBERT is used first as a cheap binary political/non-political gate.",
            "Llama is then used only on candidate rows for richer 4-way labels.",
            "This separates high-recall candidate detection from more precise political categorization.",
        ],
    },
    {
        "section": "Classifier Design",
        "title": "Label Schema",
        "table": {
            "headers": ["Label", "Meaning", "Use in Interventions"],
            "rows": [
                ["0 / non_political", "False positive or no meaningful political content", "Kept"],
                ["1 / left", "Progressive/liberal/left ideological framing", "Removed or rebalanced"],
                ["2 / center_neutral", "Political but factual, balanced, or procedural", "Removed only in no_pol"],
                ["3 / right", "Conservative/right ideological framing", "Removed or rebalanced"],
                ["unparsed", "LLM output could not be parsed", "Kept as unknown"],
            ],
        },
        "takeaway": "Unparsed rows are not treated as confirmed political examples.",
    },
    {
        "section": "Classifier Results",
        "title": "Manual SFT Review Results",
        "table": {
            "headers": ["Metric", "Value"],
            "rows": [
                ["4-way reviewed rows", "192"],
                ["4-way accuracy", "0.5833"],
                ["Non-political recall", "0.4757"],
                ["Left recall", "0.9062"],
                ["Center-neutral recall", "0.5833"],
                ["Right recall", "0.8571"],
                ["Binary political recall", "0.9885"],
            ],
        },
        "takeaway": "Llama catches political content well in binary form, but overpredicts political labels in 4-way classification.",
    },
    {
        "section": "Classifier Results",
        "title": "ModernBERT Binary Gate",
        "table": {
            "headers": ["Evaluation", "Rows", "Accuracy", "Political Recall", "Non-pol Recall"],
            "rows": [
                ["Initial validation", "52", "0.9423", "1.0000", "0.9000"],
                ["Combined validation", "127", "0.8898", "0.8511", "0.9125"],
                ["Corrected OOD sanity", "299", "0.9666", "0.9400", "0.9799"],
            ],
        },
        "takeaway": "The corrected OOD score is a sanity/model-fit check because corrected OOD data was included in the combined pool.",
    },
    {
        "section": "Classifier Results",
        "title": "Initial OOD Evaluation Before Correction",
        "table": {
            "headers": ["Metric", "Value"],
            "rows": [
                ["Rows", "299"],
                ["Accuracy", "0.6756"],
                ["Macro F1", "0.6181"],
                ["Political recall", "0.4175"],
                ["Non-political recall", "0.8112"],
                ["Confusion matrix", "[[159, 37], [60, 43]]"],
            ],
        },
        "takeaway": "The main failure mode was missing neutral/center political examples out of distribution.",
    },
    {
        "section": "DOLCI Data",
        "title": "DOLCI / OLMo-3 SFT Data",
        "table": {
            "headers": ["Split", "Rows", "Approx. File Size", "Notes"],
            "rows": [
                ["Dolci-Instruct-SFT-7B", "2,152,112", "14 GB", "Instruction-style conversations"],
                ["Dolci-Think-SFT-7B", "2,268,178", "150 GB", "Long reasoning / think traces"],
                ["Total", "4,420,290", "164 GB", "Flattened JSONL row-level examples"],
            ],
        },
        "takeaway": "Think has only slightly more rows than Instruct, but dominates token mass due to long reasoning traces.",
    },
    {
        "section": "DOLCI Data",
        "title": "ModernBERT Candidate Filtering on DOLCI",
        "table": {
            "headers": ["Split", "Total Rows", "Political Candidates", "Candidate Rate"],
            "rows": [
                ["Instruct", "2,152,112", "131,746", "6.12%"],
                ["Think", "2,268,178", "55,871", "2.46%"],
                ["Total", "4,420,290", "187,617", "4.24%"],
            ],
        },
        "takeaway": "The binary gate reduced 4.42M rows to 187,617 candidates for Llama 4-way classification.",
    },
    {
        "section": "DOLCI Results",
        "title": "Llama 4-Way Classification of Candidates",
        "table": {
            "headers": ["Label", "Rows", "Share of Candidates", "Interpretation"],
            "rows": [
                ["non_political", "88,634", "47.24%", "Binary false positives"],
                ["center_neutral", "60,916", "32.47%", "Political, balanced/factual"],
                ["left", "28,861", "15.38%", "Ideological left"],
                ["right", "3,955", "2.11%", "Ideological right"],
                ["unparsed", "5,251", "2.80%", "Kept as unknown"],
            ],
        },
        "takeaway": "Confirmed political rows = 93,732. Ideological rows = 32,816.",
    },
    {
        "section": "DOLCI Results",
        "title": "Ideological Imbalance",
        "table": {
            "headers": ["Measure", "Left", "Right", "Left/Right Ratio"],
            "rows": [
                ["Rows", "28,861", "3,955", "7.30 : 1"],
                ["Tokens", "21.68M", "5.29M", "4.10 : 1"],
                ["Avg. tokens/row", "751", "1,337", "0.56 : 1"],
            ],
        },
        "takeaway": "Left examples are much more frequent, but right examples are longer on average, reducing the token-level imbalance.",
    },
    {
        "section": "DOLCI Results",
        "title": "Rows vs Tokens",
        "table": {
            "headers": ["Group", "Rows", "Row Share", "Tokens", "Token Share"],
            "rows": [
                ["not_candidate", "4,232,673", "95.76%", "23.94B", "99.33%"],
                ["non_political candidates", "88,634", "2.01%", "76.46M", "0.32%"],
                ["center_neutral", "60,916", "1.38%", "53.14M", "0.22%"],
                ["left", "28,861", "0.65%", "21.68M", "0.09%"],
                ["right", "3,955", "0.09%", "5.29M", "0.02%"],
                ["unparsed", "5,251", "0.12%", "5.52M", "0.02%"],
            ],
        },
        "takeaway": "Political rows are 2.12% of examples but only about 0.33% of total SFT token mass.",
    },
    {
        "section": "DOLCI Results",
        "title": "Why Token Counts Matter",
        "bullets": [
            "Rows count examples; tokens count actual training text.",
            "SFT loss is computed over tokens, not over rows.",
            "A short political row and a long reasoning row are both one row, but they contribute very different amounts of training signal.",
            "Therefore, reporting token share prevents overstating the size of the intervention.",
            "Interpretation: any downstream effect is notable because the removed political token mass is small; weak effects are also explainable.",
        ],
    },
    {
        "section": "Interventions",
        "title": "Created Intervention Datasets",
        "table": {
            "headers": ["Variant", "Removed Labels", "Rows Removed", "Rows Kept", "Purpose"],
            "rows": [
                ["no_pol", "center + left + right", "93,732", "4,326,558", "Remove all confirmed political data"],
                ["no_ideology", "left + right", "32,816", "4,387,474", "Remove ideological data only"],
                ["no_left", "left", "28,861", "4,391,429", "Directional left ablation"],
                ["no_right", "right", "3,955", "4,416,335", "Directional right ablation"],
                ["lr_balanced", "extra left", "24,906", "4,395,384", "Balance left/right by row count"],
            ],
        },
        "takeaway": "All variants were built as filtered JSONL files and verified against expected row counts.",
    },
    {
        "section": "Interventions",
        "title": "Intervention Design Logic",
        "table": {
            "headers": ["Comparison", "Question Answered"],
            "rows": [
                ["Full vs no_pol", "Does removing all political SFT content change model behavior?"],
                ["Full vs no_ideology", "Does removing ideological content only matter?"],
                ["Full vs lr_balanced", "Does left/right imbalance matter if political content remains?"],
                ["Full vs no_left", "What is the effect of removing left-labeled examples?"],
                ["Full vs no_right", "What is the effect of removing right-labeled examples?"],
            ],
        },
        "takeaway": "The design supports causal comparisons rather than only one broad removal condition.",
    },
    {
        "section": "Training",
        "title": "Planned Controlled SFT Variants",
        "table": {
            "headers": ["Variant", "Training Data"],
            "rows": [
                ["OLMo3-SFT-Full", "Original DOLCI Instruct + Think"],
                ["OLMo3-SFT-NoPol", "DOLCI with center + left + right removed"],
                ["OLMo3-SFT-NoIdeology", "DOLCI with left + right removed"],
                ["OLMo3-SFT-LRBalanced", "DOLCI with extra left rows removed"],
                ["OLMo3-SFT-NoLeft", "DOLCI with left removed"],
                ["OLMo3-SFT-NoRight", "DOLCI with right removed"],
            ],
        },
        "takeaway": "Training our own Full baseline is necessary because it controls for recipe differences from released checkpoints.",
    },
    {
        "section": "Training",
        "title": "Fine-Tuning Setup",
        "bullets": [
            "Base checkpoint: OLMo-3 base / pre-SFT checkpoint, depending on final availability and compute.",
            "Method: QLoRA / LoRA SFT using a fixed training recipe across all variants.",
            "All variants must use the same hyperparameters, sequence length, tokenizer, optimizer, and training budget.",
            "A full SFT baseline is required before interpreting any ablation effect.",
            "Existing smoke/baseline training scripts confirm that the LoRA SFT stack works; final OLMo-3 runs are the next stage.",
        ],
    },
    {
        "section": "Evaluation",
        "title": "Planned Evaluation",
        "bullets": [
            "Political positioning: ProbVAA-style prompts and issue-based ideology probes.",
            "Consistency: paraphrased versions of the same political questions.",
            "Refusal behavior: measure whether political questions become more likely to be refused.",
            "Utility preservation: instruction-following and general capability checks.",
            "Sanity checks: MMLU/GSM8K/TruthfulQA/IFEval or available lightweight equivalents.",
            "Compare every intervention against the Full SFT baseline.",
        ],
    },
    {
        "section": "Findings",
        "title": "Current Key Findings",
        "bullets": [
            "Manifesto labels are useful but noisy because RILE is document-level.",
            "Clean text-level ideology benchmarks are easier and more stable.",
            "ModernBERT is practical as a cheap binary political gate.",
            "Llama provides richer 4-way labels but still produces unparsed/noisy outputs.",
            "DOLCI political examples are ideologically left-heavy.",
            "Political data is 2.12% of rows but only about 0.33% of tokens.",
            "All main intervention datasets have been created and verified.",
        ],
    },
    {
        "section": "Limitations",
        "title": "Limitations",
        "bullets": [
            "Classifier labels are not perfect ground truth.",
            "Stage-1 false negatives cannot be recovered by the Llama 4-way classifier.",
            "Center-neutral political content is difficult to classify consistently.",
            "The political token mass is small, so intervention effects may be subtle.",
            "Left/right labels depend on prompt definitions and classifier behavior.",
            "Final conclusions require downstream model evaluation after SFT.",
        ],
    },
    {
        "section": "Next Steps",
        "title": "Immediate Next Steps",
        "bullets": [
            "Manually inspect 20–50 samples per label and document classifier quality.",
            "Compute exact token removal statistics for each intervention variant.",
            "Finalize OLMo-3 Full SFT baseline recipe.",
            "Train Full, NoPol, LRBalanced, NoIdeology, NoLeft, and NoRight variants.",
            "Freeze evaluation prompts before comparing model outputs.",
            "Report row-level and token-level intervention sizes in the thesis.",
        ],
    },
    {
        "section": "Close",
        "title": "One-Sentence Summary",
        "bullets": [
            "I built a measurement-to-intervention pipeline that identifies political SFT examples, quantifies ideological imbalance, creates controlled DOLCI/OLMo-3 training variants, and prepares the setup for testing whether SFT data composition changes political model behavior.",
        ],
    },
]


# ---------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------

def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def r_pr(size: int, bold: bool = False, color: str = "111111") -> str:
    b = "<a:b/>" if bold else ""
    return f'<a:rPr lang="en-US" sz="{size}">{b}<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'


def p_text(text: str, size: int = 2100, bold: bool = False, color: str = "111111") -> str:
    return f'<a:p><a:r>{r_pr(size, bold, color)}<a:t>{esc(text)}</a:t></a:r></a:p>'


def bullet_paragraph(text: str, size: int = 1900) -> str:
    return (
        f'<a:p><a:pPr marL="320000" indent="-210000">'
        f'<a:buChar char="•"/></a:pPr>'
        f'<a:r>{r_pr(size)}<a:t>{esc(text)}</a:t></a:r></a:p>'
    )


def textbox(idx: int, x: int, y: int, cx: int, cy: int, paragraphs: list[str]) -> str:
    body = "".join(paragraphs) or p_text("")
    return f"""
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="{idx}" name="TextBox {idx}"/>
          <p:cNvSpPr txBox="1"/>
          <p:nvPr/>
        </p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          <a:noFill/>
          <a:ln><a:noFill/></a:ln>
        </p:spPr>
        <p:txBody>
          <a:bodyPr wrap="square" anchor="t"/>
          <a:lstStyle/>
          {body}
        </p:txBody>
      </p:sp>
    """


def filled_rect(idx: int, x: int, y: int, cx: int, cy: int, color: str) -> str:
    return f"""
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="{idx}" name="Rect {idx}"/>
          <p:cNvSpPr/>
          <p:nvPr/>
        </p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
          <a:ln><a:noFill/></a:ln>
        </p:spPr>
      </p:sp>
    """


def section_tag(idx: int, section: str) -> str:
    return textbox(
        idx,
        10_850_000,
        250_000,
        1_900_000,
        300_000,
        [p_text(section.upper(), 1050, True, "666666")],
    )


def table_xml(idx_start: int, table: dict, x: int, y: int, cx: int, row_h: int = 430_000) -> tuple[str, int]:
    """Draw a real-looking table using rectangles and textboxes."""
    headers = table["headers"]
    rows = table["rows"]
    n_cols = len(headers)
    n_rows = len(rows) + 1

    col_w = cx // n_cols
    xml_parts = []
    idx = idx_start

    for r in range(n_rows):
        for c in range(n_cols):
            cell_x = x + c * col_w
            cell_y = y + r * row_h
            fill = "E9EEF7" if r == 0 else ("FFFFFF" if r % 2 == 1 else "F7F7F7")
            xml_parts.append(filled_rect(idx, cell_x, cell_y, col_w, row_h, fill))
            idx += 1

            text = headers[c] if r == 0 else rows[r - 1][c]
            size = 1250 if n_cols >= 5 else 1450
            if len(text) > 34:
                size -= 120
            xml_parts.append(
                textbox(
                    idx,
                    cell_x + 70_000,
                    cell_y + 70_000,
                    col_w - 120_000,
                    row_h - 80_000,
                    [p_text(text, size, r == 0, "111111")],
                )
            )
            idx += 1

    return "".join(xml_parts), idx


def title_block(slide: dict) -> list[str]:
    shapes = [
        textbox(
            2,
            550_000,
            230_000,
            10_100_000,
            650_000,
            [p_text(slide["title"], 3000, True, "111111")],
        ),
        section_tag(3, slide.get("section", "")),
        filled_rect(4, 550_000, 950_000, 12_200_000, 25_000, "D8DEE9"),
    ]

    return shapes


def slide_xml(slide: dict, num: int) -> str:
    shapes: list[str] = []
    shapes.extend(title_block(slide))
    next_id = 10

    if slide.get("subtitle"):
        shapes.append(
            textbox(
                next_id,
                650_000,
                1_200_000,
                11_800_000,
                1_000_000,
                [p_text(slide["subtitle"], 2050, False, "333333")],
            )
        )
        next_id += 1
        bullet_y = 2_350_000
    else:
        bullet_y = 1_250_000

    if slide.get("table"):
        table, next_id = table_xml(
            next_id,
            slide["table"],
            x=650_000,
            y=1_270_000,
            cx=12_050_000,
            row_h=430_000 if len(slide["table"]["rows"]) <= 6 else 385_000,
        )
        shapes.append(table)

        if slide.get("takeaway"):
            shapes.append(
                filled_rect(next_id, 650_000, 6_350_000, 12_050_000, 560_000, "F2F6FF")
            )
            next_id += 1
            shapes.append(
                textbox(
                    next_id,
                    850_000,
                    6_460_000,
                    11_650_000,
                    350_000,
                    [p_text("Takeaway: " + slide["takeaway"], 1450, True, "1F3A5F")],
                )
            )
            next_id += 1

    elif slide.get("bullets"):
        shapes.append(
            textbox(
                next_id,
                850_000,
                bullet_y,
                11_650_000,
                5_800_000,
                [bullet_paragraph(x, 1850) for x in slide["bullets"]],
            )
        )
        next_id += 1

    # Optional slide number
    shapes.append(
        textbox(
            next_id,
            12_250_000,
            7_050_000,
            600_000,
            220_000,
            [p_text(str(num), 950, False, "777777")],
        )
    )

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>
        <a:effectLst/>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


# ---------------------------------------------------------------------
# PPTX writer
# ---------------------------------------------------------------------

def write_pptx() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    slide_ids = "\n".join(
        f'<p:sldId id="{256 + i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1)
    )

    rels = "\n".join(
        f'<Relationship Id="rId{i}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
        f'Target="slides/slide{i}.xml"/>'
        for i in range(1, len(slides) + 1)
    )

    content_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(slides) + 1)
    )

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  {content_overrides}
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
""",
        )

        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
""",
        )

        z.writestr(
            "ppt/presentation.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldIdLst>
    {slide_ids}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
""",
        )

        z.writestr(
            "ppt/_rels/presentation.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {rels}
</Relationships>
""",
        )

        z.writestr(
            "docProps/core.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Political Positioning in Open Language Models</dc:title>
  <dc:creator>Bharat Puri</dc:creator>
</cp:coreProperties>
""",
        )

        z.writestr(
            "docProps/app.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python standard library</Application>
  <Slides>{len(slides)}</Slides>
</Properties>
""",
        )

        for i, slide in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(slide, i))
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
""",
            )

    print(OUT)


if __name__ == "__main__":
    write_pptx()
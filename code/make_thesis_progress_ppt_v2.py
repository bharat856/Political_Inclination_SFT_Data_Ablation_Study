from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

OUT = "thesis_progress_update_v2.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ---------------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------------

BG = RGBColor(248, 250, 252)
NAVY = RGBColor(30, 41, 59)
BLUE = RGBColor(37, 99, 235)
GREEN = RGBColor(22, 163, 74)
ORANGE = RGBColor(234, 88, 12)
GRAY = RGBColor(100, 116, 139)
LIGHT_GRAY = RGBColor(241, 245, 249)
WHITE = RGBColor(255, 255, 255)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def make_slide():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    return slide


def add_title(slide, text, subtitle=None):
    box = slide.shapes.add_textbox(
        Inches(0.55), Inches(0.3),
        Inches(12.2), Inches(0.65)
    )
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = NAVY

    if subtitle:
        box = slide.shapes.add_textbox(
            Inches(0.6), Inches(0.98),
            Inches(12), Inches(0.4)
        )
        p = box.text_frame.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(16)
        p.font.color.rgb = GRAY


def add_table(slide, x, y, w, h, headers, rows, widths=None,
              font_size=13, highlight_last=False):

    shape = slide.shapes.add_table(
        len(rows) + 1,
        len(headers),
        Inches(x), Inches(y),
        Inches(w), Inches(h)
    )

    table = shape.table

    if widths:
        for i, width in enumerate(widths):
            table.columns[i].width = Inches(width)

    # Header
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text = str(header)

        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(font_size)
        p.font.bold = True
        p.font.color.rgb = WHITE

    # Rows
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            cell = table.cell(i, j)

            cell.fill.solid()

            if highlight_last and i == len(rows):
                cell.fill.fore_color.rgb = RGBColor(220, 252, 231)
            else:
                cell.fill.fore_color.rgb = WHITE

            cell.text = str(value)

            tf = cell.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE

            p = tf.paragraphs[0]
            p.font.size = Pt(font_size)
            p.font.color.rgb = NAVY

            if highlight_last and i == len(rows):
                p.font.bold = True

            if j > 0:
                p.alignment = PP_ALIGN.RIGHT

            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)

    return table


def add_takeaway(slide, text, y=6.25, color=BLUE, size=18):
    box = slide.shapes.add_textbox(
        Inches(0.8), Inches(y),
        Inches(11.8), Inches(0.6)
    )

    tf = box.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = color


# =====================================================================
# SLIDE 1
# =====================================================================

s = make_slide()

add_title(
    s,
    "Thesis Progress Update",
    "Political Positioning in Open Language Models"
)

add_table(
    s,
    1.1, 1.65, 11.1, 3.9,
    ["Stage", "Status"],
    [
        ["Prepare SFT data", "Complete"],
        ["Identify political content", "Complete"],
        ["Validate political-content detection", "Complete"],
        ["Label political direction", "Complete"],
        ["Create intervention datasets", "Complete"],
        ["Train model variants", "In progress"],
        ["Evaluate trained models", "Next"],
    ],
    [7.5, 3.5],
    font_size=14
)

add_takeaway(
    s,
    "Current status: data preparation and classifier validation are complete. Model training is underway.",
    color=GREEN
)


# =====================================================================
# SLIDE 2
# =====================================================================

s = make_slide()

add_title(
    s,
    "Starting Training Data",
    "The study starts from the two OLMo-3 supervised fine-tuning datasets."
)

add_table(
    s,
    1.5, 1.8, 10.3, 2.6,
    ["Dataset", "Examples"],
    [
        ["DOLCI Instruct", "2,152,112"],
        ["DOLCI Think", "2,268,178"],
        ["Total", "4,420,290"],
    ],
    [6.5, 3.5],
    font_size=15
)

add_takeaway(
    s,
    "Total starting data: 4.42 million examples and approximately 24.1 billion tokens.",
    y=5.25
)


# =====================================================================
# SLIDE 3
# =====================================================================

s = make_slide()

add_title(
    s,
    "Finding Political Content",
    "The first task was to separate potentially political examples from the full training data."
)

add_table(
    s,
    1.2, 1.75, 11, 2.7,
    ["Category", "Examples", "Share of full data"],
    [
        ["Political candidates", "693,639", "15.7%"],
        ["Everything else", "3,726,651", "84.3%"],
        ["Total", "4,420,290", "100%"],
    ],
    [5.2, 2.8, 2.8],
    font_size=15
)

add_takeaway(
    s,
    "About 1 in 6 training examples was selected as potentially political.",
    y=5.25
)


# =====================================================================
# SLIDE 4
# =====================================================================

s = make_slide()

add_title(
    s,
    "Political-Content Detection: Validation Data",
    "We did not rely on only one validation dataset."
)

add_table(
    s,
    0.65, 1.45, 12.0, 3.6,
    [
        "Validation dataset",
        "Examples",
        "Largest class",
        "Baseline accuracy",
        "Baseline Macro-F1",
    ],
    [
        [
            "20 Newsgroups",
            "7,072",
            "Non-political",
            "85.5%",
            "46.1%",
        ],
        [
            "BBC News",
            "1,225",
            "Non-political",
            "80.2%",
            "44.5%",
        ],
        [
            "Manual target-domain set",
            "527",
            "Non-political",
            "62.0%",
            "38.3%",
        ],
    ],
    [3.4, 1.6, 2.5, 2.2, 2.2],
    font_size=12
)

add_takeaway(
    s,
    "Baseline = always predicting the most common class in that validation dataset.",
    y=5.55,
    color=ORANGE
)


# =====================================================================
# SLIDE 5
# =====================================================================

s = make_slide()

add_title(
    s,
    "Why the Baseline Matters",
    "Accuracy can look high even when a classifier is doing something very simple."
)

add_table(
    s,
    1.0, 1.55, 11.3, 3.6,
    [
        "Dataset",
        "Class distribution",
        "Always predict",
        "Accuracy",
    ],
    [
        [
            "20 Newsgroups",
            "6,045 non-pol. / 1,027 pol.",
            "Non-political",
            "85.5%",
        ],
        [
            "BBC News",
            "983 non-pol. / 242 pol.",
            "Non-political",
            "80.2%",
        ],
        [
            "Manual target-domain",
            "327 non-pol. / 200 pol.",
            "Non-political",
            "62.0%",
        ],
    ],
    [2.7, 4.0, 2.7, 1.8],
    font_size=12
)

add_takeaway(
    s,
    "This is why we compare against a baseline instead of looking at accuracy alone.",
    y=5.65
)


# =====================================================================
# SLIDE 6
# =====================================================================

s = make_slide()

add_title(
    s,
    "Political Direction: Validation Results",
    "A separate balanced set of 900 manually labelled examples was used for Left / Neutral / Right."
)

add_table(
    s,
    0.55, 1.35, 12.25, 4.5,
    [
        "Method",
        "Accuracy",
        "Macro-F1",
        "Left recall",
        "Neutral recall",
        "Right recall",
    ],
    [
        [
            "Single-class baseline",
            "33.3%",
            "16.7%",
            "0.0%",
            "100.0%",
            "0.0%",
        ],
        [
            "Qwen",
            "70.0%",
            "70.1%",
            "51.3%",
            "87.0%",
            "71.7%",
        ],
        [
            "Gemma",
            "69.1%",
            "69.1%",
            "47.7%",
            "89.0%",
            "70.7%",
        ],
        [
            "Majority ensemble",
            "71.6%",
            "71.8%",
            "55.3%",
            "86.7%",
            "72.7%",
        ],
        [
            "Tanise ensemble",
            "72.1%",
            "72.3%",
            "56.3%",
            "86.0%",
            "74.0%",
        ],
    ],
    [3.0, 1.55, 1.55, 1.8, 2.0, 1.8],
    font_size=11,
    highlight_last=True
)

add_takeaway(
    s,
    "The Tanise ensemble performed best and was used to produce the final political-direction labels.",
    y=6.15,
    color=GREEN,
    size=16
)


# =====================================================================
# SLIDE 7
# =====================================================================

s = make_slide()

add_title(
    s,
    "Final Labels on the Training Data",
    "The selected method was applied to all 693,639 political candidates."
)

add_table(
    s,
    0.8, 1.6, 6.0, 3.3,
    ["Final label", "Examples", "Share"],
    [
        ["Neutral", "662,969", "95.58%"],
        ["Left", "26,712", "3.85%"],
        ["Right", "3,958", "0.57%"],
        ["Total", "693,639", "100%"],
    ],
    [2.7, 1.8, 1.5],
    font_size=14
)

add_table(
    s,
    7.2, 1.6, 5.2, 2.6,
    ["Left / Right only", "Examples", "Share"],
    [
        ["Left", "26,712", "87.1%"],
        ["Right", "3,958", "12.9%"],
        ["Total", "30,670", "100%"],
    ],
    [2.3, 1.6, 1.3],
    font_size=13
)

add_takeaway(
    s,
    "Among explicitly directional examples, Left-labelled data is much more common than Right-labelled data.",
    y=5.55,
    color=ORANGE
)


# =====================================================================
# SLIDE 8
# =====================================================================

s = make_slide()

add_title(
    s,
    "Experimental Training Datasets",
    "The labels are used to create controlled versions of the same training dataset."
)

add_table(
    s,
    0.65, 1.25, 12.0, 4.9,
    [
        "Variant",
        "What changes",
        "Training examples",
    ],
    [
        ["Full", "Nothing removed", "4,420,290"],
        ["No Political", "Remove all political candidates", "3,726,651"],
        ["No Ideology", "Remove Left + Right", "4,389,620"],
        ["No Left", "Remove Left", "4,393,578"],
        ["No Right", "Remove Right", "4,416,332"],
        ["Balanced", "Remove extra Left examples", "4,397,536"],
    ],
    [2.4, 6.3, 3.0],
    font_size=13
)

add_takeaway(
    s,
    "The model stays the same; only the training data changes.",
    y=6.25
)


# =====================================================================
# SLIDE 9
# =====================================================================

s = make_slide()

add_title(
    s,
    "Where We Are Now",
    "The corrected Full model training run is underway."
)

add_table(
    s,
    0.8, 1.3, 6.0, 4.7,
    ["Training item", "Value"],
    [
        ["Training examples", "4,420,290"],
        ["GPUs", "4 × RTX A6000"],
        ["Examples per step", "64"],
        ["Planned steps", "69,065"],
        ["Observed speed", "~13–14 sec/step"],
        ["Estimated training time", "~11 days"],
    ],
    [3.8, 2.2],
    font_size=13
)

add_table(
    s,
    7.2, 1.3, 5.2, 3.2,
    ["Run", "Steps available"],
    [
        ["Previous setup", "~17,266"],
        ["Corrected setup", "69,065"],
    ],
    [3.1, 2.0],
    font_size=14
)

add_takeaway(
    s,
    "The previous setup split the data twice. This has been fixed, tested, and the full training run has restarted correctly.",
    y=5.55,
    color=GREEN,
    size=17
)


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

prs.save(OUT)

print()
print("=" * 60)
print("PowerPoint created successfully")
print(f"Saved as: {OUT}")
print(f"Slides: {len(prs.slides)}")
print("=" * 60)
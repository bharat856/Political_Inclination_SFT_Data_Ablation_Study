from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

OUT = "thesis_progress_update.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

BG = RGBColor(248, 250, 252)
NAVY = RGBColor(30, 41, 59)
BLUE = RGBColor(37, 99, 235)
GREEN = RGBColor(22, 163, 74)
GRAY = RGBColor(100, 116, 139)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(203, 213, 225)

def set_bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG

def title(slide, text, subtitle=None):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.65))
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(34)
    p.font.bold = True
    p.font.color.rgb = NAVY
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.65), Inches(1.03), Inches(12), Inches(0.35))
        p = sub.text_frame.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(16)
        p.font.color.rgb = GRAY

def add_table(slide, x, y, w, h, headers, rows, col_widths=None):
    table_shape = slide.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table

    if col_widths:
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Inches(cw)

    for j, head in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text = head
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE

    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE
            cell.text = str(val)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(13)
            p.font.color.rgb = NAVY
            if j > 0:
                p.alignment = PP_ALIGN.RIGHT
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(13)

    for row in table.rows:
        for cell in row.cells:
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.05)
            cell.margin_bottom = Inches(0.05)
    return table

def note(slide, text, x, y, w, h, color=BLUE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = color

def make_slide():
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    return slide

# 1
s = make_slide()
title(s, "Thesis Progress Update", "Political Positioning in Open Language Models")
add_table(s, 1.0, 1.7, 11.3, 3.8,
          ["Stage", "Status"],
          [
              ["Prepare SFT data", "Complete"],
              ["Identify political content", "Complete"],
              ["Label political direction", "Complete"],
              ["Create intervention datasets", "Complete"],
              ["Train model variants", "In progress"],
              ["Evaluate political positioning", "Next"],
          ],
          [7.5, 3.5])
note(s, "Current status: data preparation is complete; model training has started.", 1.0, 6.0, 11.3, 0.5, GREEN)

# 2
s = make_slide()
title(s, "Starting Dataset", "The full training data contains 4.42 million examples.")
add_table(s, 1.4, 1.8, 10.5, 2.4,
          ["Dataset", "Examples"],
          [
              ["DOLCI Instruct", "2,152,112"],
              ["DOLCI Think", "2,268,178"],
              ["Total", "4,420,290"],
          ],
          [6.5, 3.5])
note(s, "Original data size: approximately 24.1 billion tokens.", 1.4, 5.0, 10.5, 0.5)

# 3
s = make_slide()
title(s, "Political Content Found", "About 1 in 6 examples was selected as potentially political.")
add_table(s, 1.2, 1.7, 11.0, 2.8,
          ["Category", "Examples", "Share of full data"],
          [
              ["Political candidates", "693,639", "15.7%"],
              ["Everything else", "3,726,651", "84.3%"],
              ["Total", "4,420,290", "100%"],
          ],
          [5.2, 2.8, 2.8])
note(s, "Political candidates are the examples used for the left / neutral / right labelling step.", 1.2, 5.2, 11, 0.6)

# 4
s = make_slide()
title(s, "Final Political Labels", "Most political candidates are neutral.")
add_table(s, 0.8, 1.55, 6.0, 3.1,
          ["Final label", "Examples", "Share"],
          [
              ["Neutral", "662,969", "95.58%"],
              ["Left", "26,712", "3.85%"],
              ["Right", "3,958", "0.57%"],
              ["Total", "693,639", "100%"],
          ],
          [2.6, 1.9, 1.5])
add_table(s, 7.2, 1.55, 5.3, 2.4,
          ["Direction only", "Examples", "Share"],
          [
              ["Left", "26,712", "87.1%"],
              ["Right", "3,958", "12.9%"],
              ["Total", "30,670", "100%"],
          ],
          [2.3, 1.6, 1.3])
note(s, "Main finding: the explicitly directional data is strongly left-heavy.", 1.0, 5.4, 11.5, 0.6)

# 5
s = make_slide()
title(s, "Validation Results", "Results on the 900 manually labelled documents.")
add_table(s, 0.8, 1.4, 11.8, 4.2,
          ["Method", "Accuracy", "Macro-F1", "Left recall", "Neutral recall", "Right recall"],
          [
              ["Majority baseline", "33.3%", "16.7%", "0.0%", "100.0%", "0.0%"],
              ["Qwen", "70.0%", "70.1%", "51.3%", "87.0%", "71.7%"],
              ["Gemma", "69.1%", "69.1%", "47.7%", "89.0%", "70.7%"],
              ["Majority ensemble", "71.6%", "71.8%", "55.3%", "86.7%", "72.7%"],
              ["Tanise ensemble", "72.1%", "72.3%", "56.3%", "86.0%", "74.0%"],
          ],
          [3.0, 1.5, 1.5, 1.8, 1.9, 1.8])
note(s, "The Tanise ensemble performed best and was used for the final labels.", 0.9, 6.0, 11.5, 0.5, GREEN)

# 6
s = make_slide()
title(s, "Intervention Datasets", "Six controlled training conditions were created.")
add_table(s, 0.7, 1.25, 12.0, 4.8,
          ["Variant", "What is removed", "Training examples"],
          [
              ["Full", "Nothing", "4,420,290"],
              ["No Political", "All political candidates", "3,726,651"],
              ["No Ideology", "Left + Right", "4,389,620"],
              ["No Left", "Left", "4,393,578"],
              ["No Right", "Right", "4,416,332"],
              ["Balanced", "Extra Left examples", "4,397,536"],
          ],
          [2.5, 6.2, 3.0])

# 7
s = make_slide()
title(s, "What The Experiment Tests", "Same model setup, different training data.")
add_table(s, 0.9, 1.4, 11.5, 4.3,
          ["Comparison", "Main question"],
          [
              ["Full vs No Political", "Does political training data affect the model?"],
              ["Full vs No Ideology", "Does left/right political data matter?"],
              ["Full vs No Left", "What changes when left-labelled data is removed?"],
              ["Full vs No Right", "What changes when right-labelled data is removed?"],
              ["Full vs Balanced", "Does balancing left/right data change the model?"],
          ],
          [3.8, 7.5])
note(s, "Simple idea: change the data, retrain the model, measure what changes.", 1.0, 6.1, 11.2, 0.5)

# 8
s = make_slide()
title(s, "Current Training Status", "The corrected Full run is now underway.")
add_table(s, 0.9, 1.35, 11.4, 4.5,
          ["Item", "Current value"],
          [
              ["Full training examples", "4,420,290"],
              ["GPUs", "4 × RTX A6000"],
              ["Effective examples per step", "64"],
              ["Planned training steps", "69,065"],
              ["Observed speed", "~13–14 sec/step"],
              ["Estimated full training time", "~11 days"],
              ["Confirmed progress discussed", "67 / 69,065 steps"],
          ],
          [6.4, 4.6])
note(s, "Previous issue fixed: the data was being split twice; the corrected run now uses the intended full pass.", 0.9, 6.25, 11.4, 0.5, GREEN)

prs.save(OUT)
print(f"Saved: {OUT}")

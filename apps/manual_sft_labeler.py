from pathlib import Path
import pandas as pd
import streamlit as st

INPUT_PATH = Path("outputs/modernbert_ood_test/manual_review_ood_sample_300.csv")
OUTPUT_PATH = Path("outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv")
AUTOSAVE_PATH = OUTPUT_PATH
LABEL_OPTIONS = {
    "": "unlabeled",
    "0": "0 - non_political",
    "1": "1 - left",
    "2": "2 - center_neutral",
    "3": "3 - right",
    "U": "U - unclear / unsure",
}

st.set_page_config(
    page_title="SFT Political Manual Labeler",
    layout="wide",
)

st.title("SFT Political Manual Labeler")

@st.cache_data(show_spinner=False)
def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "manual_label" not in df.columns:
        df["manual_label"] = ""
    if "manual_notes" not in df.columns:
        df["manual_notes"] = ""

    df["manual_label"] = df["manual_label"].fillna("").astype(str)
    df["manual_notes"] = df["manual_notes"].fillna("").astype(str)

    return df

def save_df(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

if "df" not in st.session_state:
    if OUTPUT_PATH.exists():
        st.session_state.df = pd.read_csv(OUTPUT_PATH)
        if "manual_label" not in st.session_state.df.columns:
            st.session_state.df["manual_label"] = ""
        if "manual_notes" not in st.session_state.df.columns:
            st.session_state.df["manual_notes"] = ""
        st.session_state.df["manual_label"] = st.session_state.df["manual_label"].fillna("").astype(str)
        st.session_state.df["manual_notes"] = st.session_state.df["manual_notes"].fillna("").astype(str)
    else:
        st.session_state.df = load_input(str(INPUT_PATH))

if "idx" not in st.session_state:
    st.session_state.idx = 0

df = st.session_state.df
n = len(df)

# Normalize useful columns
if "sft_political_label_reparsed" in df.columns:
    pred_col = "sft_political_label_reparsed"
elif "sft_political_label" in df.columns:
    pred_col = "sft_political_label"
else:
    pred_col = None
raw_col = "sft_political_raw_output" if "sft_political_raw_output" in df.columns else None

# Sidebar
st.sidebar.header("Navigation")

labeled_mask = df["manual_label"].fillna("").astype(str).str.len() > 0
num_labeled = int(labeled_mask.sum())

st.sidebar.metric("Progress", f"{num_labeled} / {n}")

progress = num_labeled / n if n else 0
st.sidebar.progress(progress)

go_to = st.sidebar.number_input(
    "Go to row number",
    min_value=1,
    max_value=n,
    value=min(st.session_state.idx + 1, n),
    step=1,
)
if st.sidebar.button("Go"):
    st.session_state.idx = int(go_to) - 1
    st.rerun()

if st.sidebar.button("Next unlabeled"):
    unlabeled_indices = df.index[~labeled_mask].tolist()
    if unlabeled_indices:
        current = st.session_state.idx
        later = [i for i in unlabeled_indices if i > current]
        st.session_state.idx = later[0] if later else unlabeled_indices[0]
        st.rerun()

if st.sidebar.button("Save file"):
    save_df(df)
    st.sidebar.success(f"Saved to {OUTPUT_PATH}")

st.sidebar.divider()
st.sidebar.write("Label guide:")
st.sidebar.write("0 = non-political / task only")
st.sidebar.write("1 = left / progressive")
st.sidebar.write("2 = center / neutral political")
st.sidebar.write("3 = right / conservative")
st.sidebar.write("U = unclear")

# Current row
idx = st.session_state.idx
row = df.iloc[idx]

st.subheader(f"Example {idx + 1} of {n}")

top_cols = st.columns(4)
with top_cols[0]:
    st.metric("Manual labeled", "yes" if str(row.get("manual_label", "")) else "no")
with top_cols[1]:
    st.write("**Predicted label**")
    st.write(row.get(pred_col, "N/A") if pred_col else "N/A")
with top_cols[2]:
    st.write("**Source**")
    st.write(row.get("source", "N/A"))
with top_cols[3]:
    st.write("**Row ID**")
    st.write(row.get("row_id", row.get("id", "N/A")))

if raw_col:
    with st.expander("Raw model output", expanded=False):
        st.code(str(row.get(raw_col, "")))

left, right = st.columns([2, 1])

with left:
    st.markdown("### Text to review")

    def clean_value(x):
        if pd.isna(x):
            return ""
        x = str(x)
        if x.lower() in ["nan", "none", "null"]:
            return ""
        return x

    user_text = clean_value(row.get("user_text", ""))
    assistant_text = clean_value(row.get("assistant_text", ""))
    full_text = clean_value(row.get("full_text", ""))
    text_preview = clean_value(row.get("text_preview", ""))
    text = clean_value(row.get("text", ""))
    prompt = clean_value(row.get("prompt", ""))

    shown_any = False

    if user_text:
        shown_any = True
        st.markdown("#### USER")
        st.text_area(
            "User text",
            value=user_text,
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )

    if assistant_text:
        shown_any = True
        st.markdown("#### ASSISTANT")
        st.text_area(
            "Assistant text",
            value=assistant_text,
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )

    if full_text:
        shown_any = True
        st.markdown("#### FULL TEXT")
        st.text_area(
            "Full text",
            value=full_text,
            height=520,
            disabled=True,
            label_visibility="collapsed",
        )

    if text_preview:
        shown_any = True
        st.markdown("#### TEXT PREVIEW")
        st.text_area(
            "Text preview",
            value=text_preview,
            height=520,
            disabled=True,
            label_visibility="collapsed",
        )

    if text and not shown_any:
        shown_any = True
        st.markdown("#### TEXT")
        st.text_area(
            "Text",
            value=text,
            height=520,
            disabled=True,
            label_visibility="collapsed",
        )

    if prompt and not shown_any:
        shown_any = True
        st.markdown("#### PROMPT")
        st.text_area(
            "Prompt",
            value=prompt,
            height=520,
            disabled=True,
            label_visibility="collapsed",
        )

    if not shown_any:
        st.error("No text column found for this row.")
        st.write("Available columns:")
        st.write(list(df.columns))
        st.write("Current row values:")
        st.dataframe(row.to_frame(name="value"))

with right:
    st.markdown("### Your label")

    current_label = str(row.get("manual_label", ""))
    if current_label not in LABEL_OPTIONS:
        current_label = ""

    selected_label = st.radio(
        "Manual label",
        options=list(LABEL_OPTIONS.keys()),
        format_func=lambda x: LABEL_OPTIONS[x],
        index=list(LABEL_OPTIONS.keys()).index(current_label),
    )

    notes = st.text_area(
        "Notes",
        value=str(row.get("manual_notes", "")),
        height=150,
        placeholder="Optional: why this label?",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Save label", type="primary", use_container_width=True):
            df.at[df.index[idx], "manual_label"] = selected_label
            df.at[df.index[idx], "manual_notes"] = notes
            save_df(df)
            st.success("Saved")

    with col_b:
        if st.button("Save + next", use_container_width=True):
            df.at[df.index[idx], "manual_label"] = selected_label
            df.at[df.index[idx], "manual_notes"] = notes
            save_df(df)
            if idx < n - 1:
                st.session_state.idx += 1
            st.rerun()

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Previous", use_container_width=True):
            if idx > 0:
                st.session_state.idx -= 1
                st.rerun()

    with nav2:
        if st.button("Next", use_container_width=True):
            if idx < n - 1:
                st.session_state.idx += 1
                st.rerun()

st.divider()

st.markdown("### Current label counts")
counts = df["manual_label"].fillna("").replace("", "unlabeled").value_counts()
st.dataframe(counts.rename("count"))

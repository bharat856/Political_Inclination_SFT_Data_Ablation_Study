from pathlib import Path

import pandas as pd
import streamlit as st


INPUT_PATH = Path("outputs/modernbert_ood_test/modernbert_ood_predictions.csv")
OUTPUT_PATH = Path(
    "outputs/modernbert_ood_test/error_analysis/modernbert_ood_error_review_labeled.csv"
)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="ModernBERT OOD Error Inspector",
    layout="wide",
)

st.title("ModernBERT OOD Error Inspector")


LABEL_OPTIONS = {
    "": "not reviewed",
    "keep_original": "Keep original manual label",
    "change_0": "Change to 0 - non_political",
    "change_1": "Change to 1 - left",
    "change_2": "Change to 2 - center_neutral",
    "change_3": "Change to 3 - right",
    "change_U": "Change to U - unclear",
    "other": "Other / needs discussion",
}

DECISION_TO_FINAL_LABEL = {
    "change_0": "0",
    "change_1": "1",
    "change_2": "2",
    "change_3": "3",
    "change_U": "U",
}

ERROR_FILTERS = {
    "wrong": "All wrong predictions",
    "false_negative": "False negatives: political predicted non-political",
    "false_positive": "False positives: non-political predicted political",
    "center_neutral_false_negative": "Center-neutral false negatives only",
}


def normalize_manual_label(value):
    s = str(value).strip()

    if s.lower() in {"nan", "none", ""}:
        return ""

    if s.endswith(".0"):
        s = s[:-2]

    return s


def binary_label_name(value):
    try:
        value = int(value)
    except Exception:
        return "N/A"

    if value == 0:
        return "non_political"
    if value == 1:
        return "political"
    return "N/A"


@st.cache_data(show_spinner=False)
def load_data():
    if not INPUT_PATH.exists():
        st.error(f"Input file not found: {INPUT_PATH}")
        st.stop()

    df = pd.read_csv(INPUT_PATH)

    required_cols = [
        "manual_label",
        "gold_binary",
        "modernbert_pred_binary",
        "prob_political",
        "prob_non_political",
        "text_preview",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Input file is missing required columns: {missing}")
        st.write("Available columns:")
        st.write(df.columns.tolist())
        st.stop()

    if "row_id" not in df.columns:
        df["row_id"] = df.index.astype(str)

    df["manual_label_norm"] = df["manual_label"].apply(normalize_manual_label)

    df["error_type"] = "correct"

    df.loc[
        (df["gold_binary"] == 0) & (df["modernbert_pred_binary"] == 1),
        "error_type",
    ] = "false_positive"

    df.loc[
        (df["gold_binary"] == 1) & (df["modernbert_pred_binary"] == 0),
        "error_type",
    ] = "false_negative"

    df["error_type_detail"] = df["error_type"]

    df.loc[
        (df["manual_label_norm"] == "2") & (df["modernbert_pred_binary"] == 0),
        "error_type_detail",
    ] = "center_neutral_false_negative"

    df["review_decision"] = ""
    df["review_notes"] = ""
    df["final_review_label"] = ""

    if OUTPUT_PATH.exists():
        reviewed = pd.read_csv(OUTPUT_PATH)

        if "row_id" in reviewed.columns:
            review_cols = [
                "row_id",
                "review_decision",
                "review_notes",
                "final_review_label",
            ]
            review_cols = [c for c in review_cols if c in reviewed.columns]

            df = df.merge(
                reviewed[review_cols],
                on="row_id",
                how="left",
                suffixes=("", "_saved"),
            )

            for col in ["review_decision", "review_notes", "final_review_label"]:
                saved_col = f"{col}_saved"
                if saved_col in df.columns:
                    df[col] = df[saved_col].combine_first(df[col])
                    df = df.drop(columns=[saved_col])

    for col in ["review_decision", "review_notes", "final_review_label"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    return df


def save_reviews(df):
    keep_cols = [
        "row_id",
        "manual_label",
        "manual_label_norm",
        "final_review_label",
        "gold_binary",
        "modernbert_pred_binary",
        "prob_political",
        "prob_non_political",
        "error_type",
        "error_type_detail",
        "review_decision",
        "review_notes",
        "text_preview",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].copy()
    out.to_csv(OUTPUT_PATH, index=False)


def apply_review(df, original_index, decision, notes):
    df.at[original_index, "review_decision"] = decision
    df.at[original_index, "review_notes"] = notes

    if decision == "keep_original":
        df.at[original_index, "final_review_label"] = str(
            df.at[original_index, "manual_label_norm"]
        )
    elif decision in DECISION_TO_FINAL_LABEL:
        df.at[original_index, "final_review_label"] = DECISION_TO_FINAL_LABEL[decision]
    else:
        df.at[original_index, "final_review_label"] = ""


def get_filtered_df(df, filter_choice, sort_choice):
    filtered = df.copy()

    if filter_choice == "wrong":
        filtered = filtered[filtered["error_type"] != "correct"]
    elif filter_choice == "false_negative":
        filtered = filtered[filtered["error_type"] == "false_negative"]
    elif filter_choice == "false_positive":
        filtered = filtered[filtered["error_type"] == "false_positive"]
    elif filter_choice == "center_neutral_false_negative":
        filtered = filtered[
            filtered["error_type_detail"] == "center_neutral_false_negative"
        ]

    if sort_choice == "prob_political ascending":
        filtered = filtered.sort_values("prob_political", ascending=True)
    elif sort_choice == "prob_political descending":
        filtered = filtered.sort_values("prob_political", ascending=False)
    elif sort_choice == "row order":
        filtered = filtered.sort_index()

    return filtered.reset_index()


if "df" not in st.session_state:
    st.session_state.df = load_data()

if "pos" not in st.session_state:
    st.session_state.pos = 0

df = st.session_state.df


st.sidebar.header("Filters")

filter_choice = st.sidebar.selectbox(
    "Inspection set",
    options=list(ERROR_FILTERS.keys()),
    index=0,
    format_func=lambda x: ERROR_FILTERS[x],
)

sort_choice = st.sidebar.selectbox(
    "Sort by",
    options=[
        "prob_political ascending",
        "prob_political descending",
        "row order",
    ],
)

filtered = get_filtered_df(df, filter_choice, sort_choice)
n = len(filtered)

st.sidebar.metric("Rows in current filter", n)

reviewed_mask = df["review_decision"].fillna("").astype(str).str.len() > 0
st.sidebar.metric("Reviewed total", int(reviewed_mask.sum()))

if st.sidebar.button("Save all reviews"):
    save_reviews(df)
    st.sidebar.success(f"Saved to {OUTPUT_PATH}")

if st.sidebar.button("Reload from disk"):
    st.cache_data.clear()
    st.session_state.df = load_data()
    st.session_state.pos = 0
    st.rerun()

if n == 0:
    st.warning("No rows match this filter.")
    st.stop()

if st.session_state.pos >= n:
    st.session_state.pos = max(0, n - 1)

go_to = st.sidebar.number_input(
    "Go to item",
    min_value=1,
    max_value=n,
    value=min(st.session_state.pos + 1, n),
    step=1,
)

if st.sidebar.button("Go"):
    st.session_state.pos = int(go_to) - 1
    st.rerun()

current = filtered.iloc[st.session_state.pos]
original_index = int(current["index"])

st.subheader(f"Item {st.session_state.pos + 1} of {n}")

top = st.columns(7)

with top[0]:
    st.metric("Manual label", str(current.get("manual_label_norm", "")))

with top[1]:
    st.metric(
        "Gold binary",
        binary_label_name(current.get("gold_binary", "")),
    )

with top[2]:
    st.metric(
        "ModernBERT pred",
        binary_label_name(current.get("modernbert_pred_binary", "")),
    )

with top[3]:
    st.metric("Prob political", f"{float(current.get('prob_political', 0)):.4f}")

with top[4]:
    st.metric("Prob non-pol", f"{float(current.get('prob_non_political', 0)):.4f}")

with top[5]:
    st.metric("Error type", str(current.get("error_type", "")))

with top[6]:
    st.metric("Row ID", str(current.get("row_id", "")))

current_final_label = str(df.at[original_index, "final_review_label"])
if current_final_label:
    st.info(f"Current final review label for this row: {current_final_label}")

left, right = st.columns([2, 1])

with left:
    st.markdown("### Text")
    text = str(current.get("text_preview", ""))
    st.text_area(
        "Text preview",
        value=text,
        height=650,
        disabled=True,
        label_visibility="collapsed",
    )

with right:
    st.markdown("### Review")

    existing_decision = str(df.at[original_index, "review_decision"])
    if existing_decision not in LABEL_OPTIONS:
        existing_decision = ""

    decision = st.radio(
        "Decision",
        options=list(LABEL_OPTIONS.keys()),
        format_func=lambda x: LABEL_OPTIONS[x],
        index=list(LABEL_OPTIONS.keys()).index(existing_decision),
    )

    notes = st.text_area(
        "Review notes",
        value=str(df.at[original_index, "review_notes"]),
        height=180,
        placeholder=(
            "Example: Math task, but the scenario models voter support for "
            "a policy reform; keep as neutral political."
        ),
    )

    if st.button("Save review", type="primary", use_container_width=True):
        apply_review(df, original_index, decision, notes)
        save_reviews(df)
        st.success("Saved")

    if st.button("Save + next", use_container_width=True):
        apply_review(df, original_index, decision, notes)
        save_reviews(df)

        if st.session_state.pos < n - 1:
            st.session_state.pos += 1

        st.rerun()

    nav1, nav2 = st.columns(2)

    with nav1:
        if st.button("Previous", use_container_width=True):
            if st.session_state.pos > 0:
                st.session_state.pos -= 1
                st.rerun()

    with nav2:
        if st.button("Next", use_container_width=True):
            if st.session_state.pos < n - 1:
                st.session_state.pos += 1
                st.rerun()

st.divider()

st.markdown("### Current filter summary")

summary_cols = st.columns(4)

with summary_cols[0]:
    st.write("Error types")
    st.dataframe(filtered["error_type"].value_counts().rename("count"))

with summary_cols[1]:
    st.write("Manual labels")
    st.dataframe(filtered["manual_label_norm"].value_counts().rename("count"))

with summary_cols[2]:
    st.write("Review decisions")
    st.dataframe(
        df["review_decision"]
        .fillna("")
        .replace("", "not reviewed")
        .value_counts()
        .rename("count")
    )

with summary_cols[3]:
    st.write("Final review labels")
    st.dataframe(
        df["final_review_label"]
        .fillna("")
        .replace("", "not assigned")
        .value_counts()
        .rename("count")
    )

st.caption(f"Reviews are saved to: {OUTPUT_PATH}")
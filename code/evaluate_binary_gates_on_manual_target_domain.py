from pathlib import Path
import json
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

OUT_DIR = Path("outputs/validation_target_domain/manual_binary_gate_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_FILES = [
    Path("data/processed/tulu_olmo2_sft/target_domain_binary_manual_train.csv"),
    Path("data/processed/tulu_olmo2_sft/target_domain_binary_ood_corrected.csv"),
    Path("outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv"),
    Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200_labeled.csv"),
]

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
MODERNBERT_MODEL = Path("outputs/target_domain_binary_modernbert_combined/best_model")

BBC_PATH = Path("data/validation_external/bbc_news.csv")

POLITICAL_20NG = {
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
}

ALL_20NG = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
]


def detect_text_col(df):
    for c in ["text", "full_text", "content", "text_preview", "document", "input"]:
        if c in df.columns:
            return c

    str_cols = []
    for c in df.columns:
        if df[c].dtype == object:
            avg_len = df[c].fillna("").astype(str).str.len().mean()
            str_cols.append((avg_len, c))

    if not str_cols:
        raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")

    return sorted(str_cols, reverse=True)[0][1]


def normalize_binary_gold(x):
    """
    Manual political labels:
      0 = non-political
      1 = left/political
      2 = center-neutral/political
      3 = right/political
      U/uncertain = exclude
    """
    if pd.isna(x):
        return None

    s = str(x).strip().lower()

    if s in {"u", "uncertain", "unknown", "ambiguous", "nan", ""}:
        return None

    if s in {"0", "0.0", "false", "non_political", "non-political", "not_political", "apolitical", "no", "n"}:
        return 0

    if s in {"1", "1.0", "2", "2.0", "3", "3.0", "true", "political", "politics", "yes", "y",
             "left", "right", "center_neutral", "neutral", "political_neutral"}:
        return 1

    return None


def detect_gold_col(df):
    # Prefer human/final labels over original weak labels.
    priority = [
        "final_review_label",
        "manual_label",
        "manual_binary_label",
        "gold_binary",
        "binary_label",
        "label_binary",
        "label",
    ]

    best = None
    best_count = -1

    for c in priority:
        if c in df.columns:
            vals = df[c].apply(normalize_binary_gold)
            count = vals.notna().sum()
            if count > best_count:
                best = c
                best_count = count

    if best is None or best_count == 0:
        print("Could not auto-detect gold column.")
        print("Columns:", df.columns.tolist())
        for c in df.columns:
            print("\nColumn:", c)
            print(df[c].value_counts(dropna=False).head(20))
        raise ValueError("No usable gold label column found.")

    return best


def load_manual_files():
    dfs = []

    for p in MANUAL_FILES:
        if not p.exists():
            print("Skipping missing:", p)
            continue

        df = pd.read_csv(p)

        text_col = detect_text_col(df)
        gold_col = detect_gold_col(df)

        tmp = pd.DataFrame({
            "source_file": str(p),
            "text": df[text_col].fillna("").astype(str),
            "raw_gold": df[gold_col],
            "gold_binary": df[gold_col].apply(normalize_binary_gold),
        })

        before = len(tmp)
        tmp = tmp[tmp["gold_binary"].notna()].copy()
        tmp["gold_binary"] = tmp["gold_binary"].astype(int)
        tmp = tmp[tmp["text"].str.len() > 0].copy()

        print("\nLoaded:", p)
        print("Rows before filtering:", before)
        print("Rows after excluding U/unknown:", len(tmp))
        print("text_col:", text_col, "gold_col:", gold_col)
        print(tmp["gold_binary"].value_counts())

        dfs.append(tmp)

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["text", "gold_binary"])

    print("\nCombined manual target-domain rows:", len(out))
    print(out["gold_binary"].value_counts())

    return out


def load_external_train():
    rows = []

    print("Loading 20NG external training data...")
    ng = fetch_20newsgroups(
        subset="test",
        categories=ALL_20NG,
        remove=("headers", "footers", "quotes"),
    )

    for text, target in zip(ng.data, ng.target):
        cat = ng.target_names[target]
        y = 1 if cat in POLITICAL_20NG else 0
        rows.append({"text": text, "gold_binary": y})

    print("Loading BBC external training data...")
    bbc = pd.read_csv(BBC_PATH)

    for _, r in bbc.iterrows():
        label = str(r["label_text"]).strip().lower()
        y = 1 if label == "politics" else 0
        rows.append({"text": str(r["text"]), "gold_binary": y})

    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0].copy()

    print("External train rows:", len(df))
    print(df["gold_binary"].value_counts())

    return df


def eval_method(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    rep = classification_report(
        y,
        pred,
        labels=[0, 1],
        target_names=["non_political", "political"],
        output_dict=True,
        zero_division=0,
        digits=4,
    )

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": cm.tolist(),
        "political_precision": float(rep["political"]["precision"]),
        "political_recall": float(rep["political"]["recall"]),
        "political_f1": float(rep["political"]["f1-score"]),
        "non_political_precision": float(rep["non_political"]["precision"]),
        "non_political_recall": float(rep["non_political"]["recall"]),
        "non_political_f1": float(rep["non_political"]["f1-score"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
    }


def predict_modernbert(texts):
    if not MODERNBERT_MODEL.exists():
        print("ModernBERT model missing:", MODERNBERT_MODEL)
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading ModernBERT:", MODERNBERT_MODEL, "device:", device)

    tok = AutoTokenizer.from_pretrained(MODERNBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODERNBERT_MODEL).to(device)
    model.eval()

    probs = []
    bs = 16

    for i in tqdm(range(0, len(texts), bs), desc="ModernBERT"):
        batch = texts[i:i + bs]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        probs.extend(p.tolist())

    probs = np.array(probs)
    pred = (probs >= 0.5).astype(int)

    return probs, pred


def main():
    manual = load_manual_files()
    y = manual["gold_binary"].to_numpy()
    texts = manual["text"].tolist()

    train_df = load_external_train()

    print("Loading embedding model:", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Encoding external training data...")
    X_train = embedder.encode(
        train_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    y_train = train_df["gold_binary"].astype(int).to_numpy()

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)

    print("Encoding manual target-domain data...")
    X_manual = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    emb_prob = clf.predict_proba(X_manual)[:, 1]

    mb_prob, mb_pred = predict_modernbert(texts)

    df = manual.copy()
    df["embedding_prob_political"] = emb_prob

    methods = {}

    for th in [0.2, 0.3, 0.4, 0.5]:
        name = f"embedding_t{th}"
        pred = (emb_prob >= th).astype(int)
        df[name] = pred
        methods[name] = pred

    if mb_pred is not None:
        df["modernbert_prob_political"] = mb_prob
        df["modernbert"] = mb_pred
        methods["modernbert"] = mb_pred

        for th in [0.3, 0.4, 0.5]:
            emb_pred = methods[f"embedding_t{th}"]
            union = ((emb_pred == 1) | (mb_pred == 1)).astype(int)
            name = f"union_t{th}"
            df[name] = union
            methods[name] = union

    summary = {
        "rows": int(len(df)),
        "gold_distribution": df["gold_binary"].value_counts().astype(int).to_dict(),
        "source_file_counts": df["source_file"].value_counts().astype(int).to_dict(),
        "methods": {},
    }

    print("\n" + "=" * 100)
    print("Manual target-domain binary gate validation")
    print("=" * 100)

    for name, pred in methods.items():
        res = eval_method(y, pred)
        summary["methods"][name] = res

        print("\nMethod:", name)
        print("Accuracy:", res["accuracy"])
        print("Political precision:", res["political_precision"])
        print("Political recall:", res["political_recall"])
        print("Political F1:", res["political_f1"])
        print("Confusion matrix:", res["confusion_matrix"])

    out_csv = OUT_DIR / "manual_target_domain_binary_gate_predictions.csv"
    out_json = OUT_DIR / "manual_target_domain_binary_gate_summary.json"

    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2))

    print("\nSaved:", out_csv)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()

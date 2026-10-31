from pathlib import Path
import json

out = {
    "modernbert_binary_gate": {
        "20newsgroups": {
            "rows": 7072,
            "accuracy": 0.8433257918552036,
            "political_precision": 0.4759,
            "political_recall": 0.7780,
            "political_f1": 0.5905,
            "non_political_recall": 0.8544,
            "confusion_matrix_labels_0_1": [[5165, 880], [228, 799]],
            "summary_file": "outputs/validation_external/20newsgroups_modernbert_binary/20newsgroups_modernbert_summary.json",
        },
        "bbc_news": {
            "rows": 1225,
            "accuracy": 0.8587755102040816,
            "political_precision": 0.6194,
            "political_recall": 0.7397,
            "political_f1": 0.6742,
            "non_political_recall": 0.8881,
            "confusion_matrix_labels_0_1": [[873, 110], [63, 179]],
            "summary_file": "outputs/validation_external/bbc_news_modernbert_binary/bbc_news_modernbert_summary.json",
        },
    },
    "llama_4way_slant_classifier": {
        "tanise_slant_testset": {
            "rows": 900,
            "strict_accuracy_counting_non_political_wrong": 0.6633333333333333,
            "target_label_only_rows": 854,
            "target_label_only_accuracy": 0.6990632318501171,
            "prediction_distribution": {
                "left": 355,
                "center_neutral": 262,
                "right": 237,
                "non_political": 46,
            },
            "gold_distribution": {
                "left": 300,
                "center_neutral": 300,
                "right": 300,
            },
            "strict_confusion_matrix_labels": ["left", "center_neutral", "right", "non_political"],
            "strict_confusion_matrix": [
                [229, 55, 8, 8],
                [84, 160, 21, 35],
                [42, 47, 208, 3],
                [0, 0, 0, 0],
            ],
            "summary_file": "outputs/validation_tanise/slant_testset_llama_4way/slant_testset_llama_4way_eval_summary.json",
        }
    },
}

out_path = Path("outputs/classifier_validation_summary.json")
out_path.write_text(json.dumps(out, indent=2))
print("Saved:", out_path)
print(json.dumps(out, indent=2))

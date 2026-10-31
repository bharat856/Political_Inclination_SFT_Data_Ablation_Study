import matplotlib.pyplot as plt

# Grouped conversation lengths
message_groups = ["2", "3–4", "5–10", "11–20", "21–30", "31+"]
examples = [1922012, 32570, 108860, 22137, 64450, 2083]

total_examples = 2152112

# Convert counts to percentages
percentages = [(count / total_examples) * 100 for count in examples]

# Create figure
plt.figure(figsize=(9, 5))

bars = plt.bar(message_groups, percentages)

# Add percentage labels above bars
for bar, percentage in zip(bars, percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{percentage:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlabel("Number of Messages per Example")
plt.ylabel("Percentage of Examples (%)")
plt.title("Distribution of Conversation Lengths in DOLCI Instruct")

plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

# Save high-resolution versions for the thesis
plt.savefig("dolci_message_distribution.pdf", bbox_inches="tight")
plt.savefig("dolci_message_distribution.png", dpi=300, bbox_inches="tight")

plt.show()
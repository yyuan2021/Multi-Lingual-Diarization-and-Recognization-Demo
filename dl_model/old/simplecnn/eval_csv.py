import argparse
import csv
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    total = 0
    correct = 0

    # 每个类别统计
    class_total = defaultdict(int)
    class_correct = defaultdict(int)

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            true = int(row["true_label"])
            pred = int(row["pred_label"])

            total += 1
            class_total[true] += 1

            if true == pred:
                correct += 1
                class_correct[true] += 1

    # ===== overall =====
    overall_acc = correct / total if total > 0 else 0

    print("===== Overall =====")
    print(f"Accuracy: {overall_acc:.4f} ({correct}/{total})")

    # ===== per class =====
    print("\n===== Per-class Accuracy =====")

    label_map = {
        0: "mix",
        1: "code_switch"
    }

    for cls in sorted(class_total.keys()):
        c_total = class_total[cls]
        c_correct = class_correct[cls]
        acc = c_correct / c_total if c_total > 0 else 0

        name = label_map.get(cls, str(cls))

        print(f"{name:12s} | acc={acc:.4f} ({c_correct}/{c_total})")


if __name__ == "__main__":
    main()
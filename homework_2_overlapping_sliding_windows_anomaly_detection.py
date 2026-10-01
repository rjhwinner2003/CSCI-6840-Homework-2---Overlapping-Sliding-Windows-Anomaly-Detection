"""
Ryan Hall
East Carolina University
HW2 - Overlapping Sliding Windows Anomaly Detection
Dr. Hooman Hedayati
30 September 2026
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WINDOW_SIZE = 500
Q_PERCENTILE = 91.0

DATA_FILE = Path("AG_NO3_fill_cells_remove_NAN.csv")
PLOT_FILE = Path("anomaly_detection_plot.png")


def sliding_window_predict(x, window_size, q):
    n = len(x)

    if window_size <= 0 or window_size > n:
        raise ValueError("window_size has to be between 1 and the series length.")

    predictions = np.zeros(n, dtype=np.int8)

    # First window: compute one threshold and label all points.
    first_threshold = np.percentile(
        x[0:window_size],
        q,
        method="linear"
    )

    predictions[0:window_size] = (
        x[0:window_size] >= first_threshold
    ).astype(np.int8)

    # Later windows: recompute the threshold and label only the new point.
    for t in range(window_size, n):
        start = t - window_size + 1
        current_window = x[start:t + 1]

        threshold = np.percentile(
            current_window,
            q,
            method="linear"
        )

        predictions[t] = int(x[t] >= threshold)

    return predictions


def calculate_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    p = tp + fn
    n = tn + fp

    normal_accuracy = tn / n
    anomaly_accuracy = tp / p

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Normal Accuracy": normal_accuracy,
        "Anomaly Accuracy": anomaly_accuracy,
    }


def make_plot(df, y_pred):
    dates = pd.to_datetime(df["Date"])
    predicted = y_pred == 1
    actual = df["Student_Flag"].to_numpy(dtype=int) == 1

    plt.figure(figsize=(8, 8))

    plt.plot(
        dates,
        df["NO3N"],
        linewidth=0.8,
        label="NO3N"
    )

    plt.scatter(
        dates[predicted],
        df.loc[predicted, "NO3N"],
        s=26,
        marker="o",
        label="Predicted anomaly"
    )

    plt.scatter(
        dates[actual],
        df.loc[actual, "NO3N"],
        s=40,
        marker="x",
        label="Ground-truth anomaly"
    )

    plt.xlabel("Date")
    plt.ylabel("Nitrate (NO3N)")
    plt.title(
        f"Sliding-Window Nitrate Anomaly Detection "
        f"(W={WINDOW_SIZE}, q={Q_PERCENTILE:g})"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()


def main():
    df = pd.read_csv(DATA_FILE)

    required = {"Date", "NO3N", "Student_Flag"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    if df["NO3N"].isna().any():
        raise ValueError(
            "NO3N contains NaN values. The assignment specifies the cleaned file."
        )

    x = df["NO3N"].to_numpy(dtype=float)
    y_true = df["Student_Flag"].to_numpy(dtype=int)

    y_pred = sliding_window_predict(
        x,
        WINDOW_SIZE,
        Q_PERCENTILE
    )

    metrics = calculate_metrics(y_true, y_pred)

    make_plot(df, y_pred)

    print(f"Window size (W): {WINDOW_SIZE}")
    print(f"Percentile (q): {Q_PERCENTILE:g}")
    print("Rule: upper-tail anomaly when NO3N >= current-window threshold")
    print()

    print(f"TP: {metrics['TP']}")
    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")
    print(f"TN: {metrics['TN']}")
    print(f"Normal accuracy:  {metrics['Normal Accuracy']:.4%}")
    print(f"Anomaly accuracy: {metrics['Anomaly Accuracy']:.4%}")
    print()

    print(f"Saved figure to: {PLOT_FILE}")


if __name__ == "__main__":
    main()

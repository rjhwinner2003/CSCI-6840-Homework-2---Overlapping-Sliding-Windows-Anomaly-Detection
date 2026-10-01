# HW2 - Overlapping Sliding Windows Anomaly Detection

## Overview

This project detects anomalies in a nitrate time series using a threshold-based
method with fixed-size, overlapping sliding windows.

The dataset that was used is:

`AG_NO3_fill_cells_remove_NAN.csv`

This dataset was given along with the assignment.

From this dataset, one can see that the nitrate values are stored in the `NO3N` column, and the ground truth anomaly labels are stored in the `Student_Flag` column.

The goal for this program is to classify nitrate observations as either normal or anomalous
using an adaptive percentile threshold.

## Method

A fixed-size sliding window with a step size of 1 was used.

The parameters used were:

- Window size: `W = 500`
- Percentile: `q = 91`
- Step size: `1`
- Threshold type: one-sided upper-tail

For the first window, which contains the first 500 observations, the threshold was calculated using the following code:

```python
np.percentile(x[0:500], 91, method="linear")

```

All 500 observations in the first window were classified using this threshold.

For each window after the first one, the window moves forward by one data point. The 91st-percentile threshold is recalculated using only the data inside the current window. Only the newly added point is then classified.

A point is classified as an anomaly when its nitrate value is greater than or equal to the current-window threshold.

## Choice of W and q

I chose a window size of `W = 500` because it provides enough observations to calculate a stable local percentile threshold while still allowing the threshold to adjust as the nitrate data changes over time.

I chose `q = 91` because this percentile produced results that met both of the required accuracy targets. A higher percentile would make the detector more selective, but it could also cause more actual anomalies to be missed. A lower percentile would classify more points as anomalies and increase the number of false positives.

## Results

The dataset contains 141 ground-truth anomalies.

The results from the sliding-window anomaly detector were:

- True Positives (TP): 106
- False Positives (FP): 4363
- False Negatives (FN): 35
- True Negatives (TN): 26286

The total number of actual anomalies was:

`TP + FN = 106 + 35 = 141`

The total number of normal events was:

`TN + FP = 26286 + 4363 = 30649`

### Normal Event Detection Accuracy

`TN / N = 26286 / 30649 = 85.76%`

**Normal Event Detection Accuracy: 85.76%**

### Anomaly Event Detection Accuracy

`TP / P = 106 / 141 = 75.18%`

**Anomaly Event Detection Accuracy: 75.18%**

Both required accuracy targets were met:

- Normal accuracy >= 80%
- Anomaly accuracy >= 75%

## Anomaly Detection Plot

![Anomaly Detection Plot](anomaly_detection_plot.png)

The line represents the nitrate values over time. The circular markers represent the points predicted as anomalies by the sliding-window method. The X markers represent the ground-truth anomalies from the `Student_Flag` column.

## Design Choices

A one-sided upper-tail threshold was used because the method is intended to identify unusually high nitrate values.

The provided dataset was already cleaned. The program checks the `NO3N` column for missing values before running the anomaly detection algorithm.

For the first window, all 500 observations were classified using the first threshold. For every window after the first one, the threshold was recalculated and only the newly added point was classified.

The program also creates `sliding_window_predictions.csv`, which contains the calculated threshold and predicted anomaly label for each observation.

## Files

The main files for this assignment are:

- `homework_2_overlapping_sliding_windows_anomaly_detection.py`
- `README.md`
- `anomaly_detection_plot.png`

The Python script loads the dataset, performs the sliding-window anomaly detection, calculates the evaluation metrics, creates the anomaly detection figure, and saves the prediction results.

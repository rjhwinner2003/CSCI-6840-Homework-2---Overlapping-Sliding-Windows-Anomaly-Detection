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

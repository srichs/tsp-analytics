# Troubleshooting

## CSV data is missing

`TspAnalytics` will attempt to download the CSV if it is missing. If your network
blocks the request, retry later or provide a pre-downloaded CSV in the
`data_dir` you pass to `TspAnalytics`.

## Matplotlib display issues

In headless environments, configure a non-interactive backend:

```python
import matplotlib
matplotlib.use("Agg")
```

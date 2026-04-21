# Results

The earlier single-example result note in this repository has been superseded.

Use these files instead:

- [TECHNICAL_REPORT.md](/Users/sirishag/Desktop/runara-finqa/TECHNICAL_REPORT.md)
- [PRESENTATION.md](/Users/sirishag/Desktop/runara-finqa/PRESENTATION.md)
- [quick_eval_and_report.py](/Users/sirishag/Desktop/runara-finqa/quick_eval_and_report.py)

To generate benchmark results in the target GPU environment:

```bash
python3 quick_eval_and_report.py --split validation --num-examples 100
```

This will produce:

- `data/eval_results_validation_100.json`
- `data/eval_summary_validation_100.md`

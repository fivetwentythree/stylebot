# Tinker runs

Runtime files stay local. The prepared first run uses this layout.

```text
white-v1/
  baselines/   Saved base-model outputs, captured before training
  tinker/      Cookbook config.json, metrics.jsonl and checkpoints.jsonl
  outputs/     Trained-model outputs for the same evaluation inputs
  review/      Blinded comparison, ratings and a separate answer key
  run.json     Trained checkpoint, exact prompt and experiment identity
```

The Cookbook creates metrics and checkpoint records during real training. Preparation does not create pretend training results. Remote checkpoint paths refer to weights stored by Tinker.

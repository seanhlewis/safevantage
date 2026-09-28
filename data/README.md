# Benchmark data

Download benchmark datasets and simulator assets from their official sources and follow the listed access terms.

- **ProcTHOR-10K:** [Dataset](https://github.com/allenai/procthor-10k) · [Generation code](https://github.com/allenai/procthor) · [AI2-THOR simulator](https://github.com/allenai/ai2thor). The SafeVantage evaluator uses prepared scene records; see [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) for their layout.
- **HM3D:** [Official dataset page](https://aihabitat.org/datasets/hm3d/). Access is subject to the [Matterport academic and non-commercial terms](https://matterport.com/legal/matterport-terms-and-conditions-academic-and-non-commercial-use-spaces).
- **ScanNet:** Used for the four-state negative-closure audit. Request data through the [official ScanNet page](https://www.scan-net.org/ScanNet/) and follow its [Terms of Use](https://www.scan-net.org/ScanNet/). The evaluator accepts prepared four-state episode JSONL via `--episodes`; its command is in [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md).

The repository ignores `data/local_inputs/` so obtained assets stay out of version control. The full ProcTHOR preparation steps are in [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md). To use a prepared scene tree from another location, stage it with:

```bash
python scripts/prepare_local_inputs.py \
  --dataset procthor \
  --source /path/to/prepared/scenes \
  --destination data/local_inputs/procthor_external
```

Add `--mode copy` to copy the input instead of linking it.

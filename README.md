# SEAV: Sequential Epistemic and Action-Level Validation

Reproducibility code.

## Setup

```bash
pip install -r requirements.txt
```

## Environment Variables

```bash
export GEMINI_API_KEY="..."        # For Gemini models (default backend)
export OPENROUTER_API_KEY="..."    # For OpenRouter models (alternative)
export OPENAI_API_KEY="..."        # For OpenAI-compatible baselines
export TAVILY_API_KEY="..."        # For Tavily web search
```

## Quick Demo

```bash
python demo/run_demo.py
```

## Running SEAV

```bash
python experiments/run_methods.py \
    --method seal \
    --dataset jailbreakqr \
    --model gemini-3-flash-preview \
    --jailbreakqr-path <path-to-dataset.json>
```

See `experiments/run_methods.py --help` for all options, methods, and datasets.

## Datasets

- **OrdSense**: Included in `datasets/ordsense/`. Construction details in paper Appendix B.2.
- **SD, JQR-Binary**: Not included due to licensing restrictions on source data. See paper for construction details.
- **JBB**: Publicly available via `pip install datasets` and `load_dataset("JailbreakBench/JBB-Behaviors")`.
- **GPTFuzz, WildGuardMix, UltraSafety**: Publicly available; see paper references for download links.

## Package Structure

```
seav/           SEAV pipeline (4 nodes: extraction, verification, ordering, judgment)
baselines/      Baseline evaluation methods
jades/          JADES baseline
experiments/    Experiment runner and metrics computation
demo/           Quick demo script
datasets/       OrdSense ordering-sensitivity evaluation dataset
```

## License

MIT License

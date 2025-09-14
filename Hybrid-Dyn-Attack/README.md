Hybrid-Dyn-Attack (modularized)

This folder contains a modular reorganization of the dynamic Hybrid attack from:
`scripts/Hybrid_dyn_N_LLM.py`.

Key addition: configurable N selection for N-ary search
- Auto: pick N dynamically via `get_n_gpt_neo(len, dataset)` per example.
- Manual: set N explicitly from the CLI.

How to run (example)
- Auto N per example:
  python Hybrid-Dyn-Attack/cli.py --Just_Hybrid --datasets imdb --models jialicheng/deberta-base-imdb --slice 0:1000 --classifier_name deberta --n_mode auto

- Manual N:
  python Hybrid-Dyn-Attack/cli.py --Just_Hybrid --datasets imdb --models jialicheng/deberta-base-imdb --slice 0:1000 --classifier_name deberta --n_mode manual --ns 2 3 6

Structure
- Hybrid-Dyn-Attack/attack: NSNode, N-ary selection, greedy WNR, Hybrid orchestrator
- Hybrid-Dyn-Attack/replace: WordNet synonym replacement
- Hybrid-Dyn-Attack/metrics: perturbation rate, token diff, semantic similarity (SBERT)
- Hybrid-Dyn-Attack/utils: prediction outputs, dynamic N selector, misc
- Hybrid-Dyn-Attack/experiments: batch runner and TSV logger
- Hybrid-Dyn-Attack/cli.py: argument parsing and entrypoint

Note: The original script is left unchanged.

Hybrid-Sentence-Attack-LLM (modularized)


Structure:
- `attack/`: N-ary search, greedy WNR, hybrid orchestration
- `replace/`: WordNet synonym replacement
- `metrics/`: perturbation rate, token diff, semantic similarity (SBERT)
- `utils/`: small helpers (prediction outputs, token index lookup)
- `experiments/`: batch runner that logs TSV results
- `cli.py`: argument parsing and main entry similar to the original script

Usage (equivalent to the original):
  python -m Hybrid_Sentence_Attack_LLM.cli --HybridSent --datasets imdb --models jialicheng/deberta-base-imdb --ns 2 3 6 --ks -1 --slice 0:1000 --classifier_name deberta

Note: The original script is NOT changed and still works as before.


# Hybrid-Dynamic-Select

Modular implementations of score-based adversarial NLP attacks and a Raw-Scripts area for single-file, ad‑hoc experimentation.

This repo now provides clean, reusable packages for different attack variants, while keeping the original one-file scripts (moved under Raw-Scripts) for convenience.

Project Structure
- DynNnary: N-ary WNR and Greedy WNR (auto/manual N)
- Hybrid-Dyn-Attack: Hybrid token-level attack with optional automatic N selection
- Hybrid_Sentence_Attack_LLM: Hybrid sentence-level attack (modularized)
- Raw-Scripts: Formerly scripts; de‑modularized single‑file versions for quick, ad‑hoc experiments

Folder Tree (top-level)
- DynNnary/
  - attack/ (ns_node, n_ary, nnary, greedy)
  - replace/ (wordnet_replace, bert_replace)
  - metrics/ (perturbation, similarity)
  - utils/ (prediction, n_selector)
  - experiments/ (runner)
  - cli.py
- Hybrid-Dyn-Attack/
  - attack/ (ns_node, n_ary, hybrid, greedy)
  - replace/ (wordnet_replace)
  - metrics/ (perturbation, similarity)
  - utils/ (prediction, n_selector)
  - experiments/ (runner)
  - cli.py
- Hybrid_Sentence_Attack_LLM/
  - attack/ (ns_node, n_ary, hybrid)
  - replace/ (wordnet_replace)
  - metrics/ (perturbation, similarity)
  - utils/ (prediction, text)
  - experiments/ (runner)
  - models/ (classifier wrapper)
  - cli.py
- Raw-Scripts/ (de‑modularized originals for ad‑hoc use)

Usage (CLIs)
- DynNnary (N-ary / Greedy WNR):
  - Auto N per example: python DynNnary/cli.py --datasets imdb --models jialicheng/deberta-base-imdb --n_mode auto --ks -1 --classifier_name deberta
  - Manual N values:   python DynNnary/cli.py --datasets imdb --models jialicheng/deberta-base-imdb --n_mode manual --ns 2 3 6 --ks -1 --classifier_name deberta
  - Greedy:            python DynNnary/cli.py --datasets imdb --models jialicheng/deberta-base-imdb --greedy --ks -1 --classifier_name deberta

- Hybrid-Dyn-Attack (token-level Hybrid):
  - Auto N per example: python Hybrid-Dyn-Attack/cli.py --Just_Hybrid --datasets imdb --models jialicheng/deberta-base-imdb --n_mode auto --ks -1 --classifier_name deberta
  - Manual N values:   python Hybrid-Dyn-Attack/cli.py --Just_Hybrid --datasets imdb --models jialicheng/deberta-base-imdb --n_mode manual --ns 2 3 6 --ks -1 --classifier_name deberta

- Hybrid_Sentence_Attack_LLM (sentence-level Hybrid):
  - python Hybrid_Sentence_Attack_LLM/cli.py --HybridSent --datasets imdb --models jialicheng/deberta-base-imdb --ns 2 3 6 --ks -1 --classifier_name deberta

Raw-Scripts
- Single-file versions of the above (formerly in scripts/). These are convenient for quick, ad‑hoc experiments when modular structure isn’t necessary.

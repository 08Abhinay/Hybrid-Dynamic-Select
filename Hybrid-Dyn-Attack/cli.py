from __future__ import annotations

import argparse

from transformers import pipeline

from experiments.runner import run_attack_experiments


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-run dynamic Hybrid attacks with auto/manual N.",
    )
    p.add_argument(
        "--output_base",
        type=str,
        default="/scratch/gilbreth/abelde/NLP_Score_Based_Attacks/Outputs_test",
        help="root directory for all experiment outputs",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["imdb"],
        help="🤗 dataset IDs (e.g. imdb, yelp_polarity, ag_news)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["jialicheng/deberta-base-imdb"],
        help="Hugging Face model IDs, same order as --datasets",
    )
    p.add_argument(
        "--ns",
        nargs="+",
        type=int,
        default=[2, 3, 6],
        help="values of N for manual mode (one run per N)",
    )
    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[-1],
        help="values of K (synonym candidates)",
    )
    p.add_argument(
        "--Just_Hybrid",
        action="store_true",
        help="Run experiments for Just_Hybrid",
    )
    p.add_argument(
        "--replace",
        type=str,
        default="wordnet",
        help="synonym source (reserved)",
    )
    p.add_argument(
        "--slice",
        type=str,
        default="0:1000",
        help="dataset slice start:end (useful in SLURM arrays)",
    )
    p.add_argument(
        "--classifier_name",
        type=str,
        default="deberta",
        help="Name of the classifier for the output path",
    )
    p.add_argument(
        "--n_mode",
        type=str,
        choices=["auto", "manual"],
        default="auto",
        help="Select N automatically per example or set manually",
    )
    return p.parse_args()


def main() -> None:
    args = parse_arguments()
    start_idx, end_idx = map(int, args.slice.split(":"))

    ds_name = args.datasets[0]
    model_id = args.models[0]
    print("Using model:", model_id)

    classifier = pipeline(
        "text-classification",
        model=model_id,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_all_scores=True,
        device=0,
        framework="pt",
    )

    # One-line monkey-patch to ensure long() on inputs for some models
    orig_preprocess = classifier.preprocess

    def _ensure_long_ids(inputs, **kwargs):
        model_inputs = orig_preprocess(inputs, **kwargs)
        if "input_ids" in model_inputs:
            model_inputs["input_ids"] = model_inputs["input_ids"].long()
        if "token_type_ids" in model_inputs:
            model_inputs["token_type_ids"] = model_inputs["token_type_ids"].long()
        return model_inputs

    classifier.preprocess = _ensure_long_ids

    classifier_name = args.classifier_name
    print(f"Entering attack loop for {ds_name}")

    if args.n_mode == "auto":
        # Single pass, N chosen per-example dynamically
        for k in args.ks:
            # `n` passed to runner is used for directory labeling; actual N per example chosen inside
            n_label = "auto"
            run_attack_experiments(
                classifier,
                classifier_name,
                ds_name,
                k,
                n_label,
                args.output_base,
                args.Just_Hybrid,
                args.replace,
                (start_idx, end_idx),
                n_mode="auto",
            )
    else:
        # Manual mode: run once per N in --ns
        for n in args.ns:
            for k in args.ks:
                run_attack_experiments(
                    classifier,
                    classifier_name,
                    ds_name,
                    k,
                    n,
                    args.output_base,
                    args.Just_Hybrid,
                    args.replace,
                    (start_idx, end_idx),
                    n_mode="manual",
                )


if __name__ == "__main__":
    main()

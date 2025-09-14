from __future__ import annotations

import argparse
from transformers import pipeline

from experiments.runner import run_attack_experiments


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DynNnary: N-ary / Greedy WNR with auto/manual N.")
    p.add_argument("--output_base", type=str, default="/scratch/gilbreth/abelde/NLP_Score_Based_Attacks/Outputs_test")
    p.add_argument("--datasets", nargs="+", default=["imdb"])
    p.add_argument("--models", nargs="+", default=["jialicheng/deberta-base-imdb"])
    p.add_argument("--ns", nargs="+", type=int, default=[2, 3, 6])
    p.add_argument("--ks", nargs="+", type=int, default=[-1])
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--replace", type=str, default="wordnet")
    p.add_argument("--slice", type=str, default="0:1000")
    p.add_argument("--classifier_name", type=str, default="deberta")
    p.add_argument("--n_mode", type=str, choices=["auto", "manual"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_arguments()
    start_idx, end_idx = map(int, args.slice.split(":"))

    ds_name = args.datasets[0]
    model_id = args.models[0]
    print("Using model:", model_id)

    cls = pipeline(
        "text-classification",
        model=model_id,
        max_length=512,
        truncation=True,
        return_all_scores=True,
        device=0,
        framework="pt",
    )

    if args.greedy:
        for k in args.ks:
            run_attack_experiments(
                classifier=cls,
                classifier_name=args.classifier_name,
                ds_name=ds_name,
                k=k,
                n=1,
                output_base=args.output_base,
                greedy=True,
                replace=args.replace,
                slice_=(start_idx, end_idx),
                n_mode="manual",
            )
    else:
        if args.n_mode == "auto":
            for k in args.ks:
                run_attack_experiments(
                    classifier=cls,
                    classifier_name=args.classifier_name,
                    ds_name=ds_name,
                    k=k,
                    n="auto",
                    output_base=args.output_base,
                    greedy=False,
                    replace=args.replace,
                    slice_=(start_idx, end_idx),
                    n_mode="auto",
                )
        else:
            for n in args.ns:
                for k in args.ks:
                    run_attack_experiments(
                        classifier=cls,
                        classifier_name=args.classifier_name,
                        ds_name=ds_name,
                        k=k,
                        n=n,
                        output_base=args.output_base,
                        greedy=False,
                        replace=args.replace,
                        slice_=(start_idx, end_idx),
                        n_mode="manual",
                    )


if __name__ == "__main__":
    main()


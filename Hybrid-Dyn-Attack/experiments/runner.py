from __future__ import annotations

import csv
import json
import os
from typing import Tuple

from datasets import load_dataset
from tqdm import tqdm

from attack.hybrid import Hybrid_WNR
from metrics.perturbation import perturbation_rate, get_changed_words
from metrics.similarity import SemanticSimilarity
from utils.prediction import pred_class, get_prob, get_attack_class
from utils.n_selector import get_n_gpt_neo


def run_attack_experiments(
    classifier,
    classifier_name: str,
    ds_name: str,
    k: int,
    n: int | str,
    output_base: str,
    Just_Hybrid: bool,
    replace: str,
    slice_: Tuple[int, int],
    n_mode: str = "auto",  # "auto" or "manual"
) -> None:
    """Run a (ds, model, n, k, scheme) configuration and append results to TSV.

    n_mode controls how N is chosen per example:
      - "auto": dynamically using get_n_gpt_neo(length, ds_name)
      - "manual": use the provided `n` argument as is
    """
    scheme = "Just_Hybrid_Dyn_N" if Just_Hybrid else None

    out_dir = os.path.join(output_base, classifier_name, ds_name, scheme, f"k={k}", f"n={n}")
    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, "results.tsv")

    start_idx, end_idx = slice_
    data = load_dataset(ds_name)["test"].shuffle(seed=30).select(range(start_idx, end_idx))

    headers = [
        "id",
        "success",
        "text",
        "query count",
        "original prob",
        "attacked prob",
        "golden class",
        "original class",
        "modified class",
        "perturbation_rate",
        "similarity",
        "changed_words",
    ]

    mode = "a" if os.path.exists(tsv_path) else "w"
    start_resume = 0
    if mode == "a":
        with open(tsv_path, "r", encoding="utf-8") as f_in:
            lines = f_in.read().splitlines()
            if len(lines) > 1:
                last_id = int(lines[-1].split("\t")[0])
                start_resume = last_id + 1

    sim_model = SemanticSimilarity("sentence-transformers/all-mpnet-base-v2")

    with open(tsv_path, mode, encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        if mode == "w":
            writer.writerow(headers)

        for rel_idx in tqdm(range(start_resume, len(data)), desc=f"{ds_name}-{scheme}-n={n}-k={k}"):
            idx = start_idx + rel_idx
            ex = data[rel_idx]
            text, gold = ex["text"], int(ex["label"])

            # Choose N automatically or manually
            if n_mode == "auto":
                senlen = len(text.split())
                n_eff = get_n_gpt_neo(senlen, ds_name)
            else:
                n_eff = n

            orig_scores = classifier(text)[0]
            pred = pred_class(orig_scores)
            orig_prob = get_prob(orig_scores)

            if pred != gold:
                writer.writerow([idx, "Skipped", text, 0, orig_prob, 0.0, gold, pred, None, 0.0, 1.0, "[]"])
                continue

            split_threshold_percentage = 0.1
            if Just_Hybrid:
                succ, final_txt, qcnt, final_prob = Hybrid_WNR(
                    classifier, text, gold, n_eff, k, split_threshold_percentage
                )
            else:
                succ, final_txt, qcnt, final_prob = False, text, 1, orig_prob

            atk_label = get_attack_class(final_prob)
            prate = perturbation_rate(text, final_txt)
            simsc = sim_model(text, final_txt)
            changed = get_changed_words(text, final_txt)

            writer.writerow(
                [
                    idx,
                    succ,
                    final_txt,
                    qcnt,
                    orig_prob,
                    final_prob,
                    gold,
                    pred,
                    atk_label,
                    round(prate, 4),
                    round(simsc, 4),
                    json.dumps(changed),
                ]
            )
            if (rel_idx + 1) % 25 == 0:
                f_out.flush()

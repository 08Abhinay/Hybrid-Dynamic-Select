from __future__ import annotations


def get_n_gpt_neo(length: float, ds: str) -> int:
    """Select N dynamically based on text length bins per dataset.

    Mirrors the mapping from scripts/Hybrid_dyn_N_LLM.py
    """
    dataset_bins = {
        "imdb": [
            ((-0.001, 200.2), 3),
            ((200.2, 400.4), 3),
            ((400.4, 600.6), 3),
            ((600.6, 800.8), 3),
        ],
        "yelp_polarity": [
            ((-0.001, 135.0), 3),
            ((135.0, 270.0), 3),
            ((270.0, 405.0), 3),
            ((405.0, 540.0), 2),
            ((540.0, 675.0), 3),
        ],
        "fancyzhx/ag_news": [
            ((-0.001, 23.0), 3),
            ((23.0, 46.0), 3),
            ((46.0, 69.0), 3),
            ((69.0, 92.0), 3),
            ((92.0, 115.0), 2),
        ],
        "cornell-movie-review-data/rotten_tomatoes": [
            ((-0.001, 9.8), 3),
            ((9.8, 19.6), 3),
            ((19.6, 29.4), 6),
            ((29.4, 39.2), 2),
            ((39.2, 49.0), 3),
        ],
    }

    if ds not in dataset_bins:
        raise ValueError(f"Dataset '{ds}' is not recognized.")

    for (lower, upper), n in dataset_bins[ds]:
        if lower < length <= upper:
            return n
    return 2


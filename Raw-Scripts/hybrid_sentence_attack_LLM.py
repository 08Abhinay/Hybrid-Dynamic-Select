from tkinter.constants import N
import sys
import csv
import json

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
import requests
import os
from tqdm import tqdm
from typing import Tuple, Sequence


# Imports
from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )


"""# Utilies"""

def get_synonyms(word):
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())

    return list(synonyms - {word})

# takes in an output from textclassifier and returns the highest probability
# needed for multiclass problems
def pred_class(preds):
    pred = 0
    score = preds[0]['score']  # Start with the first element's score
    for i in range(1, len(preds)):  
        cur_score = preds[i]['score']
        if cur_score > score:
            pred = i
            score = cur_score

    return pred

def get_prob(preds):
    if isinstance(preds[0], list):  
        preds = preds[0]

    score = max(p['score'] for p in preds)
    return score

    return score
def get_attack_class(atk_prob):
    if atk_prob < 0.5:
          return 1
    else:
        return 0

# Redefining the find_word_indices function
def find_word_indices(query_tokens, subquery_tokens):
    for i in range(len(query_tokens) - len(subquery_tokens) + 1):
        if query_tokens[i:i + len(subquery_tokens)] == subquery_tokens:
            return i, i + len(subquery_tokens) - 1
    return -1, -1

def wordnet_replace(classifier, query, initial_prob, replace_pos, attack_class):
    split_query = query.split()
    # if replace_pos >= len(split_query):
    #     # replace_pos = len(split_query) - 1
    replace_word = split_query[replace_pos]
    syns = get_synonyms(replace_word)
    query_count = 0

    # if no synonyms found, can't change this word!
    if(len(syns) == 0):
        return False, query, initial_prob, query_count

    #print(syns)
    prob_list = []
    #test each synonym, choose first to cause failure, or else choose synonym which caused the greater probability drop
    for cur_syn in syns:
        cur_query = split_query[:replace_pos] + [cur_syn] + split_query[replace_pos +1:]
        cur_query = ' '.join(cur_query)
        cur_preds = classifier(cur_query)[0]
        #print(cur_query, cur_preds)
        query_count += 1
        cur_prob = cur_preds[attack_class]['score']
        cur_label = pred_class(cur_preds)

        # this means the attack flipped the label
        if(cur_label != attack_class):
            return True, cur_query, cur_prob, query_count
        else:# add to prob list for later
            prob_list.append((cur_prob, cur_query))


    # here we didn't find any synonyms to cause failure, so we choose the one which caused the greater drop
    great_drop = 0
    choice = 0
    for i in range(len(prob_list)):
        prob_drop = initial_prob - prob_list[i][0]
        if(prob_drop > great_drop):
            choice = i
            great_drop = prob_drop

    return False, prob_list[choice][1], prob_list[choice][0], query_count

"""# Hybrid Attack Algorithm"""

class NSNode:

    def __init__(self, data, prob, N = 2):
        self.data = data
        self.prob = prob
        self.exclusion = []
        self.explored = False
        self.children = []
        self.N = N
        self.prob_sorted = {}
        for _ in range(N): #initialize children
          self.children.append(None)


    def isLeaf(self): # if any children is initialized, then not a leaf
        if(len([x for x in self.children if x]) > 0):
          return False
        return True


    #prints out node, left children, data/prob/explored, then right children
    def printNode(self):
        if(self.isLeaf()):
            print(self.data, self.prob, self.explored)
        else:
            for i in range(self.N//2):
              if(self.children[i]):
                self.children[i].printNode()

            print(self.data, self.prob, self.explored)

            for i in range(self.N//2, self.N):
              if(self.children[i]):
                self.children[i].printNode()


    def lowestProb(self):
        if self.isLeaf():  # If the node is a leaf
            if self.explored:  # If it has already been explored, exit
                return None
            else:
                return self  # If it hasn't been explored, it is the lowest

        else:  # If it is not a leaf, explore its children
            lows = [None for _ in self.children]

            # Recursively call lowestProb on all children
            for i in range(len(self.children)):
                if self.children[i] and not self.children[i].explored:  # Only explore if the child is not None and not explored
                    lows[i] = self.children[i].lowestProb()

            # Check if all branches have been explored
            if all(child is None or child.explored for child in self.children):  # If all children are explored
                self.explored = True
                return None

            # Find the child with the lowest probability among non-explored nodes
            lowest_pos = -1
            lowest_prob = float('inf')

            for i in range(len(lows)):
                if lows[i] and lows[i].prob < lowest_prob:
                    lowest_pos = i
                    lowest_prob = lows[i].prob

            if lowest_pos == -1:
                self.explored = True  # If no valid children, mark this node as explored
                return None

            return lows[lowest_pos]



def n_nary_select_iterative_NSNode(classifier, query, all_segments, split_threshold_percentage = 0.1, attack_class=0,
                                   NSStruct=None, initial_prob=None, n=2, previous_segment=None):


    query_count = 0
    split_query = query.split()
    # Determine split_threshold based on sentence length if not provided
    split_threshold = max(1, int(split_threshold_percentage * len(split_query)))
    # set up initial root NSNode
    if not NSStruct:
        if not initial_prob:
            # get initial probability for query
            initial_prob = classifier(query)[0][attack_class]["score"]
        query_count += 1
        NSStruct = NSNode(query, initial_prob, n)

    initial_prob = NSStruct.prob

    # search to find lowest prob node, which has also not been explored
    cur_struct = NSStruct.lowestProb()
    if previous_segment and cur_struct in previous_segment.children:
        return 0, NSStruct, previous_segment, all_segments
    
    split_subquery = cur_struct.data.split()

    while True:

        # Base condition if the current segement is below the split threshold for non-root nodes
        if len(cur_struct.exclusion) > 0 and len(cur_struct.exclusion) <= split_threshold:
            # If the segment size is below or equal to the threshold, stop further splitting
            if cur_struct not in all_segments:
                all_segments.append(cur_struct)
            return query_count, NSStruct, cur_struct, all_segments

        for segment in all_segments:
            if cur_struct in segment.children:
                return query_count, NSStruct, segment, all_segments

        if len(cur_struct.exclusion) == 1:  # only 1 word, no need to explore more, just return
            cur_struct.explored = True  # exploring!
            return 0, NSStruct, cur_struct, all_segments # exclusion list with only 1 item is that word's position in the text
        else:  # set up start and end
            if len(cur_struct.exclusion) == 0:  # root node, no exclusion, entire text
                start, end = find_word_indices(split_query, split_subquery)
            else:
                start = cur_struct.exclusion[0]
                end = start + len(cur_struct.exclusion) - 1

        positions = [start]
        cur_pos = start
        step = (end - start) // n

        for i in range(n - 1):
            next_pos = cur_pos + step + 1
            if next_pos > len(split_query):
                continue
            positions.append(next_pos)
            cur_pos = next_pos
        positions.append(end)

        exclusions = []
        for i in range(len(positions) - 1):
            if i + 1 == len(positions) - 1:
                cur_exclusions = list(range(positions[i], positions[i + 1] + 1))
            else:
                cur_exclusions = list(range(positions[i], positions[i + 1]))
            exclusions.append(cur_exclusions)

        queries = []
        for i in range(len(exclusions)):
            if len(exclusions[i]) == 0:
                continue
            cur_n_query = ' '.join([split_query[j] for j in range(len(split_query)) if j not in exclusions[i]])
            queries.append(cur_n_query)

        prob_drops = []
        for i in range(len(queries)):
            cur_query = queries[i]
            cur_prob = classifier(cur_query)[0][attack_class]["score"]
            query_count += 1
            prob_drops.append(initial_prob - cur_prob)

            # Only create a new child NSNode if it doesn't already exist
            if cur_struct.children[i] is None:
                cur_struct.children[i] = NSNode(cur_query, cur_prob, n)
                cur_struct.children[i].exclusion = exclusions[i]

        # Determine the child with the greatest drop in probability
        g_drop = prob_drops.index(max(prob_drops))

        # If the exclusion for the selected child is only one word, mark it as explored
        if len(cur_struct.children[g_drop].exclusion) == 1:
            cur_struct.children[g_drop].explored = True

        # Update the current structure to explore the next segment
        cur_struct = cur_struct.children[g_drop]

        start = cur_struct.exclusion[0]
        end = cur_struct.exclusion[-1]

def GreedySelect(segment_to_explore, classifier, query, initial_prob, attack_class=0, excluded_indices=None, n=2):

    split_query = query.split()
    queries = []
    prob_drops = {}

    # Ensure the children list is the correct size (at least as large as excluded_indices)
    while len(segment_to_explore.children) < len(excluded_indices):
        segment_to_explore.children.append(None)  # Fill with None initially to hold space for future NSNode objects

    # Create queries for only the excluded indices and generate NSNode objects
    for exclusion in segment_to_explore.exclusion:
    # If `exclusion` is the last index, take up to `exclusion` only
        if exclusion < len(split_query) - 1:
            cur_query = ' '.join(split_query[:exclusion] + split_query[exclusion + 1:])
        else:
            cur_query = ' '.join(split_query[:exclusion])

        queries.append(cur_query)


    # Batch classifier
    batch_results = classifier(queries)
    # Process results and update children and prob_sorted
    for i, exclusion in enumerate(segment_to_explore.exclusion):
        # Get probability for attack class
        cur_prob = batch_results[i][attack_class]['score']
        prob_drops[i] = initial_prob - cur_prob  # Store the probability drop
        ns_node = NSNode(data=split_query[exclusion], prob=cur_prob, N=n)
        segment_to_explore.children[i] = ns_node

    return prob_drops, segment_to_explore


def GS_WNR(segment_to_explore, classifier, initial_prob, query, attack_class, k=-1, n=2):

    query_count = 0
    cur_query = query

    # Check if `prob_sorted` is already fully populated
    if len(segment_to_explore.prob_sorted) < len(segment_to_explore.exclusion):
        # If not, call GreedySelect to fill in missing values
        prob_drops, segment_to_explore = GreedySelect(
            segment_to_explore, classifier, query, initial_prob,
            attack_class=attack_class,
            excluded_indices=list(range(len(segment_to_explore.exclusion))),  # Exclude all indices initially
            n=n
        )

        segment_to_explore.prob_sorted.update(prob_drops)

        # Step 1: Initialize a list to hold valid drops
        valid_prob_drops = []

        # Step 2: Iterate through the items of the prob_sorted dictionary
        for _, probability in segment_to_explore.prob_sorted.items():
            # Step 3: Filter out invalid entries
            if probability is not None:
                # Add valid drops to the list
                valid_prob_drops.append(probability)

        # Step 4: Count the number of valid drops and add it to query_count
        query_count += len(valid_prob_drops)

    # Sort words by their probability drop and pick the one with the highest drop
    prob_sorted = dict(sorted(segment_to_explore.prob_sorted.items(), key=lambda item: item[1], reverse=True))

    # Loop through the sorted words and find the first unexplored one
    for index in prob_sorted.keys():
        if segment_to_explore.children[index].explored:
            continue
        replace_pos = segment_to_explore.exclusion[index]
        # Attempt to replace the word using WordNet
        success, cur_query, cur_prob, queries = wordnet_replace(
            classifier, cur_query, initial_prob, replace_pos, attack_class
        )

        query_count += queries  # Update the query count based on WordNet replacement calls

        if success:
            segment_to_explore.children[index].explored = True
            return True, cur_query, query_count, cur_prob, segment_to_explore
        else:
            # Mark the word as explored if the attack failed
            segment_to_explore.children[index].explored = True
            break # Exit after one `wordnet_replace` attempt

    return False, cur_query, query_count, cur_prob, segment_to_explore

def get_most_influential_sentence(classifier, query, attack_class, root, n=2):
    query_count = 0
    original_prob = classifier(query)[0][attack_class]["score"]
    query_count += 1
    sentences = sent_tokenize(query)
    num_sentences = len(sentences)

    # Initialize the root node if not provided
    if not root:
        root = NSNode(query, original_prob, N=num_sentences)
        root.children = [None] * num_sentences  # Initialize the children list

    # Ensure all sentence nodes are initialized
    for i, sentence in enumerate(sentences):
        if root.children[i] is None:
            root.children[i] = NSNode(sentence, original_prob, N=n)

    # Only compute probabilities if not already computed
    if len(root.prob_sorted) < num_sentences:
        # Generate modified texts excluding one sentence at a time
        modified_texts = [
            ' '.join(sentences[:i] + sentences[i+1:])
            for i in range(num_sentences)
        ]

        # Classify the modified texts in batch
        modified_preds = classifier(modified_texts)
        query_count += len(modified_preds)

        # Update probabilities and root's prob_sorted dictionary
        for i, pred in enumerate(modified_preds):
            prob = pred[attack_class]['score']
            root.children[i].prob = prob
            root.prob_sorted[i] = prob

    max_change = 0


    # Find the most influential sentence
    for i, child in enumerate(root.children):
        if not child.explored:
            child_prob = child.prob
            prob_change = original_prob - child_prob
            # Update the most influential sentence if the change is greater
            if prob_change > max_change:
                max_change = prob_change


    return root, query_count, original_prob


def Hybrid_WNR(classifier, query, attack_class, n=2, k=7, split_threshold_percentage=None):
    """
    A hybrid approach combining n-ary search and greedy search with a split threshold.

    Args:
    classifier: The classifier pipeline to evaluate the queries.
    query: The input query to attack.
    attack_class: The target class to attack.
    n: Initial n for n-ary search.
    k: Maximum number of word replacements allowed.
    split_threshold: The threshold at which n-ary splitting stops and switches to greedy search.

    Returns:
    success: True if the attack was successful, False otherwise.
    final_query: The modified query after the attack.
    query_count: The number of queries made during the attack.
    final_prob: The final probability of the attack class.
    """
    done = False
    query_count = 0
    root_node = None

    cur_query = query  # Initialize the current version of the query
    words_changed = 0  # Track the number of words changed in the query

    all_segments = []
    segment_to_explore = None
    # Continue until either a successful attack is found or the process is terminated

    root_node, queries, initial_prob = get_most_influential_sentence(classifier, query,
                                                                                attack_class, root=None, n=n)
    query_count += queries

    while not done:
        # Use n-ary search to identify word replacements until the threshold is reached
        if root_node:
            # Use the iterative n-ary search method with an existing structure from previous iterations
            queries, root_node, segment_to_explore, all_segments= n_nary_select_iterative_NSNode(
                classifier, query, all_segments, split_threshold_percentage, attack_class,
                NSStruct=root_node, initial_prob=initial_prob,
                n=n, previous_segment=segment_to_explore,
            )

        # Update query count with the number of queries made during n-ary search
        query_count += queries

        success, cur_query, queries, cur_prob, segment_to_explore = GS_WNR(
                segment_to_explore,  # Use the segment identified by n-ary search
                classifier,
                initial_prob,
                query=cur_query,  # Curre2nt modified query
                attack_class=attack_class,
                k=k,  # Maximum number of word replacements allowed
                n=n
            )

        query_count += queries  # Update the total query count
        words_changed += 1

        if(success):
            return True, cur_query, query_count, cur_prob
        else:
            #check if everything explored
            if(not root_node.lowestProb()):
                done = True

        if(k != -1 and words_changed >= k):
            done = True

    # Return the original query and its probability if no successful attack is found
    return False, cur_query, query_count, initial_prob

# Perturbation Rate
def perturbation_rate(original_text: str, attacked_text: str) -> float:
    """
    Compute fraction of changed words between original and attacked text.
    The standard definition used in most adversarial NLP work:
        (# changed words) / (# words in original).
    """
    orig_tokens = word_tokenize(original_text)
    attacked_tokens = word_tokenize(attacked_text)
    
    # Count how many positions differ (up to the shorter length)
    min_len = min(len(orig_tokens), len(attacked_tokens))
    changed = sum(
        1 for i in range(min_len)
        if orig_tokens[i] != attacked_tokens[i]
    )
    # If there's a length mismatch, remaining tokens are also considered "changed"
    changed += abs(len(orig_tokens) - len(attacked_tokens))

    # Avoid division by zero if the original text is empty
    if len(orig_tokens) == 0:
        return 0.0

    return changed / len(orig_tokens)


#  Semantic Similarity with Sentence-BERT
class SemanticSimilarity:
    """
    Uses a Sentence-BERT model (e.g., 'all-mpnet-base-v2') to measure
    the cosine similarity between two sentence embeddings.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        # Load a Sentence-BERT model
        self.model = SentenceTransformer(model_name)

    def __call__(self, original_text: str, attacked_text: str) -> float:
        """
        Returns a scalar similarity (0 to 1). 
        1 means identical in embedding space; 0 means totally different.
        """
        # Encode sentences
        # Note: set `convert_to_tensor=True` for a torch.Tensor output
        orig_emb = self.model.encode(original_text, convert_to_tensor=True)
        atk_emb = self.model.encode(attacked_text, convert_to_tensor=True)

        # Cosine similarity: (a · b) / (||a||*||b||)
        cos_sim = torch.nn.functional.cosine_similarity(orig_emb, atk_emb, dim=0)
        return cos_sim.item()



def get_changed_words(original_text: str, attacked_text: str):
    """
    Return a list of positions where tokens differ between original and attacked text,
    including insertions/deletions if the lengths differ.

    Each item in the returned list is a tuple:
       (token_index, original_token, attacked_token)

    If attacked text is longer, the extra tokens appear as (index, None, new_token).
    If original text is longer, the missing tokens appear as (index, old_token, None).
    """
    orig_tokens = word_tokenize(original_text)
    attacked_tokens = word_tokenize(attacked_text)

    min_len = min(len(orig_tokens), len(attacked_tokens))
    changed = []

    # 1) Identify changed tokens where both strings still have positions
    for i in range(min_len):
        if orig_tokens[i] != attacked_tokens[i]:
            changed.append((i, orig_tokens[i], attacked_tokens[i]))

    # 2) Handle length mismatch: insertions or deletions
    if len(attacked_tokens) > len(orig_tokens):
        # Attacked text has extra tokens
        for i in range(min_len, len(attacked_tokens)):
            changed.append((i, None, attacked_tokens[i]))
    elif len(orig_tokens) > len(attacked_tokens):
        # Original text had extra tokens that disappeared
        for i in range(min_len, len(orig_tokens)):
            changed.append((i, orig_tokens[i], None))

    return changed

# Define Sentiment Predictor Class
class SentimentPredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text: str):
        return self.predict_batch([text])

    def predict_batch(self, texts: list):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)

        id2label = self.model.config.id2label

        # Build results per sample
        results = []
        for prob in probs:
            result = [
                {"label": id2label[i], "score": round(prob[i].item(), 4)}
                for i in range(len(prob))
            ]
            results.append(result)
        return results

import os, csv, json, argparse
from pathlib import Path 
#
# -----------------------------------------------------------------------------#
#                                  Helpers                                     #
# -----------------------------------------------------------------------------#
def run_attack_experiments(
    classifier,
    classifier_name: str,
    ds_name: str,
    k: int,
    n: int,
    output_base: str,
    HybridSent: bool,
    replace: str,
    slice_: Tuple[int, int],
) -> None:
    """
    Run a *single* (ds, model, n, k, scheme) combo and append results to TSV.

    Parameters
    ----------
    slice_  tuple(start, end)
        Which part of the test split to attack (Python-style half-open).
    """
    scheme = "Hybrid_Sentence" if HybridSent else None
        
    out_dir = os.path.join(
        output_base, classifier_name, scheme, ds_name, f"k={k}", f"n={n}"
    )
    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, "results.tsv")

    # ------------------------------------------------------------------ data --
    start_idx, end_idx = slice_
    print(ds_name)
    data = (
        load_dataset(ds_name)["test"]
        .shuffle(seed=30)
        .select(range(start_idx, end_idx))
    )

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
    # resume if the TSV already exists
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

        print(f"Starting from {start_resume}")
        for rel_idx in tqdm(
            range(start_resume, len(data)),
            desc=f"{ds_name}-{scheme}-n={n}-k={k}",
        ):
            idx = start_idx + rel_idx           # global ID (0-999)
            ex  = data[rel_idx]
            text, gold = ex["text"], int(ex["label"])

            # -------------------------------------------------------- predict --
            orig_scores = classifier(text)[0]
            pred        = pred_class(orig_scores)
            orig_prob   = get_prob(orig_scores)

            # skip misclassified originals
            if pred != gold:
                writer.writerow(
                    [
                        idx,
                        "Skipped",
                        text,
                        0,
                        orig_prob,
                        0.0,
                        gold,
                        pred,
                        None,
                        0.0,
                        1.0,
                        "[]",
                    ]
                )
                continue
           
            # --------- run the attack --------------------
            split_threshold_percentage = 0.1
            if HybridSent:
                succ, final_txt, qcnt, final_prob = Hybrid_WNR(
                    classifier, text, gold, n, k, split_threshold_percentage
                )

            atk_label = get_attack_class(final_prob)
            prate     = perturbation_rate(text, final_txt)
            simsc     = sim_model(text, final_txt)
            changed   = get_changed_words(text, final_txt)

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

    print(
        f"✓ Done {ds_name}, scheme={scheme}, n={n}, "
        f"k={k}, slice={slice_}  →  {tsv_path}"
    )


# -----------------------------------------------------------------------------#
#                                    Main                                      #
# -----------------------------------------------------------------------------#
def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-run Hybrid-Sentence attacks over multiple (ds,model,n,k)."
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
        help="values of N for N-ary WNR",
    )
    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[-1],
        help="values of K (synonym candidates)",
    )
    p.add_argument(
        "--HybridSent",
        action="store_true",
        help="Run experiments for HybridSentence",
    )
    p.add_argument(
        "--replace",
        type=str,
        default="wordnet",
        help="synonym source",
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
        help="Mention the name of the classifier for the output path",
    )
    return p.parse_args()

from transformers import pipeline
import torch

def main() -> None:
    args = parse_arguments()
    start_idx, end_idx = map(int, args.slice.split(":"))

    ds_name = args.datasets[0]
    model_id = args.models[0]
    print("Using model:", model_id)

    # --------------- build the classifier -----------------
    classifier = pipeline(
        "text-classification",
        model=model_id,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_all_scores=True,
        device=0,          # GPU‑0
        framework="pt",
    )

    # --------------- one‑line monkey‑patch ----------------
    orig_preprocess = classifier.preprocess
    def _ensure_long_ids(inputs, **kwargs):
        model_inputs = orig_preprocess(inputs, **kwargs)
        if "input_ids" in model_inputs:
            model_inputs["input_ids"] = model_inputs["input_ids"].long()
        if "token_type_ids" in model_inputs:          # (BERT‑style models)
            model_inputs["token_type_ids"] = model_inputs["token_type_ids"].long()
        return model_inputs
    classifier.preprocess = _ensure_long_ids
    # ------------------------------------------------------

    classifier_name = args.classifier_name
    print(f"Entering attack loop for {ds_name}")

    for n in args.ns:
        for k in args.ks:
            run_attack_experiments(
                classifier,
                classifier_name,
                ds_name,
                k,
                n,
                args.output_base,
                args.HybridSent,
                args.replace,
                (start_idx, end_idx),
            )

if __name__ == "__main__":
    main()

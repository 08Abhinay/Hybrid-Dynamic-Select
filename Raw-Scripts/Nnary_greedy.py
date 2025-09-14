
from tkinter.constants import N
import sys
import csv
from transformers import pipeline
import json

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
import requests
import os
from tqdm import tqdm

class NSNode:

    def __init__(self, data, prob, N = 2):
        self.data = data
        self.prob = prob
        self.exclusion = []
        self.explored = False
        self.children = []
        self.N = N
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
        if(self.isLeaf()): # cannot explore further
            if(self.explored): # already explored, exit
                return None
            else:
                return self # only one node here, so must be the lowest

        else: #explore right and left
            lows = [None for _ in self.children]

            for i in range(self.N):
              if(self.children[i]):
                lows[i] = self.children[i].lowestProb()


            if(len([x for x in lows if x]) == 0): #all branches explored, which means this node already is fully explored
              self.explored = True
              return None


            #find lowest prob, check all that exist (ie not None)
            lowest_pos = 0
            lowest_prob = 10000

            for i in range(self.N):
              if(lows[i]):
                if(lows[i].prob < lowest_prob):
                  lowest_pos = i
                  lowest_prob = lows[i].prob

            return lows[lowest_pos]

def n_nary_select_iterative_NSNode(classifier, query, attack_class = 0, NSStruct = None, initial_prob = None, n = 2): #for analysis

    query_count = 0

    # set up initial root BSNode
    if(not NSStruct):
        if(not initial_prob):
            # get initial probability for query
            initial_prob = classifier(query)[0][attack_class]["score"]

        query_count += 1

        NSStruct = NSNode(query, initial_prob, n)



    initial_prob = NSStruct.prob

    # search to find lowest prob node, which has also not been explored
    cur_struct = NSStruct.lowestProb()
    final_prob = initial_prob

    split_query = query.split()


    if(len(cur_struct.exclusion) == 1): # only 1 word, no need to explore more, just return
        cur_struct.explored = True # exploring!
        return cur_struct.exclusion[0], cur_struct.prob, 0, NSStruct #exclusion list with only 1 item is that word's position in the text
    else: # set up start and end
        if(len(cur_struct.exclusion) == 0): # root node, no exclusion, entire text
            start = 0
            end = len(split_query) - 1#makes slicing easier
        else:
            # need to update to nnary still...
            start = cur_struct.exclusion[0]
            end = start + len(cur_struct.exclusion) - 1

    while start < end:
      positions = [start]
      cur_pos = start
      #cur_struct.printNode()

      step = (end - start)//n
      for i in range(n - 1):
        next_pos = cur_pos + step + 1
        positions.append(next_pos)
        cur_pos = next_pos

      positions.append(end)

      exclusions = []
      for i in range(len(positions) - 1):
        if(i+1 == len(positions) - 1):
          cur_exclusions = list(range(positions[i], positions[i+1] + 1))
        else:
          cur_exclusions = list(range(positions[i], positions[i+1]))
        exclusions.append(cur_exclusions)

      #print(exclusions)
      queries = []
      for i in range(len(exclusions)):
        if(len(exclusions[i]) == 0):
          continue

        cur_n_query = ' '.join([split_query[j] for j in range(len(split_query)) if j not in exclusions[i]])
        queries.append(cur_n_query)


      prob_drops = []
      for i in range(len(queries)):
        cur_query = queries[i]
        cur_prob = classifier(cur_query)[0][attack_class]["score"]
        query_count += 1
        prob_drops.append(initial_prob - cur_prob)
        #update NSStruct
        cur_struct.children[i] = NSNode(cur_query, cur_prob, n)
        cur_struct.children[i].exclusion = exclusions[i]

      g_drop = prob_drops.index(max(prob_drops))
      if(len(cur_struct.children[g_drop].exclusion) == 1):
        cur_struct.children[g_drop].explored = True

      final_prob = prob_drops[g_drop]
      cur_struct = cur_struct.children[g_drop]
      start = cur_struct.exclusion[0]
      end = cur_struct.exclusion[-1]


      most_influential_pos = start


    return most_influential_pos, final_prob, query_count, NSStruct



def get_synonyms(word):
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())

    return list(synonyms - {word})

# takes in an output from textclassifier and returns the highest probability
# needed for multiclass problems
# def pred_class(preds):
#     """
#     Given the model's output (list of dicts), dynamically determine the label names
#     and return:
#       - 1 for the label with the highest sentiment (e.g., "POSITIVE" or "FAKE")
#       - 0 for the label with the lowest sentiment (e.g., "NEGATIVE" or "REAL")
    
#     Works with any classifier without hardcoded label names.
#     """
#     # Get the highest scoring label
#     best_dict = max(preds, key=lambda x: x['score'])
    
#     # Extract all label names
#     label_names = [p['label'] for p in preds]

#     # Sort labels lexicographically (this ensures "NEGATIVE" < "POSITIVE", "REAL" < "FAKE", etc.)
#     sorted_labels = sorted(label_names)

#     # Assign 0 to the "lower" label and 1 to the "higher" label
#     return 1 if best_dict['label'] == sorted_labels[-1] else 0
# def pred_class(preds):
#     """
#     Given the model's output (list of dicts), return the label with the highest confidence score.
    
#     Works with any classifier that outputs numerical labels.
#     """
#     if isinstance(preds[0], list):  
#         preds = preds[0]  # Extract the inner list if it's nested

#     # Get the label with the highest confidence score
#     best_dict = max(preds, key=lambda x: x['score'])
#     return best_dict['label']


# def get_prob(preds):
#     score = preds[0]['score']

#     for i in range(1, len(preds)):
#         cur_score = preds[i]['score']
#         if(cur_score > score):
#             score = cur_score

#     return score
def pred_class(preds):
    pred = 0
    score = preds[0]['score']  # Start with the first element's score
    for i in range(1, len(preds)):  
        cur_score = preds[i]['score']
        if cur_score > score:
            pred = i
            score = cur_score

    return pred

def wordnet_replace(classifier, query, initial_prob, replace_pos, attack_class):
    split_query = query.split()
    if replace_pos >= len(split_query):
        replace_pos = len(split_query) - 1
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


# fill_mask = pipeline("fill-mask", model="bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def bert_replace(classifier, query, initial_prob, replace_pos, attack_class):
    split_query = query.split()
    if replace_pos >= len(split_query):
        replace_pos = len(split_query) - 1

    replace_word = split_query[replace_pos]
    mask_token = tokenizer.mask_token

    masked_query = ' '.join(split_query)
    # Replace the word at the specified position with [MASK]
    split_query[replace_pos] = mask_token
    
    tokenized_query = tokenizer.encode(masked_query, add_special_tokens=False)

    # Manually truncate to a maximum of 512 tokens while keeping the mask in place
    if len(tokenized_query) > 512:
        tokenized_query = tokenized_query[:512]  # Just truncate if over the limit

    # Ensure that the [MASK] token is in the sequence and at the correct position
    masked_query = tokenizer.decode(tokenized_query, skip_special_tokens=True)

    # Check if we still have a valid [MASK] token in the input
    if mask_token not in masked_query:
        return False, query, initial_prob, 0  # No [MASK] token found after truncation
    
    # Get the mask infill predictions
    mask_predictions = fill_mask(masked_query, top_k=10)

    # Filter only valid candidates and select their tokens
    mask_candidates = [pred['token_str'] for pred in mask_predictions if pred['token_str'] != replace_word]
    query_count = 0

    # If no suitable infill found, cannot change this word
    if not mask_candidates:
        return False, query, initial_prob, query_count

    prob_list = []
    # Test each candidate and choose the first to cause failure or the one with the highest probability drop
    for candidate in mask_candidates:
        cur_query = split_query[:replace_pos] + [candidate] + split_query[replace_pos + 1:]
        cur_query = ' '.join(cur_query)

        cur_preds = classifier(cur_query)[0]
        query_count += 1
        cur_prob = cur_preds[attack_class]['score']
        cur_label = pred_class(cur_preds)

        # If the attack flipped the label, return immediately
        if cur_label != attack_class:
            return True, cur_query, cur_prob, query_count
        else:
            prob_list.append((cur_prob, cur_query))

    # If no synonym caused a label flip, choose the one with the greatest probability drop
    great_drop = 0
    choice = 0
    for i in range(len(prob_list)):
        prob_drop = initial_prob - prob_list[i][0]
        if prob_drop > great_drop:
            choice = i
            great_drop = prob_drop

    return False, prob_list[choice][1], prob_list[choice][0], query_count


# takes in a text and attempts to flip the label with n-nary selection and wordnet synonym replacement
def NS_WNR(classifier, query, attack_class, n = 2, k = -1, replace = "wordnet"):
    done = False
    query_count = 0
    cur_struct = None
    initial_prob = classifier(query)[0][attack_class]["score"]
    query_count += 1

    cur_query = query
    chars_changed = 0

    while(not done):
        # call binary select
        #
        if(cur_struct):
            replace_pos, final_prob, queries, cur_struct = n_nary_select_iterative_NSNode(classifier, query, attack_class, NSStruct = cur_struct, initial_prob = initial_prob, n = n)
        else:
            replace_pos, final_prob, queries, cur_struct = n_nary_select_iterative_NSNode(classifier, query, attack_class, initial_prob = initial_prob, n = n)

        #cur_struct.printNode()

        #add new amount of queries
        query_count += queries


        # try and replace with wordnet replace
        if(replace == "bert"):
          success, cur_query, cur_prob, queries = bert_replace(classifier, cur_query, initial_prob, replace_pos, attack_class)
        else:
          success, cur_query, cur_prob, queries = wordnet_replace(classifier, cur_query, initial_prob, replace_pos, attack_class)
        chars_changed += 1


        query_count += queries
        #print(success, cur_query, cur_prob, query_count)

        if(success):
            return True, cur_query, query_count, cur_prob
        else:
            #check if everything explored
            if(not cur_struct.lowestProb()):
                done = True

        if(k != -1 and chars_changed >= k):
            done = True

    return False, cur_query, query_count, cur_prob


def GreedySelect(classifier, query, attack_class = 0, initial_prob = None):
    done = False
    if(not initial_prob):
        initial_prob = classifier(query)[0][attack_class]["score"]

    prob_drops = {}

    split_query = query.split()
    # remove each word and then note the drop in probability
    for i in range(len(split_query)):
        cur_query = ' '.join(split_query[:i] + split_query[i+1:])
        cur_prob = classifier(cur_query)[0][attack_class]["score"]
        prob_drops[i] = initial_prob - cur_prob


    return prob_drops

def GS_WNR(classifier, query, attack_class, k = -1, replace = "wordnet"):
  done = False
  query_count = 0
  cur_struct = None
  initial_prob = classifier(query)[0][attack_class]["score"]

  query_count += 1

  cur_query = query

  prob_drops = GreedySelect(classifier, query, attack_class, initial_prob)
  query_count += len(prob_drops)

  prob_sorted = dict(sorted(prob_drops.items(), key=lambda item: item[1], reverse=True))

  chars_changed = 0

  while(not done):

      replace_pos = list(prob_sorted.keys())[0]
      prob_sorted.pop(replace_pos)

      # try and replace with wordnet replace
      if replace == "bert":
        success, cur_query, cur_prob, queries = bert_replace(classifier, cur_query, initial_prob, replace_pos, attack_class)
      else:
        success, cur_query, cur_prob, queries = wordnet_replace(classifier, cur_query, initial_prob, replace_pos, attack_class)
      chars_changed += 1

      query_count += queries
      #print(success, cur_query, cur_prob, query_count)

      if(success):
          return True, cur_query, query_count, cur_prob
      else:
          #check if everything explored
          if(len(prob_sorted) == 0):
              done = True

      if(k != -1 and chars_changed >= k):
          done = True

  return False, cur_query, query_count, cur_prob


# def pred_class(preds):
#     """
#     Given the model's output (list of dicts), return the label with the highest confidence score.
    
#     Works with any classifier that outputs numerical labels.
#     """
#     if isinstance(preds[0], list):  
#         preds = preds[0]  # Extract the inner list if it's nested

#     # Get the label with the highest confidence score
#     best_dict = max(preds, key=lambda x: x['score'])
#     return best_dict['label']



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

import nltk
from nltk.tokenize import word_tokenize
import json

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

# -------------------------------------------------------------------
# The "run_attack_experiments" method with new columns
# -------------------------------------------------------------------
def run_attack_experiments(classifier, ds, k, n, greedy=False, replace="wordnet"):
    """
    Runs an attack experiment on dataset `ds` using `classifier`.
    - If `greedy` is True, uses GS_WNR (greedy).
    - Otherwise uses NS_WNR (n-ary).
    - Saves results to .tsv, appending new metrics:
        - Perturbation Rate
        - Semantic Similarity
        - Words Changed
    """
    # Decide on short dataset name & path
    if ds == "cornell-movie-review-data/rotten_tomatoes":
        temp_ds_name = "rottom"
    else:
        temp_ds_name = "imdb"

    # Build path based on "greedy" or "n-ary"
    if greedy:
        path = f"/scratch/gilbreth/abelde/NLP_Score_Based_Attacks/Outputs/Deberta/Greedy/{temp_ds_name}/k={k}/n={n}"
        out_filename = f"{temp_ds_name}_Greedy_WNR.tsv"
    else:
        path = f"/scratch/gilbreth/abelde/NLP_Score_Based_Attacks/Outputs/Deberta/Nnary/{temp_ds_name}/k={k}/n={n}"
        out_filename = f"{temp_ds_name}_Nnary.tsv"

    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, out_filename)

    # Load & shuffle dataset
    data = load_dataset(ds)
    ds_test_data = data['test'].shuffle(seed=30)
    dataset = ds_test_data.select(range(1000))

    # 1) Extra columns for metrics
    headers = [
        'id', 'success', 'text', 'query count', 'original prob',
        'attacked prob', 'golden class', 'original class', 'modified class',
        'perturbation_rate', 'similarity', 'changed_words'
    ]

    # 2) Resume if file exists
    start_idx = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.read().splitlines()
            if len(lines) > 1:
                last_line = lines[-1]
                cols = last_line.split('\t')
                start_idx = int(cols[0]) + 1

    mode = 'a' if os.path.exists(output_path) else 'w'
    
    # 3) Initialize similarity model once to avoid reloading each loop
    similarity_model = SemanticSimilarity("sentence-transformers/all-mpnet-base-v2")

    with open(output_path, mode, encoding='utf-8', newline='') as hybrid_f:
        writer = csv.writer(hybrid_f, delimiter='\t')
        if mode == 'w':
            writer.writerow(headers)

        for idx in tqdm(range(start_idx, len(dataset)), desc="Running attack experiments"):
            example = dataset[idx]
            text = example['text']
            gold_label = int(example['label'])

            # Original classification
            orig_pred = classifier.predict(text)[0]
            predicted_label = pred_class(orig_pred)
            orig_prob = get_prob(orig_pred)

            # Skip misclassified
            if predicted_label != gold_label:
                writer.writerow([
                    idx, 'Skipped', text, 0, orig_prob, 0.0,
                    gold_label, predicted_label, 'None',
                    0.0, 1.0, "[]"
                ])
            else:
                # Attack
                if greedy:
                    success, final_query, query_count, final_prob = GS_WNR(
                        classifier, text, gold_label, k, replace
                    )
                else:
                    success, final_query, query_count, final_prob = NS_WNR(
                        classifier, text, gold_label, n, k, replace
                    )

                attacked_label = get_attack_class(final_prob)

                # ---- NEW METRICS ----
                p_rate = round(perturbation_rate(text, final_query), 4)
                sim_score = round(similarity_model(text, final_query), 4)
                changed = get_changed_words(text, final_query)
                # We'll store the changed words as a JSON string for a "nice" format
                changed_str = json.dumps(changed)

                writer.writerow([
                    idx, success, final_query, query_count,
                    orig_prob, final_prob,
                    gold_label, predicted_label, attacked_label,
                    p_rate, sim_score, changed_str
                ])

            if (idx + 1) % 25 == 0:
                hybrid_f.flush()

    print(f"\nAll done! Results saved to: {output_path}")



# Run experiments for each dataset, classifier, n, and k
def main():
    device = torch.device(('cuda'
                          if torch.cuda.is_available() else 'cpu'))
    # imdb_classifier = pipeline(
    #     'text-classification',
    #     model='textattack/distilbert-base-uncased-imdb',
    #     max_length=512,
    #     truncation=True,
    #     return_all_scores=True,
    #     device=device,
    #     )
    # yelp_classifier = pipeline(
    #     'text-classification',
    #     model='randellcotta/distilbert-base-uncased-finetuned-yelp-polarity',
    #     max_length=512,
    #     truncation=True,
    #     return_all_scores=True,
    #     device=device,
    #     )
    
    # ag_news_classifier = pipe = pipeline(
    #     "text-classification", 
    #     model="textattack/distilbert-base-uncased-ag-news",
    #     max_length=512,
    #     truncation=True,
    #     return_all_scores=True,
    #     device=device,)
    
    

    imdb_deberta_classifier = pipeline("text-classification", model="jialicheng/deberta-base-imdb", max_length=512, truncation=True, return_all_scores=True, device=0, framework="pt")    
    # yelp_deberta_classifier = pipeline("text-classification", 
    #                                 model="utahnlp/yelp_polarity_microsoft_deberta-v3-base_seed-2",
    #                                 return_all_scores=True,
    #                                 max_length=512, truncation=True,
    #                                 device=0,
    #                                 framework="pt")
    # rottom_deberta_classifier = pipeline("text-classification", model="utahnlp/rotten_tomatoes_microsoft_deberta-v3-base_seed-2", max_length=512, truncation=True, return_all_scores=True, device=0, framework="pt") 
    # agnews_deberta_classifier = pipeline("text-classification", model="utahnlp/ag_news_microsoft_deberta-v3-base_seed-2", max_length=512, truncation=True, return_all_scores=True,device=0, framework="pt")
    # rottom_distilbert_classifier = pipeline("text-classification", model="textattack/distilbert-base-uncased-rotten-tomatoes", max_length=512, truncation=True, return_all_scores=True,device=0, framework="pt")                             
    # replace = 'wordnet'
    # rottom_distilbert_classifier = SortedPipeline("textattack/distilbert-base-uncased-rotten-tomatoes")
    # rottom_deberta_classifer = SortedPipeline("utahnlp/rotten_tomatoes_microsoft_deberta-v3-base_seed-2")
    # agnews_deberta_classifier = SortedPipeline("utahnlp/ag_news_microsoft_deberta-v3-base_seed-2")
    
    # data = ["cornell-movie-review-data/rotten_tomatoes","fancyzhx/ag_news"]
    # data = ["fancyzhx/ag_news"]
    
    print("Starting attack for Nnary6, deberta")
    classifier = imdb_deberta_classifier
    run_attack_experiments(classifier, ds="imdb", k=-1, n=6, greedy=False, replace="wordnet")
    print("Finished!")

if __name__=="__main__":
    main()
    
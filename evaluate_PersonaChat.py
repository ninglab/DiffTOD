import openai
import anthropic
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from bert_score import score

import nltk
import numpy as np
import json
import random
import torch
import argparse
import os

# all the hyps/refs should be tokenized with nltk.word_tokenize()
def calc_bleu(hyps, refs):
    return sentence_bleu([refs], hyps, smoothing_function=SmoothingFunction().method1)
    
def calc_f1(hyps, refs):
    """ Calculate word-level f1 score """
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for response, golden_response in zip(hyps, refs):
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total if pred_char_total > 0 else 0
    r = hit_char_total / golden_char_total if golden_char_total > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1
    
def calc_distinct_n_gram(hyps,n):
    hyp_ngrams = []
    hyp_ngrams += nltk.ngrams(hyps, n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(set(hyp_ngrams))
    if total_ngrams == 0:
        return 0
    else:
        return  unique_ngrams/total_ngrams

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = calc_bleu(s1, s2)
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx], idx

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="./PersonaChat_test", type=str)
parser.add_argument("--num", default=10, type=int)
args = parser.parse_args()
indices = list(range(1, args.num + 1))
files_list = []
original_content_dict = {}
keywords_content_dict = {}
planned_content_dict = {}
for index in indices:
    path = os.path.join(args.file_path, "PersonaChat_test_%d.json"%index)
    f = open(path, "r")
    files_list.append(f)
    for line in f.readlines():
        sample = json.loads(line.rstrip("\n"))
        original_content_dict[sample["id"]] = sample["original_dialogue"]
        keywords_content_dict[sample["id"]] = sample["key_word_list"]
        planned_dialogues_list = planned_content_dict.get(sample["id"], [])
        planned_dialogues_list.append(sample["planned_dialogue"])
        planned_content_dict[sample["id"]] = planned_dialogues_list
bleu_list = []
dist1_list = []
f1_list = []
bert_score_list = []
for dialogue_id in original_content_dict.keys():
    refs = original_content_dict[dialogue_id]
    hyps = planned_content_dict[dialogue_id]
    refs_tokenized = nltk.word_tokenize(refs)
    hyps_tokenized = [nltk.word_tokenize(hyp) for hyp in hyps]
    hyps_best_tokenized, idx = selectBest(hyps_tokenized)
    #idx = random.randint(0, args.num - 1)
    hyps_best_tokenized = hyps_tokenized[idx]
    hyps_best = hyps[idx]
    bleu_list.append(calc_bleu(hyps_best_tokenized, refs_tokenized))
    dist1_list.append(calc_distinct_n_gram(hyps_best_tokenized, 1))
    f1_list.append(calc_f1(hyps_best_tokenized, refs_tokenized))
    bert_score_list.append(score([hyps_best], [refs], lang='en')[2].item())
print("bleu", np.mean(np.array(bleu_list)))
print("dist-1", np.mean(np.array(dist1_list)))
print("f1", np.mean(np.array(f1_list)))
print("bert score", np.mean(np.array(bert_score_list)))
for f in files_list:
    f.close()
    

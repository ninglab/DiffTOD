import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import json
import os
from openai import OpenAI
import re
from itertools import groupby
import nltk
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import wordnet

backend = "gpt4o"
os.environ['OPENAI_API_KEY'] = ''
    
def messagefunction(messages):
    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o", messages=messages)
    return completion.choices[0].message.content.strip()

def detect_keywords_in_sentence(sentence, keywords_list):
    sentence_keyword_list = []
    sentence_tokenized = nltk.word_tokenize(sentence)
    for word in sentence_tokenized:
        current_turn_keywords = set()
        for keyword in keywords_list:
            if keyword in [wnl.lemmatize(word, wordnet.VERB), wnl.lemmatize(word, wordnet.NOUN), wnl.lemmatize(word, wordnet.ADJ), wnl.lemmatize(word, wordnet.ADV)]:
                current_turn_keywords.add(keyword)
        for word in current_turn_keywords:
            sentence_keyword_list.append(word)
    sentence_keyword_list = [key for key, _ in groupby(sentence_keyword_list)]
    return sentence_keyword_list

def calc_bleu(hyps, refs):
    return sentence_bleu([refs], hyps, smoothing_function=SmoothingFunction().method1)

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

num = 10
indices = list(range(1, num + 1))
files_list = []
original_content_dict = {}
keywords_content_dict = {}
planned_content_dict = {}
for index in indices:
    f = open("PersonaChat_test/PersonaChat_test_%d.json"%index, "r")
    files_list.append(f)
    for line in f.readlines():
        sample = json.loads(line.rstrip("\n"))
        original_content_dict[sample["id"]] = sample["original_dialogue"]
        keywords_content_dict[sample["id"]] = sample["key_word_list"]
        planned_dialogues_list = planned_content_dict.get(sample["id"], [])
        planned_dialogues_list.append(sample["planned_dialogue"])
        planned_content_dict[sample["id"]] = planned_dialogues_list
        
file_to_save = open("PersonaChat_dynamic_dialogue.json", "a")
keywords_mentioned_ratio_list = []
keywords_mentioned_order_distance_list = []
for dialogue_id in list(original_content_dict.keys()):
    key_word_list = keywords_content_dict[dialogue_id]
    hyps = planned_content_dict[dialogue_id]
    hyps_tokenized = [nltk.word_tokenize(hyp) for hyp in hyps]
    hyps_best_tokenized, idx = selectBest(hyps_tokenized)
    hyps_best = hyps[idx]
    conv_history = []
    done = False
    num_turns = 0
    while not done:
        num_turns += 1
        if num_turns>10: break
        messages = [{"role":"user", "content":"You are an intelligent chatbot with expertise in dialogue planning. Your task is to ensure that the conversation naturally incorporate a given list of keywords in the specified order. These keywords can be mentioned by either the user or the system, and should be seamlessly integrated into the dialogue flow. The keyword list is :{{{}}}. Your conversation must strictly follow this conversation plan:{{{}}}.\nHere is the conversation history: {{{}}}. If the dialogue history is empty, please generate a response to start the conversation. Now generate a succinct response (no longer than 30 words) for the next turn:".format(", ".join(key_word_list), hyps_best, "\n".join(conv_history))}]
        response = messagefunction(messages).strip()
        if "system" not in response.lower():
            conv_history.append("system: "+response)
        else:
            conv_history.append(response)
        messages = [{"role":"user", "content":"You are engaging in an open conversation with the system. Here is the conversation history: {{{}}}. Based on the dialogue history, generate a natural and relevant response (no longer than 30 words) for the next turn.".format("\n".join(conv_history))}]
        response = messagefunction(messages).strip()
        if "user" not in response.lower():
            conv_history.append("user: "+response)
        else:
            conv_history.append(response)
        if num_turns>5:
            messages = [{"role":"user", "content":"The user and the system are engaging in an open conversation. Here is the conversation history: {{{}}}. Please decide whether the conversation has incorporated a given list of keywords in the specified order. These keywords can be mentioned by either the user or the system. The keyword list is :{{{}}}. Answer yes only if all the keywords are mentioned. Please only answer yes or no.".format("\n".join(conv_history), ", ".join(key_word_list))}]
            response = messagefunction(messages).strip()
            done = ("yes" in response.lower())
    sample = {"id": dialogue_id, "key_word_list": key_word_list, "original_dialogue": original_content_dict[dialogue_id], "planned_dialogue": hyps_best}
    sample["num_turns"] = min(num_turns, 10)
    sample["history"] = conv_history

    keywords_mentioned = detect_keywords_in_sentence("\n".join(sample["history"]), sample["key_word_list"])
    keywords_mentioned_ratio = len(set(keywords_mentioned))/len(set(sample["key_word_list"]))
    keywords_mentioned_ratio_list.append(keywords_mentioned_ratio)
    keywords_mentioned_order_distance_list.append(nltk.edit_distance(keywords_mentioned, sample["key_word_list"]))
    file_to_save.write(json.dumps(sample)+"\n")
    file_to_save.flush()
f.close()
file_to_save.close()
print("keywords_mentioned_ratio: ", np.mean(np.array(keywords_mentioned_ratio_list)))
print("keywords_mentioned_order_distance: ", np.mean(np.array(keywords_mentioned_order_distance_list)))

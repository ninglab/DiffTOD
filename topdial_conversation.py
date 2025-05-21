import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import json
import os
import anthropic
from openai import OpenAI
import re

backend = "gpt4o"
os.environ['OPENAI_API_KEY'] = ''

def generate_response_claude(messages):
    client = anthropic.Anthropic(api_key='')
    response = client.messages.create(model="claude-3-5-sonnet-latest", messages=messages, max_tokens=256)
    return response.content[0].text.strip()
    
def generate_response_gpt4o(messages):
    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o", messages=messages)
    return completion.choices[0].message.content.strip()
if backend=="claude":
    messagefunction = generate_response_claude
elif backend=="gpt4o":
    messagefunction = generate_response_gpt4o

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

def get_prompt(user_profile, target):
    domain_desc = target.lower()
    target_item = target.split(", ", 1)[1].rstrip()
    print(target, target_item)
    target_action = domain_desc
    if "movie" in domain_desc:
        domain = "movie"
    elif "music" in domain_desc:
        domain = "music"
    elif "food" in domain_desc:
        domain = "food"
    elif "poi" in domain_desc:
        domain = "poi"
    if "movie" in domain_desc or "music" in domain_desc :
        env_desc = "You are participating in a conversation about music or movies."
    else:
        env_desc = "You are participating in a conversation about delicious food or point-of-interest (POI)."
    fields = user_profile.strip().split("\t")
    simulated_profile = {key.strip(): value.strip() for field in fields for key, value in [field.split(":", 1)]}
    user_desc = "You are {}, ".format(simulated_profile["Name"])
    
    if simulated_profile["Occupation"] == "Student":
        if simulated_profile["Gender"] == "Male":
            profile_desc = "a male student in the age range of {}, living in {}".format(simulated_profile["Age Range"].lower(), simulated_profile["Residence"])
        else:
            profile_desc = "a female student in the age range of {}, living in {}".format(simulated_profile["Age Range"].lower(), simulated_profile["Residence"])
    elif simulated_profile["Occupation"] == "Employed":
        if simulated_profile["Gender"] == "Male":
            profile_desc = "a man in the age range of {}, working in a company and living in {}".format(simulated_profile["Age Range"].lower(), simulated_profile["Residence"])
        else:
            profile_desc = "a woman in the age range of {}, working in a company and living in {}".format(simulated_profile["Age Range"].lower(), simulated_profile["Residence"])
    else:
        if simulated_profile["Gender"] == "Male":
            profile_desc = "a retired man in the age range of {}, living in {}".format(simulated_profile["Age Range"].lower(), simulated_profile["Residence"])
        else:
            profile_desc = "a retired woman in the age range of {}, living in {}".format(simulated_profile["Age Range"].lower(), simulated_profile["Residence"])
    user_desc += profile_desc + ".\n\n"
    
    user_desc += "Based on your past experiences, you have the following preferences:\n"
    if "movie" in domain_desc or "music" in domain_desc :
        for k in ["Accepted movies", "Accepted music", "Accepted celebrities", "Rejected movies", "Rejected music"]:
            kk = k.replace("Accepted", "preferred").replace("Rejected", "disliked")
            user_desc += "Your {}: {}.\n".format(kk, simulated_profile[k]) if simulated_profile.get(k, "") != "" else ""
    else:
        for k in ["Accepted food", "Accepted POI"]:
            kk = k.replace("Accepted", "preferred")
            user_desc += "Your {}: {}.\n".format(kk, simulated_profile[k]) if simulated_profile.get(k, "") != "" else ""
    user_desc += "\n"
    
    user_desc += "Your response should match your profile and personality, and be concise (no longer than 30 words).\n"
    user_desc += "You don't need to recommend anything, but feel free to express your personal interests."

    gender_desc = "his" if simulated_profile["Gender"] == "Male" else "her"
    if "movie" in domain_desc or "music" in domain_desc :
        for k in ["Accepted movies", "Accepted music", "Accepted celebrities", "Rejected movies", "Rejected music"]:
            kk = k.replace("Accepted", "preferred").replace("Rejected", "disliked")
            profile_desc += "; {} {}: {}".format(gender_desc, kk, simulated_profile[k]) if simulated_profile.get(k, "") != "" else ""
    else:
        for k in ["Accepted food", "Accepted POI"]:
            kk = k.replace("Accepted", "preferred")
            profile_desc += "; {} {}: {}".format(gender_desc, kk, simulated_profile[k]) if simulated_profile.get(k, "") != "" else ""
    profile_desc += "."
    
    if domain == "movie":
        assistant_desc = "You are a conversational recommender system that recommends films.\n"
    elif domain == "music":
        assistant_desc = "You are a conversational recommender system that recommends music.\n"
    elif domain == "food":
        assistant_desc = "You are a conversational recommender system that recommends food.\n"
    elif domain == "poi":
        assistant_desc = "You are a conversational recommender system that recommends restaurants.\n"
    
    assistant_desc += "You are conversing with {}, whose profile is below: \n## {}\n\n".format(simulated_profile["Name"], profile_desc)
    assistant_desc += "Your goal is to proactively lead the conversation with {} towards the target {} \"{}\".\n".format(simulated_profile["Name"], domain, target_item)
    assistant_desc += "To start the conversation, please begin with a greeting and avoid mentioning the target {}.\n".format(domain)
    assistant_desc += "As the conversation progresses, use your domain knowledge to steer the discussed topic towards the target {} step by step.\n".format(domain)
    assistant_desc += "Be informative and engaging while providing insights to arouse {}'s interest.\n".format(simulated_profile["Name"])
    assistant_desc += "Your words at each turn should be concise (no longer than 30 words).\n"    
    judge_desc = "Please decide whether the user {} has aceepted the target {} \"{}\" in the conversation. Please only respond with yes or no.\n".format(simulated_profile["Name"], domain, target_item)
    return user_desc, assistant_desc, judge_desc


success_list = []
avg_turn_list = []

f = open("topdial_generation.json", "r")
file_to_save = open("topdial_dynamic_dialogue.json", "a")
for line in f.readlines():
    sample =  json.loads(line.rstrip("\n"))
    user_profile = sample["user_profile"].replace("user profile: ", "")
    target = sample["target"].replace("target: ", "")
    user_desc, assistant_desc, judge_desc = get_prompt(user_profile, target)
    refs = sample["original_content"]
    hyps = sample["planned_dialogues"]
    refs_tokenized = nltk.word_tokenize(refs)
    hyps_tokenized = [nltk.word_tokenize(hyp) for hyp in hyps]
    hyps_best_tokenized, idx = selectBest(hyps_tokenized)
    num = len(hyps_tokenized)
    hyps_best_tokenized = hyps_tokenized[idx]
    hyps_best = hyps[idx]
    conv_history = []
    done = False
    num_turns = 0
    while not done:
        num_turns += 1
        if num_turns>10: break
        messages = [{"role":"user", "content":assistant_desc+"Your conversation must strictly follow this conversation plan:{{{}}}.\nHere is the conversation history: {{{}}}. If the dialogue history is empty, please generate a response to start the conversation. Now generate a succinct response for the next turn:".format(hyps_best, "\n".join(conv_history))}]
        response = messagefunction(messages).strip()
        if "system" not in response.lower():
            conv_history.append("system: "+response)
        else:
            conv_history.append(response)
        messages = [{"role":"user", "content":user_desc+"\nHere is the conversation history: {{{}}}. Now generate a succinct response for the next turn:".format("\n".join(conv_history))}]
        response = messagefunction(messages).strip()
        if "user" not in response.lower():
            conv_history.append("user: "+response)
        else:
            conv_history.append(response)
        messages = [{"role":"user", "content":judge_desc+"Here is the conversation history: {{{}}}".format("\n".join(conv_history))}]
        response = messagefunction(messages).strip()
        done = ("yes" in response.lower())
    sample["success"] = done
    sample["num_turns"] = min(num_turns, 10)
    success_list.append(int(done))
    avg_turn_list.append(num_turns)
    sample["history"] = conv_history
    file_to_save.write(json.dumps(sample)+"\n")
    file_to_save.flush()
f.close()
file_to_save.close()
print("success_rate: ", np.mean(np.array(success_list)))
print("average turn: ", np.mean(np.array(avg_turn_list)))

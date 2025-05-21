import torch
import argparse

from load_model import load_model, load_model_local
from transformers import GPT2TokenizerFast
import sampling
import pickle
import os
import json
import numpy as np
import anthropic
from openai import OpenAI
from mcts import Node
import random
import re

backend = "claude"
os.environ['OPENAI_API_KEY'] = ''

CBAct = {'greet': 'Please say hello or chat randomly.',
         'inquire': 'Please ask any question about product, year, price, usage, etc.',
         'inform': 'Please provide information about the product, year, usage, etc.',
         'propose': 'Please initiate a price or a price range for the product.',
         'counter': 'Please propose a new price or a new price range.',
         'counter-noprice': 'Please propose a vague price by using comparatives with existing price.',
         'confirm': 'Please ask a question about the information to be confirmed.',
         'affirm': 'Please give an affirmative response to a confirm.',
         'deny': 'Please give a negative response to a confirm.',
         'agree': 'Please agree with the proposed price.',
         'disagree': 'Please disagree with the proposed price.'}
candidate_actions = list(CBAct.keys())

#user: seller system: buyer

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

def evaluate_reward(dialogue_history):

    messages = [{"role":"user", "content":"Please decide whether the buyer and the seller have reached a deal at the end of the conversation. If they have reached a deal, please extract the deal price as [price]. You can only reply with one of the following sentences: They have reached a deal at [price]. They have not reached a deal.\nThe following is the conversation: buyer: Can we meet in the middle at $15?\nseller: Sure, let's meet at $15 for this high-quality balloon.\nQuestion: Have they reached a deal? Answer: They have reached a deal at $15.\n\nThe following is the conversation: buyer: That's still a bit high, can you go any lower?\nseller: Alright, I can sell it to you for $15.\nQuestion: Have they reached a deal? Answer: They have not reached a deal.\n\nThe following is the conversation: %s\nQuestion: Have they reached a deal? Answer: " % dialogue_history}]
    response = messagefunction(messages).strip()
    if 'have not' in response.lower():
        deals = -1
        sell_price = None
    elif 'have reached' in response.lower():
        deals = 1
        prices = re.findall(r"[-+]?\d*\.?\d+", response.replace(",",""))
        if prices:
            sell_price = prices[0]
        else:
            sell_price = None
    else:
        deals = -1
        sell_price = 0
    return deals, sell_price


    
def conditional_generation(model, graph, noise, tokenizer, steps, input_ids, input_locs):
    input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(1, 1)
    
    def proj_fun(x):
        x[:, input_locs] = input_ids
        return x
    
    device = torch.device('cuda')

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (1, 1024), 'analytic', steps, device=device, proj_fun=proj_fun
    )

    samples = proj_fun(sampling_fn(model))

    text_samples = tokenizer.batch_decode(samples)
    return text_samples[0]

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="exp_local/cb/...", type=str)
    parser.add_argument("--dataset_path", default="data/cb/", type=str)
    parser.add_argument("--file_save_name", default="cb_conversation_seller.json", type=str)
    parser.add_argument("--stage", default="test", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--max_conv_turn", default=10, type=int)
    parser.add_argument("--num_mcts_sims", default=10, type=int, help="number of simulations for mcts")
    parser.add_argument("--discount_factor", default=0.999, type=float)
    parser.add_argument("--w", default=1.5, type=float)
    parser.add_argument("--role", default="seller", type=str, help="the role the planner plays in the bargain")
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    dialog_file = open(os.path.join(args.dataset_path, "cb_test.txt"), "r")
    file_to_save = open(args.file_save_name, "a")
    
    
    input_ids_list = []
    input_locs_list = []
    
    
    device = torch.device('cuda')
    model, graph, noise = load_model_local(args.model_path, device)    
    dialogue_corpus = []
    
    for line in dialog_file.readlines():
        data = json.loads(line.strip())
        dialogue_corpus.append(data)
        
    dialog_file.close()

    turns_list = []
    success_list = []
    sell_to_list_ratio_seller_side_list = []
    for idx, dialog in enumerate(dialogue_corpus):
        dialog_history = ["buyer: Hi, how much is the %s?" % dialog['item_name']]
        dialog_strategy = []
        for turn in range(args.max_conv_turn):
            #decide seller action
            
            prefix = "item decription: %s\nseller price: %s\nbuyer price: %s\n"%(dialog['seller_item_description'][:200],dialog['buyer_price'],dialog['seller_price'])
            prefix_length = len(prefix)
            first_turn = ""
            for idx, utterance in enumerate(dialog_history):
                if utterance[:5].lower()=="buyer":
                    first_turn +=  "system: " + utterance[7:] + "\n"
                elif utterance[:6].lower()=="seller":
                    first_turn += "(strategy: %s) user: "%dialog_strategy[idx//2] + utterance[8:] + "\n"
                else:
                    first_turn += "user: " + utterance[8:] + "\n"
            first_utterance_length = len(first_turn)
            prefix += first_turn
            root = Node()
            root.children = dict.fromkeys(candidate_actions, None)
            root.is_root = True
            root.dialogue_history_so_far = prefix
            current_node = root
            num_simulations = 0
            dialog_strategy_mcts = dialog_strategy
            while num_simulations<args.num_mcts_sims:
                selected_action, selected_children = current_node.select(args.w)
                if selected_children is not None:
                    current_node = selected_children
                elif selected_children is None:
                    child = current_node.expand(selected_action)
                    guidance = current_node.dialogue_history_so_far + "(strategy: %s) user: "%selected_action
                    input_ids = tokenizer(guidance)['input_ids'][:1024]
                    input_locs = np.arange(len(input_ids))
                    #simulation with conditional generation
                    text = conditional_generation(model, graph, noise, tokenizer, args.steps, input_ids, input_locs).split("<|endoftext|>")[0]
                    text_without_prefix = text[prefix_length:]
                    system = 0
                    user = 0
                    child.dialogue_history_so_far = current_node.dialogue_history_so_far
                    for utterance in text[len(current_node.dialogue_history_so_far):].split("\n"):
                        if system and user:  
                            break
                        else:
                            child.dialogue_history_so_far += utterance.strip()+"\n"
                        if "system" in utterance.lower():
                            system+=1
                        if "user" in utterance.lower():
                            user+=1
                    temp = text_without_prefix.replace("user", "seller").replace("system", "buyer")
                    deals, sell_price = evaluate_reward(temp)
                    if deals==1 and sell_price is not None:
                        sell_to_list_ratio_seller_side = (-float(sell_price)+float(dialog["buyer_price"]))/(float(dialog["buyer_price"])-float(dialog["seller_price"]))
                        if sell_to_list_ratio_seller_side<0:
                            sell_to_list_ratio_seller_side = 0
                        if sell_to_list_ratio_seller_side>1:
                            sell_to_list_ratio_seller_side = 1
                    else:
                        sell_to_list_ratio_seller_side = -1
                    reward = sell_to_list_ratio_seller_side
                    if selected_action in ["agree", "affirm"] and turn>2: reward += 0.2
                    child.back_propagate(reward, args.discount_factor)
                    num_simulations+=1
                    current_node = root

            action, _ = root.select(0)  
            dialog_strategy.append(action)
            #seller utterance
            
            messages = [{"role":"user", "content":"Now enter the role-playing mode. In the following conversation, you will act as a seller negotiating to sell the %s for %s. Product description: %s\nRespond with a short, succinct and persuasive sentence aimed at securing the best possible deal. %s Now start the game. " % (dialog['item_name'], dialog['seller_price'], dialog['seller_item_description'], CBAct[action])+"\n".join(dialog_history)}]
            response = messagefunction(messages).strip()
            if response[:6].lower()!="seller":
                response = "seller: "+ response
            response = response[:6].lower() + response[6:]
            dialog_history.append(response)        
            #buyer utterance            
            messages = [{"role":"user", "content":"Now enter the role-playing mode. In the following conversation, you will act as a buyer negotiating to purchase the %s for %s. Product description: %s\nRespond with a short, succinct and persuasive sentence aimed at securing the best possible deal. Now start the game. " % (dialog['item_name'], dialog['buyer_price'], dialog['buyer_item_description'])+"\n".join(dialog_history)}]
            
            response = messagefunction(messages).strip()
            if response[:5].lower()!="buyer":
                response = "buyer: "+ response
            response = response[:5].lower() + response[5:]
            dialog_history.append(response)
            deals, sell_price = evaluate_reward("\n".join(dialog_history))
            if deals == 1:
                break
        #save_response_to_file
        dialog["dialog_history_generated"] = dialog_history
        dialog["turns"] = turn
        dialog["deals"] = deals
        dialog["sell_price"] = sell_price
        dialog["dialog_strategy"] = dialog_strategy
        if sell_price:
            sell_to_list_ratio_seller_side = (-float(dialog["sell_price"])+float(dialog["buyer_price"]))/(float(dialog["buyer_price"])-float(dialog["seller_price"]))
            sell_to_list_ratio_seller_side = max(0, min(sell_to_list_ratio_seller_side, 1))
            sell_to_list_ratio_seller_side_list.append(sell_to_list_ratio_seller_side)
        else:
            sell_to_list_ratio_seller_side_list.append(0)
        turns_list.append(turn)
        success_list.append((deals+1)/2)
        file_to_save.write(json.dumps(dialog)+"\n")
        file_to_save.flush()

    file_to_save.close()
    print("Avg Turn:%.6f. Success rate:%.6f%%. sell_to_list_ratio_seller_side:%.6f%%"%(np.mean(turns_list),np.mean(success_list)*100, np.mean(sell_to_list_ratio_seller_side)*100))
            


if __name__=="__main__":
    main()

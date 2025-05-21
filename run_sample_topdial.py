import torch
import argparse

from load_model import load_model, load_model_local
from transformers import GPT2TokenizerFast
import sampling
import pickle
import os
import json
import numpy as np

def locate_semantics_in_sentence(tokenizer, semantic_guidance_sentence):
    semantic_guidance_sentence_tokenized = tokenizer("system: " +semantic_guidance_sentence)['input_ids']
    return semantic_guidance_sentence_tokenized, np.arange(192, 192+len(semantic_guidance_sentence_tokenized))

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="./exp_local/TopDial/...", type=str)
    parser.add_argument("--dataset_path", default="./data/TopDial/", type=str)
    parser.add_argument("--file_save_name", default="./topdial_generation.json", type=str)
    parser.add_argument("--stage", default="test", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    dialog_file = open(os.path.join(args.dataset_path, "dialogue_test_unseen.jsonl"))
    semantic_guidance_file = open(os.path.join(args.dataset_path, "dialogue_test_unseen_semantic_guidance.json"))
    
    
    input_ids_list = []
    input_locs_list = []
    device = torch.device('cuda')
    model, graph, noise = load_model_local(args.model_path, device)
    
    semantics_guidance_dict = {}
    
    for line in semantic_guidance_file.readlines():
        data = json.loads(line)
        idx = data["id"]
        target_semantic_guidance = data["target_semantic_guidance"]
        semantics_guidance_dict[idx] = target_semantic_guidance
    
    semantic_guidance_file.close()
    file_to_save = open(args.file_save_name, "w")
    for line in dialog_file.readlines():
        dialog = json.loads(line.strip())
        idx = dialog["id"]
        target = dialog['target']
        dialog_content = ""
        for utterance in dialog['conversation']:
            if "user" in utterance.keys():
                prefix = 'user: '
                key = 'user'
            elif "system" in utterance.keys():
                prefix = 'system: '
                key = 'system'
            dialog_content += prefix
            dialog_content += utterance[key]
            dialog_content += "\n"
        user_profile = "user profile: "
        for k, v in dialog["user_profile"].items():
            user_profile += "%s: %s\t" % (k, v)
        target_text = "target: %s, %s\n"%(target[0], target[1])
        data_dict = {"id": idx, "user_profile": user_profile, "target": target_text, "original_content": dialog_content, "planned_dialogues": []}
        guidance = semantics_guidance_dict[idx]
        for guidance_idx, semantics_guidance in enumerate(guidance):
            input_ids, input_locs = locate_semantics_in_sentence(tokenizer, semantics_guidance)
            user_profile_ids = tokenizer(user_profile)['input_ids']
            length = len(user_profile_ids)
            user_profile_locs = np.arange(length)
            input_ids = user_profile_ids + input_ids
            input_locs = input_locs + length
            input_locs = np.concatenate((user_profile_locs, input_locs))
            input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)
            def proj_fun(x):
                x[:, input_locs] = input_ids
                return x
            sampling_fn = sampling.get_pc_sampler(graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device, proj_fun=proj_fun)
            samples = proj_fun(sampling_fn(model))
            text_samples = tokenizer.batch_decode(samples)
            for i in text_samples:
                generated_text = i[len(user_profile)+len(target_text):].split("<|endoftext|>")
                if len(generated_text) == 1:
                    text = generated_text[0]
                else:
                    if "user profile:" in generated_text[1]:
                        text = generated_text[0]
                    else:
                        text = generated_text[0]+generated_text[1]
                data_dict["planned_dialogues"].append(text)
        json_string = json.dumps(data_dict)+"\n"
        
        file_to_save.write(json_string)
        file_to_save.flush()        
    dialog_file.close()
    file_to_save.close()
if __name__=="__main__":
    main()

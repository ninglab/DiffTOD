import torch
import argparse

from load_model import load_model, load_model_local
from transformers import GPT2TokenizerFast
import sampling
import pickle
import json
import itertools
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="./exp_local/PersonaChat/...", type=str)
    parser.add_argument("--dataset_path", default="./data/PersonaChat/source_data.pk", type=str)
    parser.add_argument("--file_to_save", default="./PersonaChat_test/PersonaChat_test_1.json", type=str)
    parser.add_argument("--stage", default="test", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    data = pickle.load(open(args.dataset_path,"rb"))
    device = torch.device('cuda')
    model, graph, noise = load_model_local(args.model_path, device)
    file_to_save = open(args.file_to_save, "w")
    for index, sample in enumerate(data[args.stage]):
        num_turns =  len(sample['dialog'])
        dialog_content = ""
        kw_list = list(itertools.chain(*sample['kwlist']))
        input_ids = []
        input_locs = []
        length_so_far = 0
        
        instructions="You are an intelligent chatbot with expertise in dialogue planning. Your task is to generate a dialogue plan that follows the list of keywords. The keywords from the provided list must be mentioned in the exact order. Ordered keywords list: %s."%", ".join(kw_list)
        
        tokenized_input = tokenizer(instructions)
        input_ids = np.array(tokenized_input['input_ids'])
        instructions_input_id_length = len(input_ids)
        input_locs = np.arange(instructions_input_id_length)
        current_length = instructions_input_id_length
        input_ids_list = [input_ids]
        input_locs_list = [input_locs]
        dialog_content = ""
        for turn in range(num_turns):
            if turn%2==0:
                prefix = 'system'
            else:
                prefix = 'user'
            if sample['kwlist'][turn]:
                guidance = "\n(keywords to mention: %s) " % ", ".join(sample['kwlist'][turn]) + prefix
            else:
                guidance = "\n"+prefix
            dialog_content += guidance+": "+sample["dialog"][turn]
            input_ids = np.array(tokenizer(guidance)['input_ids'])
            length = len(input_ids)
            input_locs = np.arange(current_length, current_length + length)
            current_length += length + 10 + len(sample['kwlist'][turn])*2
            input_ids_list.append(input_ids)
            input_locs_list.append(input_locs)
        input_ids = np.concatenate(input_ids_list)
        input_locs = np.concatenate(input_locs_list)

        input_ids = torch.tensor(input_ids, device="cuda")[None].repeat(args.batch_size, 1)
        data_dict = {"id": index, "key_word_list": kw_list, "original_dialogue": dialog_content}
        def proj_fun(x):
            x[:, input_locs] = input_ids
            return x
        sampling_fn = sampling.get_pc_sampler(graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device, proj_fun=proj_fun)
        samples = proj_fun(sampling_fn(model))
        text_samples = tokenizer.batch_decode(samples)
        for i in text_samples:
            data = i[len(instructions):].split("<|endoftext|>")
            if len(data[0].split("\n"))>6:
                planned_dialogue = data[0]
            else:
                planned_dialogue = data[0]+data[1]
        print(planned_dialogue)
        data_dict["planned_dialogue"] = planned_dialogue
        json_string = json.dumps(data_dict)+"\n"
        file_to_save.write(json_string)
        file_to_save.flush()
    file_to_save.close()

if __name__=="__main__":
    main()

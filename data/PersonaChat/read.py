
from collections import Counter
import pickle
import random
import os
import itertools
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
data = pickle.load(open("source_data.pk","rb"))
for stage in ['train', 'valid', 'test']:
    file = open("PersonaChat_{}.json".format(stage), "w")
    for sample in data[stage]:
        current_dialogue_length = len(sample['dialog'])
        dialogcontent="You are an intelligent chatbot with expertise in dialogue planning. Your task is to generate a dialogue plan that follows the list of keywords. The keywords from the provided list must be mentioned in the exact order. Ordered keywords list: %s.\n"%", ".join(list(itertools.chain(*sample['kwlist'])))
        for index in range(current_dialogue_length):
            if index%2==0:
                prefix = 'system: '
            else:
                prefix = 'user: '
            if sample['kwlist'][index]:
                dialogcontent += "(keywords to mention: %s) " % ", ".join(sample['kwlist'][index]) + prefix
            else:
                dialogcontent += prefix
            dialogcontent += sample['dialog'][index]+"\\n"
        dialogcontent = dialogcontent.replace('\n', '\\n')
        
        file.write('{"text":"'+dialogcontent+'"'+"}\n")
    file.close()           
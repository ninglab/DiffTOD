from openai import OpenAI
import json
def get_message(target_1, target_2):
    client = OpenAI()
    completion = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": 'You are an intelligent conversational recommender system. Your task is to recommend a target item (e.g., movie, food) in a succinct, helpful, and engaging way. The target action is "%s" for "%s". Write a very brief, helpful and appropriate message that encourages the user to perform the target action. Give 5 possible and different versions of such message separated with \n. Do not give any number or index at the beginning of each message.'%(target_1, target_2)}])
    return completion.choices[0].message.content
f = open("dialogue_test_unseen.jsonl", "r", encoding = 'utf-8')
file = open("dialogue_test_unseen_semantic_guidance.json", "w", encoding = 'utf-8')
for line in f.readlines():
    json.loads(line)
    dialog = json.loads(line.strip())
    idx = dialog["id"]
    target = dialog["target"]
    responses = get_message(target[0], target[1])
    new_list = []
    for response in responses.split("\n"):
        if len(response)>10:
            new_list.append(response)
    data = {"id": idx, "target_semantic_guidance": new_list}
    json_string = json.dumps(data)+"\n"
    file.write(json_string)
    file.flush()
f.close()
file.close()



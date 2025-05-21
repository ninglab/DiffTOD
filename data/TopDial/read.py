import json
f = open("dialogue_dev.jsonl", "r", encoding = 'utf-8')
file = open("TopDial_valid.json", "w", encoding = 'utf-8')
for line in f.readlines():
    json.loads(line)
    dialog = json.loads(line.strip())
    user_profile = "user profile: "
    for k, v in dialog["user_profile"].items():
        user_profile += "%s: %s\t" % (k, v)
    #print(dialog['conversation'])
    target = dialog['target']
    #print(target)
    target = "target: %s, %s"%(target[0], target[1])
    dialogcontent = user_profile + target + "\n"
    for utterance in dialog['conversation']:
        if "user" in utterance.keys():
            prefix = 'user: '
            key = 'user'
        elif "system" in utterance.keys():
            prefix = 'system: '
            key = 'system'
        dialogcontent += prefix
        dialogcontent += utterance[key]
        dialogcontent += "\n"
    #dialogcontent = dialogcontent.replace('\n', '\\n')
    #dialogcontent = dialogcontent.replace('\t', '\\t')
    #dialogcontent = dialogcontent.replace('"', '\\"')
    data = {"text": dialogcontent}
    json_string = json.dumps(data)+"\n"
    file.write(json_string)
f.close()
file.close()


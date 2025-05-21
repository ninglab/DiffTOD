import json
f = open("cb_test.txt", "r", encoding = 'utf-8')
file = open("cb_test.json", "w", encoding = 'utf-8')
for line in f.readlines():
    json.loads(line)
    dialog = json.loads(line.strip())
    dialogcontent = "item decription: %s\nseller price: %s\nbuyer price: %s\n"%(dialog['seller_item_description'],dialog['buyer_price'],dialog['seller_price'])
    for utterance in dialog['dialog']:
        dialogcontent+= "(strategy: %s) "%utterance['strategy']
        if utterance['speaker']=='sys':
            dialogcontent+="system: "
        elif utterance['speaker']=='usr':
            dialogcontent+="user: "
        if dialogcontent[-1]!="\n":
            dialogcontent+= utterance['text']+"\n"            
    data = {"text": dialogcontent}
    json_string = json.dumps(data)+"\n"
    file.write(json_string)
f.close()
file.close()

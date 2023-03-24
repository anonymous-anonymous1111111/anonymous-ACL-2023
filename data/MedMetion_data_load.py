import os
import nltk
import nltk.data

def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences

file_name = "corpus_pubtator.txt"

train_label = {'T038':"Biologic Function",'T103':'Chemical','T058':"Health Care Activity",
            'T017':'Anatomical Structure','T033':'Finding','T082':'Spatial Concept',
            'T170':'Intellectual Product','T062':'Research Activity',
            'T204':'Eukaryote','T098':'Population Group','T074':'Medical Device'}
dev_label = {'T092':'Organization','T037':'Injury or Poisoning','T201':'Clinical Attribute',
            'T005':'Virus','T091':'Biomedical Occupation or Discipline'}

test_label = {'T007':'Bacterium','T097':'Professional or Occupational Group','T168':'Food',
              'T031':'Body Substance','T022':'Body System'}




train_data = []
dev_data = []
test_data = []

train_write = open('train.data',"w",encoding="utf8")
dev_write = open('dev.data',"w",encoding="utf8")
test_write = open('test.data',"w",encoding="utf8")


with open(file_name,"r",encoding="utf8") as r:
    title = ""
    entity_dic = {}
    text = ""
    for line in r.readlines():
        if not line.strip():
            sentence_list = splitSentence(text)
            sentence_list.append(title)
            for s in sentence_list:
                s = s.replace("."," .")
                s = s.replace(","," ,")
                s = s.replace("(","( ")
                s = s.replace(")"," )")
                s = s.replace(":"," :")
                entity_type_dic = {}
                for entity in entity_dic:
                    if entity in s:
                        if entity_dic[entity] not in entity_type_dic:
                            entity_type_dic[entity_dic[entity]] = [entity]
                        else:
                            entity_type_dic[entity_dic[entity]] += [entity]
                for type in entity_type_dic:
                    if type in train_label:
                        train_write.write(s+"\t"+train_label[type]+"\t"+",".join(entity_type_dic[type])+"\n")
                        train_data.append(s+"\t"+type+"\t"+",".join(entity_type_dic[type]))
                    elif type in dev_label:
                        dev_write.write(s+"\t"+dev_label[type]+"\t"+",".join(entity_type_dic[type])+"\n")
                        dev_data.append(s+"\t"+type+"\t"+",".join(entity_type_dic[type]))
                    elif type in test_label:
                        test_write.write(s+"\t"+test_label[type]+"\t"+",".join(entity_type_dic[type])+"\n")
                        test_data.append(s+"\t"+type+"\t"+",".join(entity_type_dic[type]))
                    else:
                        continue
            entity_dic = {}
            continue
        if "|t|" in line.strip():
            _,title = line.strip().split("|t|")
        elif "|a|" in line.strip():
            _,text = line.strip().split("|a|")
        else:
            _,_,_,en,en_typ,_ = line.strip().split("\t")
            entity_dic[en] = en_typ



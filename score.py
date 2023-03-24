from data_process import Electra_Discriminator_DataSet_RC,OntoNote_Dataset,MedMention_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn as nn
import copy
import torch
from data_process import NER_Dataset_For_Zero_Relation_Extraction
from data_process import  NER_data_get_For_Zero_Relation_Extraction,NER_For_RE_DataSet,RC_For_RE_DataSet,multi_candidate_RC_For_RE_DataSet
from transformers import AutoTokenizer
import numpy as np




# Test Zero-Shot ner model
def NER_predict(electra_model,Device,config,label_type=None,test_label=None):
    model = electra_model.to(Device)
    if config["data_type"] == "medmention":
        dataset = MedMention_Dataset(config["test_data_path"],config["pretrain_model_path"],test_label,predict=True,label_type=label_type)
    else:
        dataset = OntoNote_Dataset(config["test_data_path"], config["pretrain_model_path"], test_label,
                                         predict=True, label_type=label_type)
    testdata = DataLoader(dataset, shuffle=False, batch_size=config["batch_size"])
    span_all_pred = []
    span_real_label = []

    token_all_pred = []
    token_real_label = []

    model.eval()
    sigmoid = nn.Sigmoid()
    for batch_id,x in enumerate(testdata):
        ids, attention_mask, position_ids, label, label_map, span_label = x[0].to(Device), x[1].to(Device), x[2].to(Device), x[3].to(Device), x[4].to(Device), x[5].to(Device)
        pred = model(input_ids=ids,attention_mask=attention_mask,position_ids=position_ids,label_map=label_map)
        pred = torch.round(sigmoid(pred))
        pred = pred.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()
        label_map = label_map.detach().cpu().numpy().tolist()
        span_label = span_label.detach().cpu().numpy().tolist()
        assert len(pred) == len(label)

        # test score
        for b in range(len(label)):
            start_pos_pred = []
            end_pos_pred = []
            for number,x  in enumerate(zip(label_map[b],label[b],pred[b])):
                lm, l, p = x[0],x[1],x[2]
                if lm==1:
                    if number<config["max_length"]-dataset.soft_prompts_length:
                        start_pos_pred.append(p)
                    else:
                        end_pos_pred.append(p)
                    if l ==0  and p == 0:
                        token_all_pred.append(0)
                        token_real_label.append(0)
                    elif l==1 and p==0:
                        token_all_pred.append(0)
                        token_real_label.append(1)
                    elif  l==0 and p==1:
                        token_all_pred.append(1)
                        token_real_label.append(0)
                    else:
                        continue

            assert len(start_pos_pred) == len(end_pos_pred)

            tmp_span_level = copy.copy(span_label[b])
            #delete pad label
            while tmp_span_level and tmp_span_level[-1] == [151,151]:
                tmp_span_level.pop()
            tmp_pred_span = []
            start_index = []
            end_index = []
            for index,x in enumerate(zip(start_pos_pred,end_pos_pred)):
                start, end = x[0],x[1]
                if start==0:
                    start_index.append(index)
                if end==0:
                    end_index.append(index)

            for s in start_index:
                for e in end_index:
                    if e>=s:
                        tmp_pred_span.append([s,e])
                        break

            for tmp in tmp_pred_span:
                if tmp in tmp_span_level:  # predict true,real is true
                    span_all_pred.append(0)
                    span_real_label.append(0)
                    tmp_span_level.remove(tmp)
                else:
                    span_all_pred.append(0)  # predict true,real is false
                    span_real_label.append(1)

            #predict fasle, real is true
            span_all_pred += [1]*len(tmp_span_level)
            span_real_label += [0]*len(tmp_span_level)


    return precision_score(span_real_label, span_all_pred, pos_label=0),recall_score(span_real_label, span_all_pred, pos_label=0),f1_score(span_real_label, span_all_pred, pos_label=0)




def RC_predict(electra_model,test_data_path,config,Device):
    model = electra_model.to(Device)
    dataset = Electra_Discriminator_DataSet_RC(data_path=test_data_path,pretrain_model_path=config["pretrain_model_path"], negtive_sample_rate=config["negtive_sample_ratio"],predict=True)
    testdata = DataLoader(dataset, shuffle=False, batch_size=config["batch_size"])
    all_pred = []
    real_label = []
    model.eval()
    for batch_id,x in tqdm(enumerate(testdata)):
        ids, attention_mask, type, label, label_position, label_map = x[0].to(Device), x[1].to(Device), x[2].to(Device), x[3].to(Device), x[4].to(Device), x[5].to(Device)
        pred = model(input_ids=ids,attention_mask=attention_mask,token_type_ids=type,label_map=label_map)
        pred = pred.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()
        assert len(pred) == len(label)
        for num in range(len(pred)):
            if sum(pred[num])>0:
                all_pred.append(1)
            else:
                all_pred.append(0)
            if sum(label[num]) > 0:
                real_label.append(1)
            else:
                real_label.append(0)


    print('acc:', accuracy_score(real_label, all_pred))
    print('precision:', precision_score(real_label, all_pred, pos_label=0))
    print('recall:', recall_score(real_label, all_pred, pos_label=0))
    print('f1:', f1_score(real_label, all_pred, pos_label=0))
    return f1_score(real_label, all_pred, pos_label=0)


def macro_score(model,config,label,Device):
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    print("RUN TEST WAIT !!!")
    for e in label:
        p,r,f = NER_predict(model, Device,config ,test_label=label,label_type=e)
        precision += p
        recall += r
        f1 += f

    print("precision score is:", precision / len(label))
    print("recall score is:", recall / len(label))
    print("f1 score is:",f1/len(label))




def get_positive_position(lis):
    position = []
    start = 0
    end = 0
    flag = 0
    for i in range(len(lis)):
        if lis[i] == 1 and flag == 0 and i < len(lis)-1:
            start = i
            flag = 1
            continue
        if lis[i] == 0 and flag == 1:
            end = i-1
            position.append((start,end))
            start = 0
            end = 0
            flag = 0
            continue
        if i == len(lis)-1:
            if lis[i] == 1 and flag == 1:
                end = i
                position.append((start, end))
                start = 0
                end = 0
                flag = 0
            elif lis[i] == 1 and flag == 0:
                start = i
                end = i
                position.append((start, end))
    return position

def NER_For_RE_predict(electra_model,Device,config):

    model = electra_model.to(Device)
    dataset = NER_Dataset_For_Zero_Relation_Extraction(config["dev_data_path"],config["pretrain_model_path"])
    testdata = DataLoader(dataset, shuffle=False, batch_size=config["batch_size"])
    all_pred = []
    real_label = []
    model.eval()
    sigmoid = nn.Sigmoid()
    for batch_id,x in tqdm(enumerate(testdata)):
        ids, attention_mask, type, label, label_map = x[0].to(Device), x[1].to(Device), x[2].to(Device), x[3].to(Device), x[4].to(Device)
        pred = model(input_ids=ids,attention_mask=attention_mask,token_type_ids=type,label_map=label_map)

        pred = torch.round(sigmoid(pred))
        pred = pred.detach().cpu().numpy().tolist()
        label = label.detach().cpu().numpy().tolist()
        label_map = label_map.detach().cpu().numpy().tolist()
        assert len(pred) == len(label)

        #span test
        for i in range(len(label)):
            label_length = sum(label_map[i])
            all_pred += pred[i][1:label_length+1]
            real_label += label[i][1:label_length+1]

            pred_p = get_positive_position(pred[i][1:label_length+1])
            label_p = get_positive_position(label[i][1:label_length+1])
            all_one = pred_p+label_p
            for i in all_one:
                if i in pred_p and i in label_p:
                    all_pred.append(0)
                    real_label.append(0)
                elif i in pred_p:
                    all_pred.append(0)
                    real_label.append(1)
                elif i in label_p:
                    all_pred.append(1)
                    real_label.append(0)
                else:
                    print("error!!!")
                    return ReferenceError


    print('acc:', accuracy_score(real_label, all_pred))
    print('precision:', precision_score(real_label, all_pred, pos_label=0))
    print('recall:', recall_score(real_label, all_pred, pos_label=0))
    print('f1:', f1_score(real_label, all_pred,pos_label=0))
    return precision_score(real_label, all_pred, pos_label=0),recall_score(real_label, all_pred, pos_label=0),f1_score(real_label, all_pred,pos_label=0 )

def multi_pipline_relation_extraction(path,pretrain_path,NER_model,relation_model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dataset = NER_data_get_For_Zero_Relation_Extraction(path)
    triplets, all_labels = raw_dataset.get_relation_entities()

    sigmoid = nn.Sigmoid()
    maxlength = 150
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    all_real_triplet = 0
    pred_triplet = 0
    acc_pred_triplet = 0
    pred_false = 0

    entity_predict = 0
    for line in tqdm(triplets):
        text = line["text"]
        head_entity = line["head_entity"]
        tail_entity = line["tail_entity"]
        relation = line["relation"]

        ner_text_tokens, ids, types, masks, label_map = NER_For_RE_DataSet(line, tokenizer)
        sub_preds = NER_model(input_ids=torch.LongTensor([ids]).to(device),
                              token_type_ids=torch.LongTensor([types]).to(device),
                              attention_mask=torch.LongTensor([masks]).to(device),
                              label_map=torch.LongTensor([label_map]).to(device))

        pred = torch.round(sigmoid(sub_preds))
        pred = pred.detach().cpu().numpy().tolist()

        label_length = sum(label_map)
        pred_p = get_positive_position(pred[0][1:label_length + 1])

        tmp_print = []
        for i in pred_p:
            tmp_print.append(ner_text_tokens[i[0]:i[1] + 1])

        head_tokens = []
        tail_tokens = []
        combine_e = []
        for h_e, t_e in zip(head_entity.split("//t"), tail_entity.split("//t")):
            head_tokens.append(tokenizer.tokenize(h_e))
            tail_tokens.append(tokenizer.tokenize(t_e))
            combine_e.append(tokenizer.tokenize(h_e) + tokenizer.tokenize(t_e))



        if len(tmp_print)>1:
            entity_predict += 1
            pred_relation = []
            for h in tmp_print:
                for t in tmp_print:
                    if h==t or (h in pred_relation and t in pred_relation):
                        continue

                    text_template_ids,text_template_masks,text_template_types,labelmap,rel_maps = multi_candidate_RC_For_RE_DataSet(text,tokenizer,head_entity,tail_entity,all_labels)

                    pred = relation_model(input_ids=torch.LongTensor([text_template_ids]).to(device), attention_mask=torch.LongTensor([text_template_masks]).to(device),
                                          token_type_ids=torch.LongTensor([text_template_types]).to(device), label_map=torch.LongTensor([labelmap]).to(device))

                    pred = pred.detach().cpu().numpy().tolist()
                    for num in range(len(pred)):
                        for rel,rel_map in zip(all_labels,rel_maps):
                            if sum(np.multiply(np.array(rel_map), np.array(pred[num])).tolist()) > 0:
                                pred_false += 1
                            else:
                                if h+t in combine_e and rel == tokenizer.tokenize(relation):
                                    pred_relation.append(h)
                                    pred_relation.append(t)
                                    acc_pred_triplet += 1
                                pred_triplet += 1

        all_real_triplet += 1
    P = acc_pred_triplet/pred_triplet
    R = acc_pred_triplet/all_real_triplet
    F1 = 2*P*R/(P+R)
    print("Precision: ", P)
    print("Recall: ", R)
    print("F1 score: ", F1)


#Test Zero-Shot relation extraction model
def pipline_relation_extraction(data_path,pretrain_path,NER_model,relation_model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dataset = NER_data_get_For_Zero_Relation_Extraction(data_path)
    triplets, all_labels = raw_dataset.get_relation_entities()

    sigmoid = nn.Sigmoid()
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    all_real_triplet = 0
    pred_triplet = 0
    acc_pred_triplet = 0
    pred_false = 0

    entity_predict = 0
    for line in tqdm(triplets):
        text = line["text"]
        head_entity = line["head_entity"]
        tail_entity = line["tail_entity"]
        relation = line["relation"]

        ner_text_tokens,ids,types,masks,label_map = NER_For_RE_DataSet(line,tokenizer)
        sub_preds = NER_model(input_ids=torch.LongTensor([ids]).to(device),
                              token_type_ids=torch.LongTensor([types]).to(device),
                              attention_mask=torch.LongTensor([masks]).to(device),
                              label_map=torch.LongTensor([label_map]).to(device))

        pred = torch.round(sigmoid(sub_preds))
        pred = pred.detach().cpu().numpy().tolist()

        label_length = sum(label_map)
        pred_p = get_positive_position(pred[0][1:label_length + 1])

        tmp_print = []
        for i in pred_p:
            tmp_print.append(ner_text_tokens[i[0]:i[1]+1])

        head_tokens = []
        tail_tokens = []
        combine_e = []
        for h_e,t_e in zip(head_entity.split("//t"),tail_entity.split("//t")):
            head_tokens.append(tokenizer.tokenize(h_e))
            tail_tokens.append(tokenizer.tokenize(t_e))
            combine_e.append(tokenizer.tokenize(h_e)+tokenizer.tokenize(t_e))


        if len(tmp_print)>1:
            entity_predict += 1
            pred_relation = []
            for h in tmp_print:
                for t in tmp_print:
                    if h==t or (h in pred_relation and t in pred_relation):
                        continue
                    for l in all_labels:

                        text_template_ids,text_template_masks,text_template_types,labelmap = RC_For_RE_DataSet(text,tokenizer,h,t,l)
                        rel = tokenizer.tokenize(l)
                        pred = relation_model(input_ids=torch.LongTensor([text_template_ids]).to(device), attention_mask=torch.LongTensor([text_template_masks]).to(device),
                                              token_type_ids=torch.LongTensor([text_template_types]).to(device), label_map=torch.LongTensor([labelmap]).to(device))

                        pred = pred.detach().cpu().numpy().tolist()
                        for num in range(len(pred)):
                            if sum(pred[num]) > 0:
                                pred_false += 1
                            else:
                                if h+t in combine_e and rel == tokenizer.tokenize(relation):
                                    pred_relation.append(h)
                                    pred_relation.append(t)
                                    acc_pred_triplet += 1
                                pred_triplet += 1

        all_real_triplet += 1
    P = acc_pred_triplet/pred_triplet
    R = acc_pred_triplet/all_real_triplet
    F1 = 2*P*R/(P+R)
    print("Precision: ", P)
    print("Recall: ", R)
    print("F1 score: ", F1)


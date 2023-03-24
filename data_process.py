from  process_by_ZERO_RTE import ZeroRTE_Dataset
from typing import List
from transformers import ElectraTokenizer,AutoTokenizer
import numpy as np
import torch
from data.tag_instantiation import MedMention_tag_instantiation,OntoNotes_tag_instantiation
import random
import copy


random.seed(2022)
class data_get_RC:
    def __init__(self,data_path):
        self.data = ZeroRTE_Dataset.load(data_path)
        self.triplets = []
        self.all_labels = []
    def list_to_str(self,tokens) -> str:
        return " ".join(tokens)

    def get_entities(self):
        return NotImplementedError

    def get_relation_entities(self):
        for s in self.data.sents:
            for t in s.triplets:
                if not t.head or  not t.tail:
                    continue
                triplet = {}
                triplet["text"] = self.list_to_str(t.tokens)
                triplet["head_entity"] = self.list_to_str(t.tokens[t.head[0]:t.head[-1]+1])
                triplet["tail_entity"] = self.list_to_str(t.tokens[t.tail[0]:t.tail[-1]+1])
                triplet["relation"] = t.label
                if t.label not in self.all_labels:
                    self.all_labels.append(t.label)
                self.triplets.append(triplet)
        return self.triplets, self.all_labels






def load_data_NER_Soft_Prompts(path):
    text_list = []
    entity_list = []
    entity_type_list = []
    with open(path, "r", encoding="utf8") as r:
        for line in r.readlines():
            text,entity_type,entity = line.strip().split("\t")
            text_list.append(text)
            entity_list.append(entity)
            entity_type_list.append(entity_type)
    return text_list,entity_list,entity_type_list




class OntoNote_Dataset:
    def __init__(self,path,pretrain_path,all_label,predict=False,label_type=None):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        text_list,entity_list,entity_type_list = load_data_NER_Soft_Prompts(path)
        self.all_text_id = []
        self.attention_masks = []
        self.position_ids = []
        self.span_labels = []
        self.labels = []
        self.label_maps = []
        self.length_pure_texts = []
        self.len_text_tokens = []

        self.maxlength = 512
        self.maxtext = 150
        self.soft_prompts_length = 140
        self.all_label = all_label
        self.predict = predict
        self.all = True
        self.tmp = {}
        if predict:
            for text,entity,entity_type in zip(text_list,entity_list,entity_type_list):
                if len(self.tokenizer.tokenize(text)) > self.maxtext-5:
                    continue

                self.get_sample(text, entity, entity_type)

        else:
            for text,entity,entity_type in zip(text_list,entity_list,entity_type_list):
                if len(self.tokenizer.tokenize(text)) > self.maxtext-5:
                    continue
                self.get_sample(text,entity,entity_type)

        self.len = len(self.all_text_id)

    def get_sample(self, text, entity, entity_type):

        start_pos = []
        end_pos = []
        tag_instantiation = OntoNotes_tag_instantiation()
        entity_type = tag_instantiation[entity_type]
        entity_type_token = self.tokenizer.tokenize(entity_type)
        entity_type_id = self.tokenizer.convert_tokens_to_ids(entity_type_token)

        entity_position = []
        entity_list = entity.split(",,,")

        token_list = []
        text_token = []

        first_start_soft_prompt_position = self.maxlength - 2 * self.soft_prompts_length
        first_end_soft_prompt_position = self.maxlength -  self.soft_prompts_length
        for word in text.strip().split(" "):
            start = len(text_token)+1
            text_token += self.tokenizer.tokenize(word)
            end = len(text_token)
            start_pos.append(start)
            end_pos.append(end)
            token_list.append(word)


        real_label = [1] * self.maxlength


        last_entity_position = 0
        for e in entity_list:
            if e not in self.tmp:
                self.tmp[e] = 1
            else:
               self.tmp[e] += 1

            e = e.strip().split(" ")
            e_start = self.search(e,token_list,last_entity_position)
            if e_start == -1:
                return
            last_entity_position = e_start+len(e)-1
            entity_position.append((e_start,last_entity_position))
            real_label[first_start_soft_prompt_position+e_start] = 0
            real_label[first_end_soft_prompt_position+last_entity_position] = 0

        while len(entity_position)<20:
            entity_position.append((151,151))
        while len(entity_position)>20:
            entity_position.pop()
        assert len(entity_position)==20

        text_token = ["[CLS]"] + text_token + ["[SEP]"]

        text_id = self.tokenizer.convert_tokens_to_ids(text_token)
        text_id += entity_type_id
        length_pure_text = len(text_id)

        position_id = list(range(length_pure_text))
        position_id = self.pad_id(position_id, self.maxlength - 2 * self.soft_prompts_length)
        text_id = self.pad_id(text_id,self.maxlength-2*self.soft_prompts_length)

        start_id = [1]*len(start_pos)
        start_id = self.pad_id(start_id,self.soft_prompts_length)
        text_id += start_id

        end_id = [2]*len(end_pos)
        end_id = self.pad_id(end_id, self.soft_prompts_length)
        text_id += end_id
        assert len(text_id) ==self.maxlength


        position_id += start_pos
        position_id = self.pad_id(position_id, self.maxlength-self.soft_prompts_length)
        position_id += end_pos
        position_id = self.pad_id(position_id, self.maxlength)
        assert len(start_pos) == len(end_pos)

        self.all_text_id.append(text_id)
        self.position_ids.append(position_id)
        self.labels.append(real_label)
        self.span_labels.append(entity_position)
        self.length_pure_texts.append(length_pure_text)
        self.len_text_tokens.append(len(start_pos))


    def get_neg_sample(self, text, entity_type):

        start_pos = []
        end_pos = []
        tag_instantiation = OntoNotes_tag_instantiation()
        entity_type = tag_instantiation[entity_type]
        entity_type_token = self.tokenizer.tokenize(entity_type)
        entity_type_id = self.tokenizer.convert_tokens_to_ids(entity_type_token)

        entity_position = []

        token_list = []
        text_token = []

        for word in text.strip().split(" "):
            start = len(text_token)+1
            text_token += self.tokenizer.tokenize(word)
            end = len(text_token)
            start_pos.append(start)
            end_pos.append(end)
            token_list.append(word)



        real_label = [1] * self.maxlength


        while len(entity_position)<20:
            entity_position.append((151,151))
        assert len(entity_position)==20

        text_token = ["[CLS]"] + text_token + ["[SEP]"]

        text_id = self.tokenizer.convert_tokens_to_ids(text_token)
        text_id += entity_type_id
        length_pure_text = len(text_id)

        position_id = list(range(length_pure_text))
        position_id = self.pad_id(position_id, self.maxlength - 2 * self.soft_prompts_length)
        text_id = self.pad_id(text_id,self.maxlength-2*self.soft_prompts_length)

        start_id = [1]*len(start_pos)
        start_id = self.pad_id(start_id,self.soft_prompts_length)
        text_id += start_id

        end_id = [2]*len(end_pos)
        end_id = self.pad_id(end_id, self.soft_prompts_length)
        text_id += end_id
        assert len(text_id) ==self.maxlength


        position_id += start_pos
        position_id = self.pad_id(position_id, self.maxlength-self.soft_prompts_length)
        position_id += end_pos
        position_id = self.pad_id(position_id, self.maxlength)
        assert len(start_pos) == len(end_pos)

        self.all_text_id.append(text_id)
        self.position_ids.append(position_id)
        self.labels.append(real_label)
        self.span_labels.append(entity_position)
        self.length_pure_texts.append(length_pure_text)
        self.len_text_tokens.append(len(start_pos))

    def __getitem__(self, item):

        label_map = [0] * 512
        length_pure_text = self.length_pure_texts[item]
        len_text_token =self.len_text_tokens[item]

        attention_mask = torch.zeros((self.maxlength, self.maxlength), dtype=torch.int64)
        attention_mask[:length_pure_text, :length_pure_text] = 1

        for number_start in range((self.maxlength - 2 * self.soft_prompts_length),
                                  (self.maxlength - 2 * self.soft_prompts_length) + len_text_token):
            attention_mask[number_start][:length_pure_text] = 1
            attention_mask[number_start][number_start] = 1
            label_map[number_start] = 1
            for number_end in range((number_start + self.soft_prompts_length),
                                    (self.maxlength - self.soft_prompts_length) + len_text_token):
                attention_mask[number_start][number_end] = 1

        for number_end in range((self.maxlength - self.soft_prompts_length),
                                (self.maxlength - self.soft_prompts_length) + len_text_token):
            attention_mask[number_end][:length_pure_text] = 1
            attention_mask[number_end][number_end] = 1
            label_map[number_end] = 1
            for number_start in range((self.maxlength - 2 * self.soft_prompts_length),
                                      (number_end - self.soft_prompts_length) + 1):
                attention_mask[number_end][number_start] = 1

        return torch.LongTensor(self.all_text_id[item]), attention_mask, torch.LongTensor(self.position_ids[item]), torch.LongTensor(
            self.labels[item]), torch.LongTensor(label_map), torch.LongTensor(self.span_labels[item])

    def pad_id(self, ids, maxlength):

        if len(ids) < maxlength:
            ids = ids + [0] * (maxlength - len(ids))
        else:
            ids = ids[:maxlength]
        return ids

    def search(self, pattern, sequence, start_pos):

        n = len(pattern)
        for i in range(start_pos,len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def __len__(self):
        return self.len



class Electra_Discriminator_DataSet_RC:
    def __init__(self,data_path,pretrain_model_path,negtive_sample_rate,predict=None):

        self.tokenizer = ElectraTokenizer.from_pretrained(pretrain_model_path)
        raw_dataset = data_get_RC(data_path)
        self.triplets, self.all_labels = raw_dataset.get_relation_entities()
        self.templates = [""," is the "," of the "] 
        self.token_ids = []
        self.attention_mask = []
        self.label_position = []
        self.labels = []
        self.types = []
        self.label_map = []
        self.negtive_sample_rate = negtive_sample_rate
        self.maxlength = 128
        self.predict = predict


        for line in self.triplets:
            text = line["text"]
            head_entity = line["head_entity"]
            tail_entity = line["tail_entity"]
            relation = line ["relation"]
            if self.predict is not None:
                self.get_all_test_sample(text, head_entity, tail_entity, self.templates,relation)
            else:
                self.get_pos_sample(text, head_entity, tail_entity, relation, self.templates)
                for i in range(negtive_sample_rate):
                    self.get_neg_sample(text, head_entity, tail_entity, relation, self.templates)
                    self.get_random_neg_sample(text, head_entity, tail_entity, relation, self.templates)
        self.len = len(self.token_ids)

    def __getitem__(self, index):
        return torch.LongTensor(self.token_ids[index]), torch.LongTensor(
            self.attention_mask[index]), torch.LongTensor(self.types[index]), torch.LongTensor(self.labels[index]), torch.LongTensor(
            self.label_position[index]), torch.LongTensor(self.label_map[index])

    def get_pos_sample(self,text,head_entity,tail_entity,relation,templates):

        text_with_template = self.add_templates(text, head_entity, tail_entity, relation, templates)

        tokens = self.tokenizer.tokenize(text_with_template)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        relation_piece = self.tokenizer.tokenize(relation)
        relation_id = self.tokenizer.convert_tokens_to_ids(relation_piece)
        start_pos = self.search(relation_id, ids)
        end_pos = start_pos + len(relation_id)
        if start_pos == -1:
            return

        masks = [1] * len(ids)
        types = [0] * len(ids)

        if len(ids) < self.maxlength:
            types = types + [1] * (self.maxlength - len(ids))
            masks = masks + [0] * (self.maxlength - len(ids))
            ids = ids + [0] * (self.maxlength - len(ids))
        else:
            types = types[:self.maxlength]
            masks = masks[:self.maxlength]
            ids = ids[:self.maxlength]

        label = [0] * len(ids)
        labelmap = [0] * len(ids)
        for num in range(len(labelmap)):
            if num >= start_pos and num < end_pos:
                labelmap[num] = 1


        assert len(ids) == len(masks) == len(types) == len(label)

        self.token_ids.append(ids)
        self.attention_mask.append(masks)
        self.types.append(types)
        self.label_position.append([start_pos,end_pos])
        self.labels.append(label)
        self.label_map.append(labelmap)



    def get_neg_sample(self,text,head_entity,tail_entity,relation,templates):
        all_relation = copy.copy(self.all_labels)
        all_relation.remove(relation)
        neg_relation = np.random.choice(all_relation)

        text_with_template = self.add_templates(text, head_entity, tail_entity, neg_relation, templates)

        tokens = self.tokenizer.tokenize(text_with_template)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        relation_piece = self.tokenizer.tokenize(neg_relation)
        relation_id = self.tokenizer.convert_tokens_to_ids(relation_piece)

        start_pos = self.search(relation_id, ids)
        if start_pos == -1:
            return
        end_pos = start_pos + len(relation_id)

        masks = [1] * len(ids)
        types = [0] * len(ids)

        if len(ids) < self.maxlength:
            types = types + [1] * (self.maxlength - len(ids))
            masks = masks + [0] * (self.maxlength - len(ids))
            ids = ids + [0] * (self.maxlength - len(ids))
        else:
            types = types[:self.maxlength]
            masks = masks[:self.maxlength]
            ids = ids[:self.maxlength]

        neg_label = [0] * len(ids)
        labelmap = [0] * len(ids)
        for num in range(len(neg_label)):
            if num >= start_pos and num < end_pos:
                neg_label[num] = 1
                labelmap[num] = 1

        assert len(ids) == len(masks) == len(types) == len(neg_label)

        self.token_ids.append(ids)
        self.attention_mask.append(masks)
        self.types.append(types)
        self.label_position.append([start_pos, end_pos])
        self.labels.append(neg_label)
        self.label_map.append(labelmap)


    def get_random_neg_sample(self,text,head_entity,tail_entity,relation,templates):
        neg_relation = relation
        dice = random.randint(1,3)
        t_lis = text.split(" ")
        if len(t_lis) <5:
            return
        if dice == 1:
            dice_length = random.randint(1,5)
            dice_start = random.randint(0,len(t_lis)-5)
            tail_entity = " ".join(t_lis[dice_start:dice_start+dice_length])
        elif dice ==2:
            dice_length = random.randint(1, 5)
            dice_start = random.randint(0, len(t_lis) - 5)
            head_entity = " ".join(t_lis[dice_start:dice_start+dice_length])
        else:
            dice3 = random.randint(1,3)
            if dice3==1:
                tmp = head_entity
                head_entity = tail_entity
                tail_entity =tmp
            else:
                dice_length1 = random.randint(1, 5)
                dice_length2 = random.randint(1, 5)
                dice_start1 = random.randint(0, len(t_lis) - 5)
                dice_start2 = random.randint(0, len(t_lis) - 5)

                head_entity = " ".join(t_lis[dice_start1:dice_start1+dice_length1])
                tail_entity = " ".join(t_lis[dice_start2:dice_start2+dice_length2])

        text_with_template = self.add_templates(text, head_entity, tail_entity, neg_relation, templates)

        tokens = self.tokenizer.tokenize(text_with_template)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        relation_piece = self.tokenizer.tokenize(neg_relation)
        relation_id = self.tokenizer.convert_tokens_to_ids(relation_piece)

        start_pos = self.search(relation_id, ids)
        if start_pos == -1:
            return
        end_pos = start_pos + len(relation_id)

        masks = [1] * len(ids)
        types = [0] * len(ids)

        if len(ids) < self.maxlength:
            types = types + [1] * (self.maxlength - len(ids))
            masks = masks + [0] * (self.maxlength - len(ids))
            ids = ids + [0] * (self.maxlength - len(ids))
        else:
            types = types[:self.maxlength]
            masks = masks[:self.maxlength]
            ids = ids[:self.maxlength]

        neg_label = [0] * len(ids)
        labelmap = [0] * len(ids)
        for num in range(len(neg_label)):
            if num >= start_pos and num < end_pos:
                neg_label[num] = 1
                labelmap[num] = 1

        assert len(ids) == len(masks) == len(types) == len(neg_label)

        self.token_ids.append(ids)
        self.attention_mask.append(masks)
        self.types.append(types)
        self.label_position.append([start_pos, end_pos])
        self.labels.append(neg_label)
        self.label_map.append(labelmap)


    def get_all_test_sample(self,text,head_entity,tail_entity,templates,pos_relation):
        all_relation = copy.copy(self.all_labels)
        for rel in all_relation:
            text_with_template = self.add_templates(text, head_entity, tail_entity, rel, templates)
            tokens = self.tokenizer.tokenize(text_with_template)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            relation_piece = self.tokenizer.tokenize(rel)
            relation_id = self.tokenizer.convert_tokens_to_ids(relation_piece)

            start_pos = self.search(relation_id, ids)
            if start_pos == -1:
                return
            end_pos = start_pos + len(relation_id)

            masks = [1] * len(ids)
            types = [0] * len(ids)

            if len(ids) < self.maxlength:
                types = types + [1] * (self.maxlength - len(ids))
                masks = masks + [0] * (self.maxlength - len(ids))
                ids = ids + [0] * (self.maxlength - len(ids))
            else:
                types = types[:self.maxlength]
                masks = masks[:self.maxlength]
                ids = ids[:self.maxlength]

            real_label = [0] * len(ids)
            labelmap = [0] * len(ids)
            for num in range(len(real_label)):
                if num >= start_pos and num < end_pos:
                    if rel != pos_relation:
                        real_label[num] = 1
                    labelmap[num] = 1

            assert len(ids) == len(masks) == len(types) == len(real_label)

            self.token_ids.append(ids)
            self.attention_mask.append(masks)
            self.types.append(types)
            self.label_position.append([start_pos, end_pos])
            self.labels.append(real_label)
            self.label_map.append(labelmap)

    def add_templates(self,text,head,tail,relation,template):

        return text + "[SEP]" + template[0] +tail + template[1] + relation + template[2] + head

    def search(self,pattern, sequence):
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def __len__(self):
        return self.len

class MedMention_Dataset:
    def __init__(self,path,pretrain_path,all_label,predict=False,label_type=None):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        text_list,entity_list,entity_type_list = load_data_NER_Soft_Prompts(path)
        self.all_text_id = []
        self.attention_masks = []
        self.position_ids = []
        self.span_labels = []
        self.labels = []
        self.label_maps = []
        self.length_pure_texts = []
        self.len_text_tokens = []

        self.maxlength = 512
        self.maxtext = 200
        self.soft_prompts_length = 110
        self.all_label = all_label
        self.predict = predict
        self.label_type = label_type
        self.all = True

        self.tmp = {}
        if predict:
            for text,entity,entity_type in zip(text_list,entity_list,entity_type_list):
                if len(self.tokenizer.tokenize(text)) > self.maxtext-5 or len(text.strip().split(" "))>150:
                    text = text[:150]
                self.get_sample(text, entity, entity_type)
        else:
            for text,entity,entity_type in zip(text_list,entity_list,entity_type_list):
                if len(self.tokenizer.tokenize(text)) > self.maxtext-5 or len(text.strip().split(" "))>150:
                    text = text[:150]
                self.get_sample(text,entity,entity_type)
        self.len = len(self.all_text_id)
    def get_sample(self, text, entity, entity_type):

        if self.label_type and self.label_type != entity_type:
            return
        tag_instantiation = MedMention_tag_instantiation()
        start_pos = []
        end_pos = []
        entity_type = tag_instantiation[entity_type]
        entity_type_token = self.tokenizer.tokenize(entity_type)
        entity_type_id = self.tokenizer.convert_tokens_to_ids(entity_type_token)

        entity_position = []
        entity_list = entity.split(",")

        token_list = []
        text_token = []

        first_start_soft_prompt_position = self.maxlength - 2 * self.soft_prompts_length
        first_end_soft_prompt_position = self.maxlength -  self.soft_prompts_length
        for word in text.strip().split(" "):
            start = len(text_token)+1
            text_token += self.tokenizer.tokenize(word)
            end = len(text_token)
            start_pos.append(start)
            end_pos.append(end)
            token_list.append(word)


        real_label = [1] * self.maxlength


        for e in entity_list:
            if e not in self.tmp:
                self.tmp[e] = 1
            else:
               self.tmp[e] += 1

            e = e.strip().split(" ")
            e_start = self.search(e,token_list,0)
            if e_start == -1:
                return
            last_entity_position = e_start+len(e)-1
            entity_position.append((e_start,last_entity_position))
            real_label[first_start_soft_prompt_position+e_start] = 0
            real_label[first_end_soft_prompt_position+last_entity_position] = 0


        while len(entity_position)<20:
            entity_position.append((151,151))
        while len(entity_position)>20:
            entity_position.pop()
        assert len(entity_position)==20

        text_token = ["[CLS]"] + text_token + ["[SEP]"]

        text_id = self.tokenizer.convert_tokens_to_ids(text_token)
        text_id += entity_type_id
        length_pure_text = len(text_id)

        position_id = list(range(length_pure_text))
        position_id = self.pad_id(position_id, self.maxlength - 2 * self.soft_prompts_length)
        text_id = self.pad_id(text_id,self.maxlength-2*self.soft_prompts_length)


        start_id = [1]*len(start_pos)
        start_id = self.pad_id(start_id,self.soft_prompts_length)
        text_id += start_id

        end_id = [2]*len(end_pos)
        end_id = self.pad_id(end_id, self.soft_prompts_length)
        text_id += end_id
        assert len(text_id) ==self.maxlength


        position_id += start_pos
        position_id = self.pad_id(position_id, self.maxlength-self.soft_prompts_length)
        position_id += end_pos
        position_id = self.pad_id(position_id, self.maxlength)
        assert len(start_pos) == len(end_pos)

        self.all_text_id.append(text_id)
        self.position_ids.append(position_id)
        self.labels.append(real_label)
        self.span_labels.append(entity_position)
        self.length_pure_texts.append(length_pure_text)
        self.len_text_tokens.append(len(start_pos))


    def get_neg_sample(self, text, entity_type):

        start_pos = []
        end_pos = []
        tag_instantiation = MedMention_tag_instantiation()
        entity_type = tag_instantiation[entity_type]
        entity_type_token = self.tokenizer.tokenize(entity_type)
        entity_type_id = self.tokenizer.convert_tokens_to_ids(entity_type_token)

        entity_position = []
        token_list = []
        text_token = []

        for word in text.strip().split(" "):
            start = len(text_token)+1
            text_token += self.tokenizer.tokenize(word)
            end = len(text_token)
            start_pos.append(start)
            end_pos.append(end)
            token_list.append(word)



        real_label = [1] * self.maxlength


        while len(entity_position)<20:
            entity_position.append((151,151))
        assert len(entity_position)==20

        text_token = ["[CLS]"] + text_token + ["[SEP]"]

        text_id = self.tokenizer.convert_tokens_to_ids(text_token)
        text_id += entity_type_id
        length_pure_text = len(text_id)

        position_id = list(range(length_pure_text))
        position_id = self.pad_id(position_id, self.maxlength - 2 * self.soft_prompts_length)
        text_id = self.pad_id(text_id,self.maxlength-2*self.soft_prompts_length)
        
        start_id = [1]*len(start_pos)
        start_id = self.pad_id(start_id,self.soft_prompts_length)
        text_id += start_id

        end_id = [2]*len(end_pos)
        end_id = self.pad_id(end_id, self.soft_prompts_length)
        text_id += end_id
        assert len(text_id) ==self.maxlength


        position_id += start_pos
        position_id = self.pad_id(position_id, self.maxlength-self.soft_prompts_length)
        position_id += end_pos
        position_id = self.pad_id(position_id, self.maxlength)
        assert len(start_pos) == len(end_pos)

        self.all_text_id.append(text_id)
        self.position_ids.append(position_id)
        self.labels.append(real_label)
        self.span_labels.append(entity_position)
        self.length_pure_texts.append(length_pure_text)
        self.len_text_tokens.append(len(start_pos))

    def __getitem__(self, item):

        label_map = [0] * 512
        length_pure_text = self.length_pure_texts[item]
        len_text_token =self.len_text_tokens[item]

        attention_mask = torch.zeros((self.maxlength, self.maxlength), dtype=torch.int64)
        attention_mask[:length_pure_text, :length_pure_text] = 1
        #mask for soft prompts
        for number_start in range((self.maxlength - 2 * self.soft_prompts_length),
                                  (self.maxlength - 2 * self.soft_prompts_length) + len_text_token):
            attention_mask[number_start][:length_pure_text] = 1
            attention_mask[number_start][number_start] = 1
            label_map[number_start] = 1

            for number_end in range((number_start + self.soft_prompts_length),
                                    (self.maxlength - self.soft_prompts_length) + len_text_token):

                attention_mask[number_start][number_end] = 1

        for number_end in range((self.maxlength - self.soft_prompts_length),
                                (self.maxlength - self.soft_prompts_length) + len_text_token):
            attention_mask[number_end][:length_pure_text] = 1
            attention_mask[number_end][number_end] = 1
            label_map[number_end] = 1
            for number_start in range((self.maxlength - 2 * self.soft_prompts_length),
                                      (number_end - self.soft_prompts_length) + 1):

                attention_mask[number_end][number_start] = 1

        return torch.LongTensor(self.all_text_id[item]), attention_mask, torch.LongTensor(self.position_ids[item]), torch.LongTensor(
            self.labels[item]), torch.LongTensor(label_map), torch.LongTensor(self.span_labels[item])

    def pad_id(self, ids, maxlength):

        if len(ids) < maxlength:
            ids = ids + [0] * (maxlength - len(ids))
        else:
            ids = ids[:maxlength]
        return ids

    def search(self, pattern, sequence, start_pos):
        n = len(pattern)
        for i in range(start_pos,len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def __len__(self):
        return self.len





class NER_data_get_For_Zero_Relation_Extraction:
    def __init__(self,data_path):
        self.data = ZeroRTE_Dataset.load(data_path)
        self.triplets = []
        self.all_labels = []
    def list_to_str(self,tokens) -> str:
        return " ".join(tokens)

    def get_entities(self):
        return NotImplementedError

    def get_relation_entities(self):
        for s in self.data.sents:
            for t in s.triplets:
                if not t.head or  not t.tail:
                    continue
                triplet = {}
                triplet["text"] = self.list_to_str(t.tokens)
                triplet["head_entity"] = self.list_to_str(t.tokens[t.head[0]:t.head[-1]+1])
                triplet["tail_entity"] = self.list_to_str(t.tokens[t.tail[0]:t.tail[-1]+1])
                triplet["relation"] = t.label

                if self.triplets and triplet["text"] == self.triplets[-1]["text"]:
                    tmp_index = -1
                    flag = 1
                    while triplet["text"] == self.triplets[tmp_index]["text"]:
                        if triplet["relation"]==self.triplets[tmp_index]["relation"]:
                            self.triplets[tmp_index]["head_entity"] = self.triplets[tmp_index]["head_entity"] + "//t" + \
                                                                      triplet["head_entity"]
                            self.triplets[tmp_index]["tail_entity"] = self.triplets[tmp_index]["tail_entity"] + "//t" + \
                                                                      triplet["tail_entity"]
                            flag = 0
                            break
                        tmp_index -= 1
                    if flag == 1:
                        self.triplets.append(triplet)
                else:
                    self.triplets.append(triplet)
                if t.label not in self.all_labels:
                    self.all_labels.append(t.label)
        return self.triplets, self.all_labels




class NER_Dataset_For_Zero_Relation_Extraction:
    def __init__(self,path,pretrain_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path)

        raw_dataset = NER_data_get_For_Zero_Relation_Extraction(path)
        self.triplets, self.all_labels = raw_dataset.get_relation_entities()

        self.all_text_id = []
        self.attention_mask = []
        self.type_id = []
        self.type_start_pos = []
        self.type_end_pos = []
        self.labels = []
        self.label_maps = []
        self.maxlength = 128
        self.maxtext = 100


        for line in self.triplets:
            text = line["text"]
            head_entity = line["head_entity"]
            tail_entity = line["tail_entity"]
            relation = line["relation"]
            if len(self.tokenizer.tokenize(text)) > self.maxtext:
                continue
            self.get_pos_sample(text, str(head_entity)+"//t"+str(tail_entity), relation)

        self.len = len(self.all_text_id)
        print(self.len)
    def __getitem__(self, item):
        return torch.LongTensor(self.all_text_id[item]), torch.LongTensor(
            self.attention_mask[item]), torch.LongTensor(self.type_id[item]), torch.LongTensor(
            self.labels[item]), torch.LongTensor(self.label_maps[item])


    def get_pos_sample(self,text,entity,entity_type):

        text_token, entity_position_start, entity_position_end, text_length = self.get_sample(text, entity, entity_type)

        text_id = self.tokenizer.convert_tokens_to_ids(text_token)
        label = [0] * self.maxlength
        label_map = [0] * self.maxlength

        for start,end in zip(entity_position_start,entity_position_end):
            for i in range(start,end):
                label[i] = 1
                label_map[i] = 1

        for j in range(1, text_length+1):
            label_map[j] = 1

        masks = [1] * len(text_id)
        types = [0] * len(text_id)

        text_id, types, masks = self.pad_id(self.maxlength, text_id, types, masks)
        self.all_text_id.append(text_id)
        self.type_id.append(types)
        self.attention_mask.append(masks)
        self.labels.append(label)
        self.label_maps.append(label_map)



    def pad_id(self,maxlength, ids, types, masks):
        if len(ids) < maxlength:
            types = types + [1] * (maxlength - len(ids))
            masks = masks + [0] * (maxlength - len(ids))
            ids = ids + [0] * (maxlength - len(ids))
        else:
            types = types[:maxlength]
            masks = masks[:maxlength]
            ids = ids[:maxlength]
        return ids, types, masks

    def search(self, pattern, sequence):
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    def get_sample(self,text,entity,entity_type):

        start_pos = []
        end_pos = []
        templates = "[ Identify entities related to the relationship of " + entity_type + "]"
        entity_type_token = self.tokenizer.tokenize(templates)
        text_token = self.tokenizer.tokenize(text)
        text_length = len(text_token)  # text len
        text_token = text_token + ["[SEP]"] + entity_type_token
        text_tokens = ["[CLS]"] + text_token + ["[SEP]"]

        entity = entity.split("//t")
        for e in entity:
            entity_token = self.tokenizer.tokenize(e)
            entity_start_pos = self.search(entity_token, text_tokens)
            entity_end_pos = entity_start_pos + len(entity_token)
            start_pos.append(entity_start_pos)
            end_pos.append(entity_end_pos)
        return text_tokens,start_pos,end_pos,text_length

    def __len__(self):
        return self.len


def search(pattern, sequence):

    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1



def pad_id(maxlength,ids,types,masks):

    if len(ids) < maxlength:
        types = types + [1] * (maxlength - len(ids))
        masks = masks + [0] * (maxlength - len(ids))
        ids = ids + [0] * (maxlength - len(ids))
    else:
        types = types[:maxlength]
        masks = masks[:maxlength]
        ids = ids[:maxlength]
    return ids,types,masks


def multi_candidate_RC_For_RE_DataSet(text,tokenizer,head_entity,tail_entity,candidate_relation):

    relation_maxlength = 256
    input_tokens = ["[CLS]"] + tokenizer.tokenize(text)
    rel_ids = []

    for l in candidate_relation:
        template = ["", " is the ", " of the "]
        templates0 = tokenizer.tokenize(template[0])
        templates1 = tokenizer.tokenize(template[1])
        template2 = tokenizer.tokenize(template[2])
        rel = tokenizer.tokenize(l)
        input_tokens += templates0 + tail_entity + templates1 + rel + template2 + head_entity + ["[SEP]"]
        rel_ids.append(tokenizer.convert_tokens_to_ids(rel))


    text_template_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    text_template_masks = [1] * len(text_template_ids)
    text_template_types = [0] * len(text_template_ids)
    text_template_ids, text_template_types, text_template_masks = pad_id(relation_maxlength, text_template_ids,
                                                                         text_template_types, text_template_masks)

    rel_maps = [[0] * len(text_template_ids) for j in range(len(candidate_relation))]
    labelmap = [0] * len(text_template_ids)
    for rel_id, rel_map in zip(rel_ids, rel_maps):
        start_pos = search(rel_id, text_template_ids)
        if start_pos == -1:
            return
        end_pos = start_pos + len(rel_id)

        for num in range(len(labelmap)):
            if num >= start_pos and num < end_pos:
                labelmap[num] = 1
                rel_map[num] = 1
    return text_template_ids,text_template_masks,text_template_types,labelmap,rel_maps


def NER_For_RE_DataSet(data,tokenizer):
    maxlength = 128


    text = data["text"]
    candidate_relation = data["relation"]


    tokens = tokenizer.tokenize(text)
    if len(tokens) > 100:
        tokens = tokens[:100]

    text_length = len(tokens)

    templates = "[ Identify entities related to the relationship of " + candidate_relation + "]"
    template = tokenizer.tokenize(templates)

    text_token = tokens + ["[SEP]"] + template
    ner_text_tokens = ["[CLS]"] + text_token + ["[SEP]"]

    ids = tokenizer.convert_tokens_to_ids(ner_text_tokens)

    masks = [1] * len(ids)
    types = [0] * len(ids)

    label_map = [0] * maxlength

    for j in range(1, text_length + 1):
        label_map[j] = 1

    if len(ids) < maxlength:
        types = types + [1] * (maxlength - len(ids))
        masks = masks + [0] * (maxlength - len(ids))
        ids = ids + [0] * (maxlength - len(ids))
    else:
        types = types[:maxlength]
        masks = masks[:maxlength]
        ids = ids[:maxlength]

    return text_token,ids,types,masks,label_map




def RC_For_RE_DataSet(text,tokenizer,head_entity,tail_entity,candidate_relation):
    maxlength = 128
    template = ["", " is the ", " of the "]

    text_tokens = tokenizer.tokenize(text)
    templates0 = tokenizer.tokenize(template[0])
    templates1 = tokenizer.tokenize(template[1])
    template2 = tokenizer.tokenize(template[2])
    rel = tokenizer.tokenize(candidate_relation)
    input_tokens = ["[CLS]"] + text_tokens + templates0 + tail_entity + templates1 + rel + template2 + head_entity + ["[SEP]"]

    text_template_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    rel_ids = tokenizer.convert_tokens_to_ids(rel)

    start_pos = search(rel_ids, text_template_ids)
    if start_pos == -1:
        return
    end_pos = start_pos + len(rel_ids)

    text_template_masks = [1] * len(text_template_ids)
    text_template_types = [0] * len(text_template_ids)

    text_template_ids, text_template_types, text_template_masks = pad_id(maxlength, text_template_ids,
                                                                         text_template_types, text_template_masks)

    labelmap = [0] * len(text_template_ids)
    for num in range(len(labelmap)):
        if num >= start_pos and num < end_pos:
            labelmap[num] = 1

    return text_template_ids, text_template_masks, text_template_types, labelmap





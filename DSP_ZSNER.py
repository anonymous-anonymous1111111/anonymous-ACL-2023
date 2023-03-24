import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import ElectraForPreTraining, ElectraConfig,AdamW,get_cosine_schedule_with_warmup
import torch
import time
from torch.utils.data import DataLoader
import random
from sklearn.metrics import  f1_score, precision_score, recall_score
from data_process import MedMention_Dataset,OntoNote_Dataset
import os
import argparse
from score  import  macro_score
#set random seed
random.seed(2022)
np.random.seed(2022)



medmention_train_label = ["Biologic Function",'Chemical',"Health Care Activity",
            'Anatomical Structure','Finding','Spatial Concept',
            'Intellectual Product','Research Activity',
            'Eukaryote','Population Group','Medical Device']
medmention_dev_label = ['Organization','Injury or Poisoning','Clinical Attribute',
            'Virus','Biomedical Occupation or Discipline']
medmention_test_label = ['Bacterium','Professional or Occupational Group','Food',
              'Body Substance','Body System']


ontonotes_train_label =["PERSON","ORG","GPE","DATE"]
ontonotes_test_label = ['CARDINAL','TIME','LOC','WORK_OF_ART', 'FAC','QUANTITY','LANGUAGE']
ontonotes_dev_label = ['NORP','MONEY','ORDINAL','PERCENT','EVENT','PRODUCT','LAW']






def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

class ElectraDiscriminatorTraining_NER(nn.Module):
    def __init__(self,electra_path):
        super(ElectraDiscriminatorTraining_NER, self).__init__()
        self.config = ElectraConfig.from_pretrained(electra_path)
        self.discriminator = ElectraForPreTraining.from_pretrained(electra_path)

    def forward(self,input_ids, attention_mask=None, position_ids=None,labels=None,label_map=None):
        discriminator_outputs = self.discriminator(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids)
        logits = discriminator_outputs[0]
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits,labels.float())
            loss = torch.sum(loss*label_map)/torch.sum(label_map)
            return loss
        else:
            pred = torch.round(logits*label_map)
            return pred



def train(electra_model,Device,config):
    if config["data_type"] == "ontonotes":
        dataset = OntoNote_Dataset(config["train_data_path"], config["pretrain_model_path"], ontonotes_train_label,
                                   predict=False)
    else:
        dataset = MedMention_Dataset(config["train_data_path"], config["pretrain_model_path"], medmention_train_label,
                                     predict=False)
    traindata = DataLoader(dataset, shuffle=True, batch_size=config["batch_size"])
    model = electra_model.to(Device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(traindata),
                                                num_training_steps=config['num_epochs'] * len(traindata))
    print(get_parameter_number(model))
    lamb = config["lamda"]
    loss_reg = 0.0
    patience = 0
    last_loss = 0
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        print('epoch number:', epoch)
        train_loss_sum = 0
        train_ce_loss_sum = 0
        for batch_idx, x in tqdm(enumerate(traindata)):
            start = time.time()
            ids, attention_mask, position_ids, label, label_map = x[0].to(Device), x[1].to(Device), x[2].to(Device), x[3].to(Device), x[4].to(Device)

            ce_loss = model(input_ids=ids, attention_mask=attention_mask, position_ids=position_ids, labels=label,
                         label_map=label_map)

            if config["parameter_variation"]:
                for (_, param), (_,param_old) in zip(model.named_parameters(), org_model.named_parameters()):
                    param_old = param_old.detach()
                    loss_reg += torch.sum((param_old - param).pow(2)) / 2  # parameter variation
                loss = ce_loss + lamb*loss_reg
                loss_reg = 0.0

            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()
            train_ce_loss_sum += ce_loss.item()

            if (batch_idx + 1) % (len(traindata) // 50) == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | CE_Loss {:.4f} | Sum_Loss {:.4f} |Time {:.4f}".format(
                    epoch, batch_idx + 1, len(traindata), train_loss_sum / (batch_idx + 1),
                           train_ce_loss_sum / (batch_idx + 1), time.time() - start))
                print(abs(last_loss - (train_ce_loss_sum / (batch_idx + 1))))
                if abs(last_loss - (train_ce_loss_sum / (batch_idx + 1)))<0.01:
                    patience+=1
                else:
                    last_loss = train_ce_loss_sum / (batch_idx + 1)
                if patience>3:
                    return

        torch.save(model,config["model_save_path"])







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='data/ner_data/MedMention/train.data', type=str,
                        help='train dataset path')
    parser.add_argument('--test_data_path', default='data/ner_data/MedMention/test.data', type=str,
                        help='test dataset path')
    parser.add_argument('--model_save_path', default='save_model/medmention_ner_model/medmention_model.pth',
                        type=str,
                        help='model save path')
    parser.add_argument('--load_model_path', default='save_model/medmention_ner_model/NER_MedMetion.pth',
                        type=str,
                        help='Previous trained model save path')
    parser.add_argument('--pretrain_model_path', default='pretrain_model/electra-base-discriminator', type=str,
                        help='pretrained model')
    parser.add_argument('--data_type', default='medmention', type=str,
                        help='pretrained model')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--number_epochs', default=3, type=int,
                        help='epochs')
    parser.add_argument('--lamda', default=0.1, type=float,
                        help='parameter variation lamda')
    parser.add_argument('--test', action="store_true", default=False,
                        help='if run test mode')
    parser.add_argument('--parameter_variation', action="store_true", default=True,
                        help='if use parameter variation')
    args = parser.parse_args()

    config = {
            "pretrain_model_path":"pretrain_model/electra-base-discriminator",
            "batch_size": 4,
            "learning_rate": 2e-5,
            "num_epochs": 8,
            "train_data_path":"data/ner_data/MedMention/train.data",
            "test_data_path":"data/ner_data/MedMention/test.data",
            "max_length":512,
            "data_type":"medmention",
            "test":True,
            "parameter_variation":True
        }

    config["train_data_path"] = args.train_data_path
    config["test_data_path"] = args.test_data_path
    config["model_save_path"] = args.model_save_path
    config["load_model_path"] = args.load_model_path
    config["pretrain_model_path"] = args.pretrain_model_path
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["num_epochs"] = args.number_epochs
    config["data_type"] = args.data_type
    config["parameter_variation"] = args.parameter_variation
    config["test"] = args.test
    config["lamda"] = args.lamda/2


    print(config)

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["test"]:
        model = torch.load(config["load_model_path"]).to(Device)
        if config["data_type"] == "medmention":
            macro_score(model,config,medmention_test_label,Device)
        else:
            macro_score(ontonotes_test_label)
    else:
        if config["parameter_variation"]:
            org_model = ElectraForPreTraining.from_pretrained(config['pretrain_model_path'])
            org_model = org_model.to(Device)
            org_model.eval()

        if config["data_type"] == "medmention":
            if not os.path.exists("save_model/medmention_ner_model"):
                os.makedirs("save_model/medmention_ner_model")
        else:
            if not os.path.exists("save_model/ontonotes_ner_model"):
                os.makedirs("save_model/ontonotes_ner_model")
        model = ElectraDiscriminatorTraining_NER(config["pretrain_model_path"])
        train(model, Device, config)

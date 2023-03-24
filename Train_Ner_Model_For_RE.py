import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import ElectraForPreTraining, ElectraConfig,AdamW,get_cosine_schedule_with_warmup
import torch
import time
from torch.utils.data import DataLoader
import random
import argparse
from data_process import NER_Dataset_For_Zero_Relation_Extraction
from score import NER_For_RE_predict

random.seed(2022)
np.random.seed(2022)




def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

class ElectraDiscriminatorTraining_NER(nn.Module):
    def __init__(self,electra_path):
        super(ElectraDiscriminatorTraining_NER, self).__init__()
        self.config = ElectraConfig.from_pretrained(electra_path)
        self.discriminator = ElectraForPreTraining.from_pretrained(electra_path)

    def forward(self,input_ids, attention_mask=None, token_type_ids=None,labels=None,label_map=None):
        discriminator_outputs = self.discriminator(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
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

    dataset = NER_Dataset_For_Zero_Relation_Extraction(config["train_data_path"],config["pretrain_model_path"])
    traindata = DataLoader(dataset, shuffle=True, batch_size=config["batch_size"])
    model = electra_model.to(Device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(traindata),
                                                num_training_steps=config['num_epochs'] * len(traindata))
    print(get_parameter_number(model))
    best_f1 = 0
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        print('epoch number:', epoch)
        train_loss_sum = 0
        train_loss1_sum = 0
        lamb = 0.05
        loss_reg = 0.0
        patience = 0
        for batch_idx, x in tqdm(enumerate(traindata)):
            start = time.time()
            ids, attention_mask, type, label, label_map = x[0].to(Device), x[1].to(Device), x[2].to(Device), x[3].to(Device), x[4].to(Device)
            ce_loss = model(input_ids=ids, attention_mask=attention_mask, token_type_ids=type, labels=label,
                         label_map=label_map)

            ### Choose if use parameter variation loss
            if config["parameter_variation"]:
                for (_, param), (_, param_old) in zip(model.named_parameters(), org_model.named_parameters()):
                    param_old = param_old.detach()
                    loss_reg += torch.sum((param_old - param).pow(2)) / 2
                loss = ce_loss + lamb * loss_reg
                loss_reg = 0.0

            else:
                loss = ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()
            train_loss1_sum += ce_loss.item()
            if (batch_idx + 1) % (len(traindata) // 5) == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Loss1 {:.4f} |Time {:.4f}".format(
                    epoch, batch_idx + 1, len(traindata), train_loss_sum / (batch_idx + 1),
                           train_loss1_sum / (batch_idx + 1), time.time() - start))

        p,r,f1 = NER_For_RE_predict(model,Device,config)
        if f1>best_f1:
            torch.save(model,config["model_save_path"])
            best_f1 = f1
        else:
            patience += 1
            if patience >3:
                print("Early Stopping")
                return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='data/zero_rte/wiki/unseen_5_seed_0/train.jsonl', type=str,
                        help='train dataset path')
    parser.add_argument('--test_data_path', default='data/zero_rte/wiki/unseen_5_seed_0/test.jsonl', type=str,
                        help='test dataset path')
    parser.add_argument('--dev_data_path', default='data/zero_rte/wiki/unseen_5_seed_0/dev.jsonl', type=str,
                        help='test dataset path')
    parser.add_argument('--model_save_path', default='save_model/relation_classification_model/best_ner_model_for_RE.pth',
                        type=str,
                        help='model save path')
    parser.add_argument('--load_model_path', default='save_model/relation_classification_model/NER_wiki_5way.pth',
                        type=str,
                        help='Previous trained model save path')
    parser.add_argument('--pretrain_model_path', default='pretrain_model/electra-base-discriminator', type=str,
                        help='pretrained model')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--number_epochs', default=8, type=int,
                        help='epochs')
    parser.add_argument('--lamda', default=0.1, type=float,
                        help='parameter variation lamda')
    parser.add_argument('--test', action="store_true", default=False,
                        help='if run test mode')
    parser.add_argument('--parameter_variation', action="store_true", default=False,
                        help='if use parameter variation')

    args = parser.parse_args()

    config = {
        "pretrain_model_path":"pretrain_model/electra-base-discriminator",
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 8,
        "parameter_variation":True,
        "test": True,
    }

    config["train_data_path"] = args.train_data_path
    config["test_data_path"] = args.test_data_path
    config["dev_data_path"] = args.dev_data_path
    config["model_save_path"] = args.model_save_path
    config["load_model_path"] = args.load_model_path
    config["pretrain_model_path"] = args.pretrain_model_path
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["num_epochs"] = args.number_epochs
    config["parameter_variation"] = args.parameter_variation
    config["test"] = args.test
    config["lamda"] = args.lamda/2




    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["test"]:
        model = torch.load(config["load_model_path"]).to(Device)
        NER_For_RE_predict(model, Device, config)
    else:
        if config["parameter_variation"]:
            org_model = ElectraForPreTraining.from_pretrained(config['pretrain_model_path'])
            org_model = org_model.to(Device)
            org_model.eval()

        import os

        if not os.path.exists("save_ner_model_for_relation_extraction"):
            os.makedirs("save_ner_model_for_relation_extraction")

        model = ElectraDiscriminatorTraining_NER(config["pretrain_model_path"])
        train(model, Device, config)



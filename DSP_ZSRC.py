from transformers import ElectraForPreTraining
import torch
from data_process import Electra_Discriminator_DataSet_RC
from torch.utils.data import DataLoader
from model import  ElectraDiscriminatorTraining
from transformers.optimization import  AdamW, get_cosine_schedule_with_warmup
import time
from tqdm import tqdm
import argparse
from score import RC_predict




def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)



def train(electra_model,traindata_path,Device,config):

    dataset = Electra_Discriminator_DataSet_RC(data_path=traindata_path, pretrain_model_path=config["pretrain_model_path"],negtive_sample_rate=config["negtive_sample_ratio"])
    traindata = DataLoader(dataset, shuffle=True, batch_size=config["batch_size"])
    model = electra_model.to(Device)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(traindata),
                                                num_training_steps=config['num_epochs'] * len(traindata))
    lamb = config["lamda"]
    loss_reg = 0.0
    print(get_parameter_number(model))
    best_f1 = 0
    patience = 0
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        print('epoch number:', epoch)
        train_loss_sum = 0
        train_ce_loss_sum = 0
        for batch_idx, x in tqdm(enumerate(traindata)):
            start = time.time()
            ids, attention_mask, type, label, label_position, label_map = x[0].to(Device), x[1].to(Device), x[2].to(Device), x[3].to(Device), x[4].to(Device), x[5].to(Device)
            ce_loss = model(input_ids=ids, attention_mask=attention_mask, token_type_ids=type, labels=label,
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

            if (batch_idx + 1) % (len(traindata) // 5) == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Loss1 {:.4f} |Time {:.4f}".format(
                    epoch, batch_idx + 1, len(traindata), train_loss_sum / (batch_idx + 1),train_ce_loss_sum / (batch_idx + 1), time.time() - start))

        f1 = RC_predict(model,config["dev_data_path"],config,Device)
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
    parser.add_argument('--train_data_path', default='data/zero_rte/fewrel/unseen_5_seed_0/train.jsonl', type=str,
                        help='train dataset path')
    parser.add_argument('--test_data_path', default='data/zero_rte/fewrel/unseen_5_seed_0/test.jsonl', type=str,
                        help='test dataset path')
    parser.add_argument('--dev_data_path', default='data/zero_rte/fewrel/unseen_5_seed_0/dev.jsonl', type=str,
                        help='test dataset path')
    parser.add_argument('--model_save_path', default='save_model/relation_classification_model/best_rc_model.pth', type=str,
                        help='model save path')
    parser.add_argument('--load_model_path', default='save_model/relation_classification_model/RC_fewrel_5_way.pth', type=str,
                        help='Previous trained model save path')
    parser.add_argument('--pretrain_model_path', default='pretrain_model/electra-base-discriminator', type=str,
                        help='pretrained model')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--number_epochs', default=8, type=int,
                        help='epochs')
    parser.add_argument('--negtive_sample_ratio', default=1, type=int,
                        help='negtive sample ratio for RC train')
    parser.add_argument('--lamda', default=0.1, type=float,
                        help='parameter variation lamda')
    parser.add_argument('--test', action="store_true", default=False,
                        help='if run test mode')
    parser.add_argument('--parameter_variation', action="store_true", default=False,
                        help='if use parameter variation')

    args = parser.parse_args()

    config = {
        'train_data_path': 'data/zero_rte/fewrel/unseen_5_seed_0/train.jsonl',
        'test_data_path': "data/zero_rte/fewrel/unseen_5_seed_0/test.jsonl",
        'dev_data_path':"data/zero_rte/fewrel/unseen_5_seed_0/dev.jsonl",
        'model_save_path': 'save_model/relation_classification_model/best_rc_model.pth',
        'load_model_path': 'save_model/relation_classification_model/best_rc_model.pth',
        'pretrain_model_path': "pretrain_model/electra-base-discriminator",  # your pretrain model path
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
    config["negtive_sample_ratio"] = args.negtive_sample_ratio
    config["parameter_variation"] = args.parameter_variation
    config["test"] = args.test
    config["lamda"] = args.lamda/2

    print(config)

    Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["test"]:
        model = torch.load(config["load_model_path"])
        print("Finish load model",config["load_model_path"])
        RC_predict(model, config["test_data_path"], config,Device)
    else:
        if config["parameter_variation"]:
            org_model = ElectraForPreTraining.from_pretrained(config['pretrain_model_path'])
            org_model = org_model.to(Device)
            org_model.eval()
        model = ElectraDiscriminatorTraining(config["pretrain_model_path"])
        train(model, config["train_data_path"], Device, config)

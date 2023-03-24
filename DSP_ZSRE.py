import torch
import argparse
from score import pipline_relation_extraction
from Train_Ner_Model_For_RE import ElectraDiscriminatorTraining_NER
from model import  ElectraDiscriminatorTraining




if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', default='data/zero_rte/wiki/unseen_5_seed_0/test.jsonl', type=str,
                        help='test dataset path')
    parser.add_argument('--NER_model_save_path', default='save_model/save_ner_model_for_relation_extraction/NER_wiki_5way.pth',
                        type=str,
                        help='model save path')
    parser.add_argument('--RC_model_save_path', default='save_model/relation_classification_model/best_rc_model.pth',
                        type=str,
                        help='model save path')
    parser.add_argument('--pretrain_model_path', default='pretrain_model/electra-base-discriminator', type=str,
                        help='pretrained model')
    args = parser.parse_args()


    RC_Model = args.RC_model_save_path
    NER_Model = args.NER_model_save_path
    data_path = args.test_data_path
    pretrain_model_path = args.pretrain_model_path


    relationmodel = torch.load(RC_Model)
    nermodel = torch.load(NER_Model)

    nermodel = nermodel.to(device)
    relationmodel = relationmodel.to(device)

    pipline_relation_extraction(data_path, pretrain_model_path, nermodel, relationmodel)
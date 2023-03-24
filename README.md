# :card_file_box: Anonymous reviewing for ACL 2023

## Requirements and Installation
```
python            >=   3.7
torch             >=   1.10.0
nltk              ==   3.8
scikit-learn      ==   1.0.2
sentencepiece     ==   0.1.97
pydantic          ==   1.10.2
transformers      ==   4.18.0
typing_extensions ==   4.3.0
pathlib           ==   1.0.1
```

```
pip install -r requirements.txt
```

## Data Preparing 

* FewRel and Wiki-ZSL can be downloaded from here [Fewrel_Wiki-ZSL](https://www.dropbox.com/s/lhmuc50r8bzmuov/RC_RE_DATA.zip?dl=0) to data/RC_RE_DATA.
* The MedMention dataset can be downloaded from the github of "MedMentions: A UMLS Annotated Dataset", and use our data/MedMetion_data_load.py for data preprocessing or download our processed data from here [MedMention](https://www.dropbox.com/s/1733hrax5rnzkuu/ner_data.zip?dl=0). 
* For OntoNotes, since the data is not free, you need to download it from this [link](https://catalog.ldc.upenn.edu/LDC2013T19), and then use our data/Step1_ontonotes_BIO_data_get.py, data/Step2_ontonotes_span_data_get.py to preprocess the data.
* The pre-trained language model [Electra](https://huggingface.co/google/electra-base-discriminator) can be downloaded from hugging face   to  pretrain_model/

* We also provide models that we have trained, which can be downloaded [models](https://www.dropbox.com/scl/fo/vy48wy2j5vrcm7mva46jk/h?dl=0&rlkey=ljht4tnsdwpwe918a3umxw2a2) to save_model/

You can define the path `$Trained_Model_Save_Path` to store the trained model.

## Zero-Shot NER Task

Run training for Zero-Shot NER Task
```
python DSP_ZSNER.py --train_data_path data/ner_data/MedMention/train.data  \
            --pretrain_model_path  pretrain_model/electra-base-discriminator \
            --model_save_path  $Trained_Model_Save_Path \
            --batch_size 16 \
            --number_epochs 10 \
            --parameter_variation \
            --lamda  0.1\
```


Run evaluation for Zero-Shot NER Task; You can use our trained model or your own trained model.

```
python DSP_ZSNER.py  --test \
            --test_data_path data/ner_data/MedMention/test.data   \
            --load_model_path  $Load_Model_Save_Path \
```

## Zero-Shot RC Task


Run training for Zero-Shot RC Task (fewrel, unseen=5, random-seed=0). You can change "fewrel" to "wiki" or unseen to 5/10/15 or seed to 0/1/2/3/4 in train_data_path.

```
python  DSP_ZSRC.py  --train_data_path data/RC_RE_DATA/fewrel/unseen_5_seed_0/train.jsonl \
                --dev_data_path  data/RC_RE_DATA/fewrel/unseen_5_seed_0/dev.jsonl \
                --model_save_path $Trained_Model_Save_Path \
                --pretrain_model_path pretrain_model/electra-base-discriminator \
                --batch_size 16 \
                --number_epochs 10 \
                --parameter_variation \
                --lamda  0.1 \
```

Run evaluation for Zero-Shot RC Task; You can use our trained model or your own trained model.

```
python  DSP_ZSRC.py --test \
               --test_data_path data/RC_RE_DATA/fewrel/unseen_5_seed_0/test.jsonl \
               --load_model_path $Load_Model_Save_Path \
```

## Zero-Shot RE Task

We use the pipeline method to complete the Zero-Shot RE task (Wiki-ZSL, unseen=5, random-seed=0).

First you should train a Zero-Shot NER model to extract candidate entities in the text. You can change "fewrel" to "wiki" or unseen to 5/10/15 or seed to 0/1/2/3/4 in train_data_path.

```
python Train_Ner_Model_For_RE.py   --train_data_path data/RC_RE_DATA/wiki/unseen_5_seed_0/train.jsonl \
                        --dev_data_path data/RC_RE_DATA/wiki/unseen_5_seed_0/dev.jsonl \
                        --model_save_path $Trained_Model_Save_Path \
                        --pretrain_model_path pretrain_model/electra-base-discriminator \
                        --batch_size 16 \
                        --number_epochs 10 \
                        --parameter_variation \
                        --lamda  0.1 \
```

Then train a Zero-Shot RC model, which is the same way with Zero-Shot RC Task.

```
python  DSP_ZSRC.py  --train_data_path data/RC_RE_DATA/wiki/unseen_5_seed_0/train.jsonl \
                --dev_data_path  data/RC_RE_DATA/wiki/unseen_5_seed_0/dev.jsonl \
                --model_save_path $Trained_Model_Save_Path \
                --pretrain_model_path pretrain_model/electra-base-discriminator \
                --batch_size 16 \
                --number_epochs 10 \
                --parameter_variation \
                --lamda  0.1 \
```


Run evaluation for Zero-Shot RE Task; You can use our our trained model or your own trained model.
```bash
python DSP_ZSRE.py  --test_data_path data/RC_RE_DATA/wiki/unseen_5_seed_0/test.jsonl \
            --NER_model_save_path $Load_NER_Model_Save_Path \
            --RC_model_save_path $Load_RC_Model_Save_Path \
```

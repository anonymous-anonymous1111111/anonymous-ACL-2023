import os


train_data_file = "train.sd.conllx"
dev_data_file = "dev.sd.conllx"
test_data_file = "test.sd.conllx"


Train_BIO_NER_Format_file = "train.bio.conllx"
Dev_BIO_NER_Format_file = "dev.bio.conllx"
Test_BIO_NER_Format_file = "test.bio.conllx"


def write_ner_format_data(out_file,lis):
    with open(out_file,"w",encoding="utf8") as w:
        for text in lis:
            w.write(text)



def loda_data(file_name):
    data_lis = []
    with open(file_name,"r",encoding="utf8") as r:
        for line in r.readlines():
            if not line.strip():
                data_lis.append(line)
                continue
            lis = line.strip().split("\t")
            data_lis.append( str(lis[0]) +"\t"+lis[1]+"\t"+lis[-1]+"\n")
            
    return data_lis
            


# get bio data
write_ner_format_data(Train_BIO_NER_Format_file,loda_data(train_data_file))
write_ner_format_data(Dev_BIO_NER_Format_file,loda_data(dev_data_file))
write_ner_format_data(Test_BIO_NER_Format_file,loda_data(test_data_file))
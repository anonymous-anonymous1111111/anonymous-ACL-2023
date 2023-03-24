

data_list = []



def get_span_data(file_name,output_file):

    with open(file_name,"r",encoding="utf8") as r:
        text = ""
        entity_list = []
        entity_type = []
        entity_start_pos = []
        entity_end_pos = []

        one_entity = []
        one_position = []
        one_entity_type = ""
        with open(output_file,"w",encoding="utf8") as w:
            for line in r.readlines():
                if not line.strip():
                    if one_entity and one_position:
                        entity_list.append(one_entity)
                        entity_type.append(one_entity_type)
                        entity_start_pos.append(one_position[0])
                        entity_end_pos.append(one_position[-1])
                        one_entity = []
                        one_position = []
                        one_entity_type = ""
                    if entity_list:
                        for num in range(len(entity_list)):
                             w.write(text+"\t"+" ".join(entity_list[num])+"\t"+entity_type[num]+"\n")
                    text = ""
                    entity_list = []
                    entity_type = []
                    entity_start_pos = []
                    entity_end_pos = []
                    continue
                pos,token,type = line.strip().split("\t")
                text = text+token+" "
                if type[0] == "B" or type[0] == "I":
                    one_entity.append(token)
                    one_position.append(pos)
                    one_entity_type = type[2:]
                elif type[0] == "O":
                    if one_entity and one_position:
                        entity_list.append(one_entity)
                        entity_type.append(one_entity_type)
                        entity_start_pos.append(one_position[0])
                        entity_end_pos.append(one_position[-1])
                        one_entity = []
                        one_position = []
                        one_entity_type = ""





def combine_data(file1,file2):
    with open(file1,"a",encoding="utf8") as w:
        with open(file2, "r", encoding="utf8") as r:
            for line in r.readlines():
                w.write(line)

# combine_data(ontonotes_zeroshot_data,"discriminator_ner_data.train")
# combine_data(ontonotes_zeroshot_data,"discriminator_ner_data.dev")
# combine_data(ontonotes_zeroshot_data,"discriminator_ner_data.test")


def get_ner_label(filename):
    type_list = []
    with open(filename, "r", encoding="utf8") as r:
        for line in r.readlines():
            t,e,l = line.strip().split("\t")
            if l not in type_list:
                type_list.append(l)
    print(type_list)





def split_train_test_dev(allfile,trainfile,testfile,devfile):
    NER_label_ontonotes = ['ORG', 'WORK_OF_ART', 'LOC', 'CARDINAL', 'EVENT', 'NORP', 'GPE', 'DATE', 'PERSON', 'FAC',
                           'QUANTITY', 'ORDINAL', 'TIME', 'PRODUCT', 'PERCENT', 'MONEY', 'LAW', 'LANGUAGE']

    train_label = ['PERSON', 'ORG', 'GPE', 'DATE' ]
    dev_label = ['NORP', 'MONEY', 'ORDINAL', 'PRODUCT','EVENT', 'PERCENT', 'LAW']
    test_label = ['CARDINAL','TIME','LOC','WORK_OF_ART', 'FAC','QUANTITY','LANGUAGE']

    r = open(allfile, "r", encoding="utf8")
    traindata = open(trainfile, "w", encoding="utf8")
    testfile = open(testfile, "w", encoding="utf8")
    devfile = open(devfile, "w", encoding="utf8")


    for line in r.readlines():
        if len(line.strip().split("\t")) !=3:
            continue
        t, l, e = line.strip().split("\t")
        if l in train_label:
            traindata.write(line)
        elif l in dev_label:
            devfile.write(line)
        elif l in test_label:
            testfile.write(line)
        else:
            print(line)
            print("error!!!")
            return


trainfile = "ontonotes_same_type_span.train"
testfile = "ontonotes_same_type_span.test"
devfile = "ontonotes_same_type_span.dev"

split_train_test_dev("ontonotes_same_type_span.txt",trainfile,testfile,devfile)
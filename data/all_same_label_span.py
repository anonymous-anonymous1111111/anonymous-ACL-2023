


source_file = "ontonotes_zeroshot_data.all"

dict = {}
with open(source_file,"r",encoding="utf8") as r:
    for line in r.readlines():
        text,entity,type = line.strip().split("\t")
        if str(text)+"\t"+str(type) not in dict:
            dict[str(text)+"\t"+str(type)] = []
        dict[str(text) + "\t" + str(type)].append(entity)




target_file = "ontonotes_same_type_span.txt"
delate = ['D.', 'R.', 'R', 'D']
with open(target_file,"w",encoding="utf8") as w:
    for num,key in enumerate(dict):
        entity_lis = dict[key]
        entity = ""
        for e in entity_lis:
            if e in delate:
                print(e)
                continue
            else:
                entity += e+",,,"
        entity = entity.strip(",,,")
        if entity:
            w.write(key+"\t"+entity+"\n")

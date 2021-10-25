#coding:utf-8
import logging
import json
import numpy as np
#import os
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='myResult.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


##{'id': 0, 'tag_seq': 'B-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O O B-产品H食品 I-产品H食品 O O O O O O O O O O O O O O O O O O O O O O O O O O B-产品H食品 I-产品H食品 O B-产品H食品 I-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O O O O O B-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O', 'entities': [['产品H食品', 0, 2], ['产品H食品', 16, 17], ['产品H食品', 44, 45], ['产品H食品', 47, 50], ['产品H食品', 67, 69]]}
#白激馍为豫东商丘特产，商丘的这个激馍很好吃，是商丘人民的最爱，商丘人民喜欢早上买一到两个狗肉或羊肉激馍，然后吃着去赶上班，每天都如此，白激馍是商丘早点中不败的经典！
#[(11, 14, '组织-企业机构')]
def combine(tmpmodel,tmpjieba,sentence):
    #tmpmodel = {'id': 0, 'tag_seq': 'B-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O O B-产品H食品 I-产品H食品 O O O O O O O O O O O O O O O O O O O O O O O O O O B-产品H食品 I-产品H食品 O B-产品H食品 I-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O O O O O B-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O', 'entities': [['产品H食品', 0, 2], ['产品H食品', 16, 17], ['产品H食品', 44, 45], ['产品H食品', 47, 50], ['产品H食品', 67, 69]]}
    #tmpjieba = [(11, 17, '组织-企业机构'),(42, 44, '组织-企业机构')]
    tmpModelEntities = tmpmodel['entities']
    #print('tmpModelEntities:',tmpModelEntities)
    #print('tmpjieba',tmpjieba)
    conM = []
    for labels in tmpModelEntities:
        #print('------------------')
        if isinstance(labels[2], str):
            labels = (labels[2], labels[0], labels[1])
        theStart = labels[1]
        theEnd = labels[2]
        #print('theStart:',theStart)
        #print('theEnd:',theEnd)
        #print('labels：',labels)
        #print('tmpjieba:',tmpjieba)
        tag = 0
        for theJieba in tmpjieba:
            #print('-jieba-')
            #print('theJieba:',theJieba)
            jieStart = theJieba[1]
            jieEnd = theJieba[2]
            #print('jieStart:',jieStart)
            #print('jieEnd:',jieEnd)
            if jieStart <= theStart <= jieEnd or jieStart <= theEnd <= jieEnd:
                #conM.append(labels)
                tag = 1
        if tag == 0:
            conM.append(labels)
            
    for theJieba in tmpjieba:
        tmpList = []
        tmpList.append(theJieba[1])
        tmpList.append(theJieba[2])
        tmpList.append(theJieba[0])
        conM.append(tmpList)
    
    #print('conM:',conM)
    tmpmodel['entities'] = conM
    tmpmodel['sentence'] = sentence
    #print('tmpmodel:',tmpmodel)
    
    return tmpmodel

def getData():
    logging.info('-------start------------------------------------------------------')
    with open('/Users/bytedance/Downloads/test_prediction.json','r') as f:
        con = f.readlines()
    logging.info('con:'+str(len(con)))
    #print('con:',len(con))
    re_model = [] #模型结果
    for line in con:
        re_model.append(json.loads(line))
    #print(re_model[0])
    with open('/Users/bytedance/Downloads/combine/sentence_test.txt','r') as f1: #句子
        sens = f1.readlines()
    #print(sens[0])
    re_jieba = np.load("/Users/bytedance/Desktop/jieba_result.npy",allow_pickle=True) 
    
    print('re_model:',len(re_model))
    print('re_jieba:',len(re_jieba))
    print('sentences:',len(sens))

    now_model = []
    now_jieba = []
    now_sen = []
    for d in re_jieba:
        now_jieba.append(d["result"])
        now_model.append(re_model[d["id"]])
        now_sen.append(sens[d["id"]])

    id = 541
    print(now_model[id])
    print(now_jieba[id])
    print(now_sen[id])
    
    result = []
    for i in range(len(now_sen)):
        result.append(combine(now_model[i],now_jieba[i],now_sen[i]))
    
    print('result:',len(result))
    np.save("/Users/bytedance/Downloads/combine/combine_result.npy", result)

if __name__ == "__main__":
    getData()
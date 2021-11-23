#coding:utf-8
#author: wuxiaoju
#firstdate: 2021.11.08
#lastdate: 2021.11.16
#function: Interaction of algorithm results

import os
from flask import Flask,request
import logging
import json
import time

import sys
sys.path.append(os.getcwd())
import numpy as np
#sys.path.append("..")
os.system('cp -r /opt/tiger/theCode/* /opt/tiger/out/')
os.system('cp -r /opt/tiger/pytos /opt/tiger/out/')
os.system('pip3 install tensorboard')
from worker.server import BYTENERServer
from pytos import tos
from pytos.tos import TosException
import datetime

#外接存储TOS
# 申请的Bucket名字
bucket = "cqc-algorithm-out"
# 申请的AccessKey
accessKey = "EDTYATWV43FJAN9GVV49"

logPath = 'mylog.log'

#加载模型，现在只能单个，没办法复用
server = BYTENERServer(model_name = "bert-base-chinese", load_checkpoint_path = "/opt/tiger/out/best_model.pth",label_num = 63)

#logging.basicConfig(level=logging.DEBUG,filename=logPath,filemode='a',format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logging.basicConfig(level=logging.DEBUG,filename=logPath,filemode='a',format='%(asctime)s - %(levelname)s: %(message)s')
logging.info('---------------------------------------')
logging.info('starting service...')

#if os.path.exists('./BERT-NER-Pytorch/README.md'):
#    pass
#else:
#    os.system('tar xvf model02.tar.gz')
#    os.system('cp -r BERT-NER-Pytorch/* /opt/tiger/out')
    #os.system('rm -rf model.tar.gz')

app = Flask(__name__)

def uploadDirectly(file_path):
    client = tos.TosClient(
        bucket,
        accessKey,
        cluster="default",
        timeout=10,
        addr_family="v6"
    )
    current = str(datetime.date.today())
    #current = time.strftime("%Y-%m-%d__%H_%M_%S", time.localtime())
    key = file_path[:-4] + '_' + current + '.log'
    buffer = bytes()
    with open(file_path, "rb") as f:
        line = f.readline()
        while line:
            buffer = buffer + line
            line = f.readline()
    try:
        client.put_object(key, buffer)
    except TosException as e:
        logging.info("put object error: ", e)


def dealData(sentencePre):
    timeStart = time.time()
    #pre_result,status = dealData(theData['content'])
    pre_result = []
    status = 'Did not make predictions.'
    try:
        dicForPre = {}
        dicForPre['sentence'] = sentencePre.replace('\n','').replace('\r','')
        dicForPre['id'] = 0
        predict_data = []
        predict_data.append(dicForPre)
        results = server.predict(predict_data)
        #results: [[('产品-交通工具', 13, 15), ('地点-公共场所', 19, 23)]]
        logging.info(results)
        for a in results[0]:
            dicTmp = {}
            label = a[0]
            start = a[1]
            end = a[2] + 1
            text = sentencePre[start:end]
            dicTmp['label'] = label
            dicTmp['start'] = start
            dicTmp['end'] = end
            dicTmp['text'] = text
            pre_result.append(dicTmp)
        #pre_result = results
        status = 'Run successfully.'
    except Exception as e:
        logging.info(e)
    '''try:
        title = 'sentence'
        singleByteSen = [] #单词成句的结果
        for a in title:
            singleByteSen.append(a + ' O\n')
        singleByteSen.append('\n')
        sentence = sentencePre.replace('\n','').replace('\r','')
        for b in sentence:
            singleByteSen.append(b + ' O\n')
        singleByteSen.append('\n')
        with open('/opt/tiger/out/datasets/cner/test.char.bmes','w') as fw:
            fw.writelines(singleByteSen) #写成待预测文件
        os.system("rm -rf ./datasets/cner/cached_soft-test_*") #删除原来的cach

        #跑预测
        os.system('bash /opt/tiger/out/scripts/run_ner_softmax.sh')
        
        resultfile = '/opt/tiger/out/outputs/cner_output/bert/test_prediction.json'
        os.path.exists(resultfile) #判断结果是否存在
        status = 'Run successfully.'
        conModel = [] #预测结果
        with open(resultfile,'r') as f:
            conModel = f.readlines()
        logging.info('conModel:',len(conModel))
        reModel = eval(conModel[0])['entities']
        #res_model = []
        #for line in conModel:
            #res_model.append(json.loads(line))
        #[{'id': 0, 'tag_seq': 'I-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O O I-产品H食品 I-产品H食品 O O O O O O O O O O O O O O O O O O O O O O O O O O I-产品H食品 I-产品H食品 O I-产品H食品 I-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O O O O O I-产品H食品 I-产品H食品 I-产品H食品 O O O O O O O O O O O O', 'entities': []}]
        #处理成输出的结果格式
        
        for a in reModel:
            dicTmp = {}
            label = a[0]
            start = a[1]
            end = a[2] + 1
            text = sentencePre[start:end]
            dicTmp['label'] = label
            dicTmp['start'] = start
            dicTmp['end'] = end
            dicTmp['text'] = text
            pre_result.append(dicTmp)

        
    except Exception as e:
        logging.info(e)
    '''
    timeEnd = time.time()
    usingTime = timeEnd - timeStart
    logging.info('using time:'+str(usingTime))
    return pre_result,status


def checkInKeys(theKey,theData,theError):
    dic_result_tmp = {}
    if theKey in theData.keys():
        dic_result_tmp[theKey] = theData[theKey]
    else:
        dic_result_tmp['error'] = theError + '\n missing ' + theKey + '.'
    return dic_result_tmp

@app.route('/hello',methods=['POST'])
def exchangeData(): #数据出口
    logging.info('---------------------------------------')
    dic_result_all = {} #存储多个句子的结果
    dic_result_all['results'] = []
    logging.info('here exchange the data.')
    theData_all = request.get_json() #获取json数据
    #for theData_keys in theData_all.keys():
        #theData = theData_all[theData_keys]
    
    #处理单条数据回到最开始的格式定义
    tmpSingle = []
    if isinstance(theData_all,dict):
        tmpSingle.append(theData_all)
        theData_all = tmpSingle
    
    for data in theData_all:
        theData = data
        keys = theData.keys() #获取所有的键
        #logging.info(theData)
        dic_result = {} #判断键是否存在
        dic_result['error'] = '' #先新建一个键
        dic_result.update(checkInKeys('content_id',theData,dic_result['error']))
        dic_result.update(checkInKeys('datatype',theData,dic_result['error']))
        dic_result.update(checkInKeys('modeltype',theData,dic_result['error']))
        dic_result.update(checkInKeys('labels',theData,dic_result['error']))
        #dic_result.update(checkInKeys('content',theData))
        try:
            if 'content' in keys : #处理content
                if theData['modeltype'] == 'predict' : #做预测
                    dic_result['content'] = theData['content'] #保存原数据
                    logging.info('get the content.') 
                    pre_result,status = dealData(theData['content'])
                    dic_result['pre_result'] = pre_result #预测结果
                    dic_result['status'] = status #预测状态
                elif theData['modeltype'] == 'train' :
                    pass
                else:
                    dic_result['error'] = dic_result['error'] + '\n incorrect modeltype.'
            else:
                dic_result['error'] = dic_result['error'] + '\n missing content_id.'
            logging.info('finish the content')
        except Exception as e:
            logging.info(e)
        logging.info(dic_result)
        #dic_result_all[theData_keys] = dic_result
        dic_result_all['results'].append(dic_result)
    uploadDirectly(logPath)

    #处理单条数据回到最开始的格式定义
    if tmpSingle != []:
        return dic_result_all['results'][0]
    else:
        return dic_result_all

@app.route('/')
def index(): #首页
    logging.info('---------------------------------------')
    logging.info('Welcome to the algorithm model results.')
    return 'Welcome to the algorithm model results.'

@app.route('/exchange',methods=['POST'])
def hello_world(): #返回传进来的数据
    logging.info('---------------------------------------')
    logging.info('hello')
    myJson = request.get_json()
    logging.info(str(myJson))
    return myJson

PORT0 = int(os.getenv('PORT0'))
while 1:
    app.run(host='0.0.0.0',port=PORT0)
    #app.run(host='::',port=PORT0)
#执行语句：
#（1）export FLASK_APP=20211103/app.py
#（2）flask run
#（3）在浏览器输入http://127.0.0.1:5000/
#（4）可以看到Hello, World!
#（5）flask run --host=0.0.0.0，可以监听所有公开IP

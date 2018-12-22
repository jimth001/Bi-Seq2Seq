# Bi-Seq2Seq
An implementation of "Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems"

需要准备的数据：  
'./data/train.query',#原始英文语料。处理时会使用nltk分词。  
'./data/train.reply',  
'./data/train.target',  
'./data/val.query',  
'./data/val.reply',  
'./data/val.target',  
'./data/test.query',  
'./data/test.reply',  
'./data/test.target',#生成时实际不会用到，但是要有，都是用的同样的数据预处理。  
'./data/embedding'#fasttext的格式  

处理流程：  
先执行  
preprocess()#生成./data/train.pkl,test.pkl,val.pkl  
train_onehotkey(batch_size=32)#模型保存在./model下  
再执行  
generate_batches(model_path='./model/epoch.10.model',batch_size=32)#生成结果：./output/result  

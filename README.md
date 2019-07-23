# Bi-Seq2Seq
An implementation of "Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems".
This code serves as a baseline of "Response Generation by Context-aware Prototype Editing"(https://arxiv.org/abs/1806.07042). 


##Code:

run preprocess() to generate some pickle files for training. (./data/train.pkl, ./data/test.pkl, ./data/val.pkl) 
run train_onehotkey(batch_size=32) for training. Models are saved under "./model".  
run generate_batches(model_path='./model/epoch.10.model',batch_size=32) to generate results(./output/result). 


##data preparing：  

'./data/train.query',(raw querys, line by line)
'./data/train.reply',  
'./data/train.target',  
'./data/val.query',  
'./data/val.reply',  
'./data/val.target',  
'./data/test.query',  
'./data/test.reply',  
'./data/test.target', (If you don't have a target file, you can let query as the target to run preprocess())
'./data/embedding' (fasttext's format)  


##Dataset

You can contact the authors of “Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems” (https://arxiv.org/abs/1610.07149) if you are trying to reproduce this work. You can also get a dataset to run this code at https://github.com/MarkWuNLP/ResponseEdit.

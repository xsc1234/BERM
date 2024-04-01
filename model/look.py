# from transformers import RobertaTokenizer
# import torch
# import numpy as np
# tok = RobertaTokenizer.from_pretrained('/data/home/xushicheng/roberta')
# text = 'he is a good boy. and he is good. wao'
# enc = tok.encode(text.strip(), add_special_tokens=True)
# print(tok.tokenize(text.strip(),add_special_tokens=True))
# print(enc)
# tensor_enc = torch.Tensor(enc)
# tensor_enc = tensor_enc.unsqueeze(0)
# # juhao = (tensor_enc == 4).nonzero()
# # juhao
# # for i in range(juhao.shape[0]):
# #     print(juhao[i])
# import torch
# x = torch.empty(5, 45,3)
# print(x.shape)
# print(x[0][3:9].shape)
# print(x[0][3:9])
# # print(torch.sum(x[0][3:9],dim=0,keepdim=True))
# import joblib
#
# dic_idf = joblib.load('/data/xushicheng/ir_dataset/msmarco/msmarco/word_tf')
# print(dic_idf['gullah'])
import faiss
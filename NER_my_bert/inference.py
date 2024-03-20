# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:50:30 2024

@author: gimal
"""

import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast, BertModel
import pandas as pd
from bert import BERT, BERT_CONFIG
from utils import Config, DeployConfig
import torch.nn as nn
from task import NERTask

args = Config() 
tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
args = DeployConfig(pretrained_model_name=args.model_name,
                    downstream_model_dir=args.downstream_model_dir,
                    max_seq_length=64)

#학습한 모델 체크포인트 불러오기
fine_tuned_model_ckpt = torch.load(args.downstream_model_checkpoint_fpath,
                                   map_location=torch.device("cpu"))

#모델 파라미터 가져오고
pretrained_model_config = BertConfig.from_pretrained(args.pretrained_model_name,
                                                     num_labels = fine_tuned_model_ckpt["state_dict"]["to_output.bias"].shape.numel())
#torch.Size[13] numel = 13
my_config = BERT_CONFIG(hg_config=pretrained_model_config)
hg_model = BertModel.from_pretrained("klue/bert-base",num_labels = 13)

model = NERTask.load_from_checkpoint("/Users/ki_mimang/Desktop/NER_my_bert/pl-ner/p51wqwmq/checkpoints/epoch=19-step=2640.ckpt",model=hg_model,args = pretrained_model_config,
                                     strict=False)

model.eval()
labels = [
    'B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O'
    ]
#레이블을 알아보기 쉽게 한국어 단어로 변경해줌.
id_to_label = {}
for idx, label in enumerate(labels):
  if "PS" in label:
    label = "인명"
  elif "LC" in label:
    label = "지명"
  elif "OG" in label:
    label = "기관명"
  elif "DT" in label:
    label = "날짜"
  elif "TI" in label:
    label = "시간"
  elif "QT" in label:
    label = "수량"
  else:
    label = label
  id_to_label[idx] = label



#인퍼런스 함수.
def inference_fn(sentence):
    tokenn = []
    for i in sentence:
        tokenn.append(i)
        
    inputs = tokenizer(
        [tokenn],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
        is_split_into_words = True
    )

    with torch.no_grad():
        #13개의 예측치가 나오겠지.

        logits = model(torch.tensor(inputs["input_ids"]),torch.tensor(inputs["attention_mask"]))
        probs = logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[pred.item()] for pred in preds]
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                
                token_result = {
                    "토큰": token,
                    "태그": predicted_tag,
                    "확률": str(round(top_prob[0].item(), 4)),
                }
                result.append(token_result)
        df = pd.DataFrame(result)
    return df

inference_fn("2024년엔 추성훈 선수가 우승 후보 입니다.")

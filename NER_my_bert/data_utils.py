# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from transformers import DataCollatorForTokenClassification

class DataPreProcessing():
    def __init__(self,tokenizer, datasets):
        self.tokenizer = tokenizer
        self.datasets = datasets

    def tokenize_and_align_labels(self,examples, label_all_tokens=True):
        tokenized_inputs = self.tokenizer(examples["tokens"],padding = True ,truncation=True,is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
          word_ids = tokenized_inputs.word_ids(batch_index= i)
          previous_word_idx = None
          label_ids = []
          #CLS나 SEP 토큰같은 특수 토큰은 무시하기위해 -100으로 처리 해주는작업.
          for word_idx in word_ids:
            if word_idx is None:
              label_ids.append(-100)
            elif word_idx != previous_word_idx:
              label_ids.append(label[word_idx])
            else:
              label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
          labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_data(self):
        tokenized_datasets = self.datasets.map(self.tokenize_and_align_labels, batched=True) #위에서 선언한 함수 이용해서 매핑해줌.
        tokenized_datasets = tokenized_datasets.remove_columns(self.datasets["train"].column_names)
        return tokenized_datasets


    def train_test_eval_tokenized_datasets_func(self):
        tokenized_datasets = self.tokenize_data()
        train_test_tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size = 0.2)
        train_tokenized_datasets = train_test_tokenized_datasets["train"]
        test_tokenized_datasets =  train_test_tokenized_datasets["test"]
        eval_tokenized_datasets = tokenized_datasets["validation"]
        return train_tokenized_datasets, test_tokenized_datasets, eval_tokenized_datasets




    def data_collator_func(self,tokenizer):
        data_collator = DataCollatorForTokenClassification(tokenizer = tokenizer)
        return data_collator







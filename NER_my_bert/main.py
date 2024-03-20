# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:34:19 2024

@author: gimal
"""
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from transformers import BertModel, BertConfig


from data_utils import DataPreProcessing
from utils import Config
from task import NERTask
from trainer import get_trainer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
def get_loaders(train_tokenized_datasets,
                test_tokenized_datasets,
                eval_tokenized_datasets,
                ):
    data_collator = data.data_collator_func(tokenizer)
    train_dataloader = DataLoader(
        train_tokenized_datasets,
        shuffle = True,
        collate_fn = data_collator,
        batch_size = 32,
    )
    test_dataloader = DataLoader(
        test_tokenized_datasets,
        shuffle = True,
        collate_fn = data_collator,
        batch_size = 32
    )
    eval_dataloader = DataLoader(
        eval_tokenized_datasets,
        collate_fn = data_collator,
        batch_size = 32,
    )
    return train_dataloader, test_dataloader, eval_dataloader


if __name__ == "__main__":
    pl.seed_everything(1234)
    logger = WandbLogger(name="pl-ner",project="pl-ner")
    #이거 main method행.
    klue = load_dataset("klue","ner")
    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
    #데이터 전처리 해서 받아옴
    data = DataPreProcessing(tokenizer, klue)
    train_tokenized_datasets, test_tokenized_datasets, eval_tokenized_datasets = data.train_test_eval_tokenized_datasets_func()
    #데이터 로더 준비
    train_dataloader, test_dataloader, eval_dataloader = get_loaders(train_tokenized_datasets,
                                                                     test_tokenized_datasets,
                                                                     eval_tokenized_datasets)
    #arguments 준비
    args = Config()

    #label 수
    num_labels = 13

    #klue/bert-base의 pretrained_model_config 불러오기
    pretrained_model_config = BertConfig.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name,num_labels = 13)

    #task, trainer 선언
    task = NERTask(model, pretrained_model_config)

    trainer = pl.Trainer(
                            num_sanity_val_steps=0,
                            max_epochs=20, 
                            strategy = DDPStrategy(find_unused_parameters=True),
                            callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                            logger = logger
                        )

    #학습
    trainer.fit(task, train_dataloader, eval_dataloader)

    #채점
    trainer.test(task,test_dataloader)


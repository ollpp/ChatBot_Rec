# 이 파일은 csv 경로를 받아와서 모델 학습 후 pkl파일로 생성하는 파이썬파일입니다.
# csv 데이터는 Q,A,label 형식으로 설정이 되어 있어야 합니다. 구분자는 "," 입니다

import math
import numpy as np
import pandas as pd
import random
import re
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

from transformers.optimization import AdamW
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel


## 문장 토큰, 마스킹 설정
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
# SENT = '<unused1>'


## 허깅페이스 transformers 에 등록된 사전 학습된 koGTP2 토크나이저를 가져온다.
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2", 
    bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK,)

## 챗봇 데이터를 처리하는 클래스 생성
class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats
        # self.first = TRUE   ## 연속성 대화 데이터셋 사용 경우
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        # self.sent_token = SENT
        #self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        # self.pad = PAD
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):  # chatbotdata 의 길이를 리턴한다.
        return len(self._data)

    def __getitem__(self, idx):  # 로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx]
        q = turn["Q"]  # 질문을 가져온다.
        q = re.sub(r"([?.!,])", r" ", q)  # 구둣점들을 제거한다.

        a = turn["A"]  # 답변을 가져온다.
        a = re.sub(r"([?.!,])", r" ", a)  # 구둣점들을 제거한다.

        # 라벨링 칼럼을 사용할 경우
        # sent = str(turn['label'])

        q_toked = self.tokenizer.tokenize(self.q_token + q)
        # q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token + sent)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        #질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)


        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_toked[1:]

        # 데이터가 연속성일 경우 사용
        # if self.first:
        #     logging.info("contexts : {}".format(q))
        #     logging.info("toked ctx: {}".format(q_toked))
        #     logging.info("response : {}".format(a))
        #     logging.info("toked response : {}".format(a_toked))
        #     logging.info('labels {}'.format(labels))
        #     self.first = False

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.    
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)


def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)
    # return torch.cuda.LongTensor(data), torch.cuda.LongTensor(mask), torch.cuda.LongTensor(label)


base_model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
pklmodel_name = base_model

## (데이터 경로, 기존 돌릴 피클모델 이름, 새로 저장할 피클모델 이름) -> file은 csv만 받습니다. pkl 파일명은 string 형태로 받습니다.
def run_model(csv_path, pklmodel, new_pkl_name):

    ## train set 지정
    model = pklmodel

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = ChatbotDataset(csv_path, max_len=40)
    train_dataloader = DataLoader(
        train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch)

    # model.to(device)
    model.train()


    ## Hyperparams   ----- 파라미터 조정 -----
    epoch = 10
    learning_rate = 3e-5
    Sneg = -1e18
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)


    ## training_step, forward   ----- 모델 학습 -----
    for epoch in range(epoch):
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            token_ids, mask, label = batch
            out = model(token_ids)
            out = out.logits      #Returns a new tensor with the logit of the elements of input
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
            # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
            avg_loss = loss.sum() / mask.sum()
            avg_loss.backward()
            # 학습 끝
            optimizer.step()


    ## save model   ----- 학습된 모델을 피클파일로 생성 -----
    torch.save(model, new_pkl_name + '.pt')


    return 'done'
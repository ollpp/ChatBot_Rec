# 제일 최신 pkl 파일 불러오기

# string 파라메타 받기
# 모델 3번 반복해서 결과값 3개 얻기
# return {
#         '1' : '챗봇 리턴 답변'
#         '2' : '챗봇 리턴 답변'
#         '3' : '챗봇 리턴 답변'
#         }


import os
import torch
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


def comu(model, input):
    model_pkl = model
    input_str = input
    ans_dict = {}
    cnt = 0

    with torch.no_grad():
        while cnt < 3:
            q = input_str
            a=""
            while 1:
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + A_TKN + a)).unsqueeze(dim=0)
                # input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
                pred = model_pkl(input_ids)
                pred = pred.logits
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")

            ## 같은 답변 나올 경우 다시 생성
            if a not in ans_dict.values():
                ans_dict[cnt] = a
                cnt += 1

    return ans_dict



def main():

    print('Chatbot is on')

    path = os.getcwd().replace('\\','/') + '/model'
    model_lst = os.listdir(path)
    model_lst.sort(reverse=True)
    model_nm = path + '/' + model_lst[0]
    model_pkl = torch.load(model_nm)

    # !!!대화 서비스에서 input str 받아오는 로직 정의 필요!!!
    input_str = input('대화를 입력하세요 : ')

    ans_dict = comu(model_pkl, input_str)
    print(ans_dict)
    
    # !!!ans_lst 어디로 보낼지 정의 필요!!!!

    return 'done'


if __name__ == '__main__':
    main()

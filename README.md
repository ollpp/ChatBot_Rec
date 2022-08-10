
# ChatBot12
Chatbot for senior


# 1. 프로젝트 개요
    
    시니어들을 위한 데이트앱에서 공통 관심사에서 원활한 대화를 위해 AI가 맞춤형 대화 스크립트를 생성 및 추천해주는 챗봇 시스템
    


# 2. 모델

### 데이터셋
1. Open Source 챗봇 데이터셋
    [https://github.com/songys/Chatbot_data]

2. AI Hub (40대 이상 데이터만 추출)
    [https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=543]

3. 웰니스 데이터셋
    [https://github.com/nawnoes/WellnessConversation-LanguageModel]


### 모델 Architecture
1. Define Token
    *  Q_TKN, A_TKN, BOS, EOS, PAD, MASK, SENT, PAD

2. Define Chatbot Class
    *  len : 길이 함수
    *  getitem : 데이터 로더 함수
        1. 데이터 전처리 - regex
        2. Tokenization
            *  q_toked(질문 시작 토큰, 질문 데이터)
            *  a_toked(답변 시작 토큰, 답변 데이터, 대화 종료 토큰)
        3. max length 설정
        4. labeling - 답변지 생성 (질문 마스킹, embedded 답변 데이터, max_len 만큼 남은 길이 패딩)
        5. masking - labels 를 mask화 (0,0,0, …, 1,1,1, …, 0,0,0,…)
        6. Ids
            *  labels_id (9(MASK<unused0>),9,9,…, a1,a2,a3,…1(EOS</s>),3(PAD<pad>),3,3,…)
            *  token_ids (2(Q_TKN<usr>), q1,q2,q3,…,4(A_TKN<sys>),a1,a2,a3,…1(EOS</s>),3(PAD<pad>),3,3,…)

3. batch - batch 합쳐주는 함수

4. model - 모델 돌려서 학습해주는 함수
    *  Hyper-parameters 
        1. epoch : 10
        2. learning_rate : 3e-5
        3. Sneg : 1e18
        4. criterion : CrossEntropyLoss(다중분류함수)
        5. optimizer : Adam



# 3. 서비스 설명

### 디렉토리 구조
1. model : 모델 피클화 해서 저장하는 경로
2. data : 추가될 데이터 저장하는 경로
3. `main.py` : 가장 최신 pkl 모델 불러와서 스크립트 추천 목록 생성 파일
4. `kogpt2_model.py` : 가장 최신 데이터(csv) 불러와서 모델 학습 후 pkl 파일로 생성
5. `batch.py` : kogpt2_model.py 실행 및 model 폴더와 data 폴더 관리
6. `gpt2_modeling.ipynb` :

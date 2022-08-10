# 이 파일은 model.py 파일을 실행시키는 파이썬 파일입니다.
# model.py에 들어가는 함수에 들어가는 파라메타 중 csv데이터,파일명(날짜)을 여기서 생성하여 전달.
# csv파일은 data폴더에 존재하는 csv파일로 제일 최신파일을 읽어들여 csv데이터 파라메터로 전달.
# model폴더 및 data폴더의 파일 갯수가 5개 초과시 최신기준 5개를 제외한 모든파일을 삭제

from kogpt2_model import run_model
from datetime import datetime
from pathlib import Path
import os
from collections import deque


## 현재 파일의 절대경로
PATH= os.path.dirname(os.path.abspath(__file__)).replace('\\','/')
# print('Path : ', PATH)

## csv 저장할 폴더 이름
DATA_FOLDER = 'data'
## pkl 저장할 폴더 이름
PKL_FOLDER = 'model'
## 폴더내에 남길 csv 파일 갯수
CSV_COUNT = 5
## 폴더내에 남길 pkl 파일 갯수
MODEL_COUNT = 5

## 폴더에 존재하는 파일로 제일 최신파일을 읽어들여 경로를 파라메터로 전달
def read_file_list(folder):
    data_folder_path = PATH + '/' + folder
    file_list = sorted(Path(data_folder_path).iterdir(), key=os.path.getmtime, reverse=True)
    file_list = [str(i).replace('\\','/') for i in file_list]
    return file_list

## kogpt2_model.py에 들어가는 함수에 들어가는 파라메타 중 생성될 파일명(날짜)을 여기서 생성하여 전달
def pkl_file_name():
    save_folder_path = PATH + '/' + PKL_FOLDER
    save_file_name = datetime.now().strftime("%y%m%d%H%M")
    return save_folder_path+'/'+save_file_name


## model폴더 및 data폴더의 파일 갯수가 5개 초과시 최신기준 5개를 제외한 모든파일을 삭제
def delete_file():
    #data 폴더의 csv 파일 삭제
    csv_file_list = deque(read_file_list(DATA_FOLDER))
    while len(csv_file_list)>CSV_COUNT:
        os.remove(csv_file_list.pop())
    
    #model 폴더의 csv 파일 삭제
    model_file_list = deque(read_file_list(PKL_FOLDER))
    while len(model_file_list)>MODEL_COUNT:
        os.remove(model_file_list.pop())


## 실행
if __name__ =="__main__":
    #리스트 순서 : 최신순, 리스트 첫번째가 최신 csv파일
    data_path_list = read_file_list(DATA_FOLDER)
    pkl_path_list = read_file_list(PKL_FOLDER)
    pkl_name = pkl_file_name()
    run_model(data_path_list[0],pkl_path_list[0],pkl_name)
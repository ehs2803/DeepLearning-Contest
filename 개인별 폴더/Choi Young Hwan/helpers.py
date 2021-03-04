import numpy as np
import csv


# csv 파일의 경로를 받아 파일의 내용을 읽어들이는 함수
# 이미지와 각 이미지의 라벨값을 반환
def read_csv(path):
    width = 34          # 이미지의 너비
    height = 26         # 이미지의 높이
    dims = 1            # 채널 : 흑백

    # scv 파일을 딕셔너리 형태로 읽음
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # imgs 변수는 모든 이미지들의 numpy 배열
    # tgs 변수는 모든 이미지들의 라벨 값이 들어있는 numpy 배열
    imgs = np.empty((len(list(rows)), height, width, dims), dtype=np.uint8)
    tgs = np.empty((len(list(rows)), 1))

    # 리스트를 이미지 형태로 변환
    for row, i in zip(rows, range(len(rows))):
        img = row['image']                              # image 행의 정보 : img 변수에 저장
        img = img.strip('[').strip(']').split(', ')     # 쉼표를 기준으로 문자열 분리, 대괄호 제거
        im = np.array(img, dtype=np.uint8)              # img 변수의 데이터를 8bit 양의 정수 numpy 배열 형태로 저장
        im = im.reshape((height, width))                # 이미지의 크기를 (34, 26) 으로 저장
        im = np.expand_dims(im, axis=2)                 # 채널 차원 추가 (2차원 데이터를 3차원 데이터로 변환)
        imgs[i] = im                                    # imgs 의 i 번째 행에 이미지 저장

        # 눈을 뜬 상태의 라벨값은 1, 감은 상태의 라벨값은 0
        tag = row['state']
        if tag == 'open':
            tgs[i] = 1
        else:
            tgs[i] = 0

    # 데이터셋 셔플
    index = np.random.permutation(imgs.shape[0])
    imgs = imgs[index]
    tgs = tgs[index]

    # 이미지들과 각각의 라벨값을 반환
    return imgs, tgs
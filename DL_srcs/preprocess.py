from helpers import *
import matplotlib.pyplot as plt
import os
import seaborn as sns               # 데이터 분포 시각화 모듈
from sklearn.model_selection import train_test_split        # 데이터 분할을 위한 함수 import


# 데이터 확인
# 데이터셋의 경로  저장
base_path = 'dataset'

X, y = read_csv(os.path.join(base_path, 'dataset.csv'))

print(X.shape, y.shape)             # 이미지와 라벨의 개수 확인 (각각 2784개)

# 사용자의 오른쪽 눈 출력
plt.figure(figsize=(12, 10))
for i in range(50):
    plt.subplot(10, 5, i+1)
    plt.axis('off')
    plt.imshow(X[i].reshape((26, 34)), cmap='gray')

# 라벨의 분포 히스토그램 출력
sns.distplot(y, kde=False)


# 데이터 전처리
n_total = len(X)
X_result = np.empty((n_total, 26, 34, 1))

for i, x in enumerate(X):
    img = x.reshape((26, 34, 1))
    X_result[i] = img

# 검증데이터의 비율은 10%의 비율로해서 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(X_result, y, test_size=0.1)

# 데이터 개수 확인
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# npy 파일의 형태로 훈련, 검증 데이터 저장
np.save('dataset/x_train.npy', x_train)
np.save('dataset/y_train.npy', y_train)
np.save('dataset/x_val.npy', x_val)
np.save('dataset/y_val.npy', y_val)

# 훈련 데이터와 검증 데이터 확인
plt.subplot(2, 1, 1)
plt.title(str(y_train[0]))
plt.imshow(x_train[0].reshape((26, 34)), cmap='gray')
plt.subplot(2, 1, 2)
plt.title(str(y_val[4]))
plt.imshow(x_val[4].reshape((26, 34)), cmap='gray')

# 훈련 데이터의 라벨 분포 히스토그램 출력
sns.distplot(y_train, kde=False)

# 검증 데이터의 라벨 분포 히스토그램 출력
sns.distplot(y_val, kde=False)

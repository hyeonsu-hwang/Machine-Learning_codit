#### 머신러닝

## numpy로 행렬 사용하기
import numpy as np

# 이차원배열(리스트안의 리스트) 활용 4 * 3 행렬
A = np.array([
    [1, -1, 2],
    [3, 2, 2],
    [4, 1, 2],
    [7, 5, 6]
])

A

B = np.array([
    [0, 1],
    [-1, 3],
    [5, 2]
])

B

# 행렬의 랜덤한 값 넣기
C = np.random.rand(3, 5) # 3 by 5 랜덤(0과 1사이의 값) 행렬 만들기

# 모든 값이 0인 행렬
D = np.zeros((2, 4))  # 괄호를 한번 더 써 줘야한다

# 행렬에서 원소 받아오기
A

A[0][2] # 1행 3열 원소 가져오기 (파이썬에서는 0부터시작)

A[0][5] # 범위 넘어서면 오류


## 요소별 곱하기(Element - wise Multiplication)

import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [0, 1, 2],
    [2, 0, 1],
    [1, 2, 0]
])

A * B

## numpy로 행렬 연산하기
import numpy as np

A = np.array([
    [1, -1, 2],
    [3, 2, 2],
    [4, 1, 2]
])

B = np.random.rand(3, 3)

# 둘다 3 by 3 행렬
A
B

# 행렬 덧셈
A + B

# 스칼라 곱
5 * A

# 행렬과 행렬의 곱(내적) -> np.dot(~, ~) 또는 ~ @ ~
np.dot(A, B)
A @ B

# 요소별 곱 -> ~ * ~
A * B

# 여러 연산 섞어서 사용하기(우선순위대로)
A @ B + (A + 2 * B)



## numpy로 전치, 단위, 역행렬 사용하기
import numpy as np

A = np.array([
    [1, -1, 2],
    [3, 2, 2],
    [4, 1, 2]
])

# 전치행렬 -> np.transpose(~), ~.T
A_transpose = np.transpose(A)

A.T

# 단위행렬 -> np.identity(상수)
I = np.identity(3)

A @ I

# 역행렬 -> np.linalg.pinv(~)
A_inverse = np.linalg.pinv(A) # pinv함수는 역행렬이 없는 행렬도 비슷한 효과를 낼수있는 행렬을 만들어준다

A @ A_inverse # 대각선 원소는 전부 1 나머지 원소는 복잡한값 6.66133815e-16 , 6의 -16승 0.0000000 0에 엄청 가까운수 



###############################################################################################################
## 경사 하강법 구현 시각화
import numpy as np
import matplotlib.pyplot as plt

def prediction(theta_0, theta_1, x):
    return theta_0 + theta_1 * x
    
def prediction_difference(theta_0, theta_1, x, y):
    return prediction(theta_0, theta_1, x) - y
    
def gradient_descent(theta_0, theta_1, x, y, num_iterations, alpha):
    m = len(x)
    cost_list = [] # 손실을 저장하는 리스트 경사하강한번 할때 마다 손실을 여기에 저장
    
    for i in range(num_iterations):  # 정해진 번만큼 경사 하강을 한다
        error = prediction_difference(theta_0, theta_1, x, y)  # 모든 오차가 계산된 벡터
        cost = (error@error) / (2*m) # 손실함수
        cost_list.append(cost)
        
        theta_0 = theta_0 - alpha * error.mean()
        theta_1 = theta_1 - alpha * (error * x).mean()
        
        # 그래프 200번하기는 부담 10번에 1번씩만 작성
        if i % 10 == 0:
            # 가설함수가 계선되는 모습 시각화
            plt.scatter(house_size, house_price)
            # 가설함수 x축 집 크기, y축은 예측된 집 가격
            plt.plot(house_size, prediction(theta_0, theta_1, x), color='red')
            plt.show()
        
    return theta_0, theta_1, cost_list
    
    
house_size = np.array([0.9, 1.4, 2, 2.1, 2.6, 3.3, 3.35, 3.9, 4.4, 4.7, 5.2, 5.75, 6.7, 6.9])
house_price = np.array([0.3, 0.75, 0.45, 1.1, 1.45, 0.9, 1.8, 0.9, 1.5, 2.2, 1.75, 2.3, 2.49, 2.6])

# theta 값들 초기화 (아무 값이나 시작함)
theta_0 = 2.5
theta_1 = 0

# 학습률 0.1로 200번 경사 하강
theta_0, theta_1 , cost_list = gradient_descent(theta_0, theta_1, house_size, house_price, 200, 0.1)

plt.plot(cost_list) # 경사하강을 할때 마다 손실이 어떻게 변하는지 확인가능
# 경사하강을 반복할수록 손실이 줄어든다


## 손실함수 설명
cost = (error@error) / (2*m) 
'''
첫 번째 파라미터가 1차원이면 (벡터면) 기존 n차원 앞에 1를 붙여주어 1 x n 차원으로 만들고, 계산이 끝난 후 앞에 1을 제거해주고,
두 번째 파라미터가 1차원이면 (벡터면) 기존 n차원 위에 1을 붙여주고 n x 1 차원으로 만들고, 계산이 끝난 후 뒤에 1을 제거해준다

라고 나와 있습니다. @은  그냥 matmul을 쉽게 사용하는 연산자인데요. 
그렇기 때문에 error @ error를 했을 때. 사실은 벡터가 1xn인 행렬과 nx1인 행렬로 바뀌어서 곱해져서 
하나의 스칼라 값이 나오는 겁니다. 
결국 error x error.T를 한 거라고 생각하시면 되는데요. 
계산 결과값은 모든 원소의 제곱의 합이 나옵니다.'''


###### scikit-learn (sklearn) 소개 및 데이터 준비
from sklearn.datasets import load_boston # 머신러닝 교육용 데이터셋 (보스턴 집값 데이터셋)
import pandas as pd
boston_dataset = load_boston()

# DESCR -> 데이터셋의 정보 보기 
boston_dataset.DESCR 
print(boston_dataset.DESCR)

# .frature_names -> 속성이름 확인하기
boston_dataset.feature_names

# .data 입력변수를 행렬로 보기
boston_dataset.data

# 행렬의 모양 출력
boston_dataset.data.shape # 506개의 집데이터가 있고 각 집은 13개의 속성이 있다

# 목표변수 출력
boston_dataset.target # 506개의 집 가격

# 목표변수 모양 출력
boston_dataset.target.shape # 차원이 506인 벡터

# 넘파이 배열을 판다스 데이터프레임으로 만들기
# pd.DataFrame(행렬 데이터, columns= 속성이름)
x = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
x

# 입력변수 1개일때 공부중이니 열1개 골라서 분석하기 -> 입력변수 준비
x = x[['AGE']] # df[]  안에 리스트를 넣게 되면 해당 리스트에 속하는 모든 컬럼들이 인덱싱될 수 있어서

# [['CRIM']] 은 DataFrame 을 반환하고
# ['CRIM'] 은 Series 를 반환합니당.

# 목표변수 정리 데이터프레임으로 만들기
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])



#######
## scikit-learn 데이터 셋 나누기
# trainnig set , test set 나누기
from sklearn.datasets import load_boston # 머신러닝 교육용 데이터셋 (보스턴 집값 데이터셋)
from sklearn.model_selection import train_test_split # 데이터셋 나누는 함수

# 데이터 셋 나누기
# train_test_split(입력변수, 목표변수, test_size=0.2 , random_state=5)
# test_size=0.2 -> 전체 데이터중 20%만 골라서 test셋으로 사용하고 나머지 80%는 trainning셋으로 사용
# random_state=5 -> 그 20%를 어떻게 고를지 정하는 파라미터 옵셔널 파라미터이기 떄문에 안넘겨줘도 된다
# 안넘겨주면 실행때 마다 매번 랜덤한 20%를 test set으로 새롭게 고르게된다
# 어떤 정수값을 넘겨주면 매번 20%의 똑같은 값을 고르게 된다. 아무정수나 써도 된다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#######
## scikit-learn으로 선형 회귀 쉽게 하기
from sklearn.linear_model import LinearRegression

# 선형회귀 함수 
model = LinearRegression()

# trainning set 학습시키기
model.fit(x_train, y_train)

# 세타1의 값
model.coef_

# 세타0의 값
model.intercept_

# f(x) = 31.04617413 -0.12402883x -> 우리가 구한 최적선

# 최적선 평가하기 trainning 셋으로 구한 최적선을 test셋으로 test
# model.predict(test셋 입력변수)
y_test_prediction = model.predict(x_test) # -> test셋에 대한 예측값(y 예측값)

# 실제 test셋의 목표변수(아웃풋) -> y_test 와 y예측값 비교! 평균제곱근 오차로!
# 평균제곱오차를 구해주는 함수
from sklearn.metrics import mean_squared_error

# mean_squared_error(실제 y test목표변수, y test 예측값) -> 평균제곱오차
mean_squared_error(y_test, y_test_prediction) 

# 평균제곱근 오차! 루트는 0.5제곱과 같다 -> ** 0.5
mean_squared_error(y_test, y_test_prediction) ** 0.5
# 8000달러정도의 오차가 있을수 있다.



#################################################################################################################
## 다중 선형 회귀 scikit-learn 데이터 준비

from sklearn.datasets import load_boston
boston_dataset = load_boston()

# 데이터셋에 대한 설명서
print(boston_dataset.DESCR)

# 데이터셋의 속성 이름
boston_dataset.feature_names

# 데이터셋의 데이터 받아오기
boston_dataset.data # numpy 행렬

# numpy 행렬을 깔끔하게 데이터프레임으로 만들기
import pandas as pd

X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names) # 열이름을 속성이름으로

# 목표변수 받아오기 
boston_dataset.target

y = pd.DataFrame(boston_dataset.target, columns=['MEDV']) # 목표변수의 열이름은 MEDV



## scikit-learn으로 다중 선형 회귀 쉽게 하기 (입력변수 여러개)
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split # train, test데이터셋 나누는 함수
from sklearn.linear_model import LinearRegression # 선형회귀하는것
from sklearn.metrics import mean_squared_error # 모델을 평가할때 쓰는 평균제곱오차

# train, test데이터셋 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 선형회귀 함수 
model = LinearRegression()

# trainning set 학습시키기
model.fit(X_train, y_train)

# 세타1, 세타2 ,세타3 ... 이 벡터로 들어가 있다(손실 최소로하는 세타값들)
model.coef_ # coef : 기울기

# 세타0의 값(손실 최소로하는 세타값 상수)
model.intercept_ # ㅑintercept : 절편


# 최적선 평가하기 trainning 셋으로 구한 최적선을 test셋으로 test
# model.predict(test셋 입력변수)
y_test_prediction = model.predict(X_test) # -> test셋에 대한 예측값(y 예측값)

# 실제 test셋의 목표변수(아웃풋) -> y_test 와 y예측값 비교! 평균제곱근 오차로!
# 평균제곱오차를 구해주는 함수
from sklearn.metrics import mean_squared_error

# mean_squared_error(실제 y test목표변수, y test 예측값) -> 평균제곱오차
mean_squared_error(y_test, y_test_prediction) 

# 평균제곱근 오차! 루트는 0.5제곱과 같다 -> ** 0.5
mean_squared_error(y_test, y_test_prediction) ** 0.5
# 4500달러정도의 오차가 있을수 있다.

# 입력변수 1개 만했을때 오차 8000달려였는데 다중회귀로 하니까 4500으로 오차가 줄었다(더 정확한 결과)


##############################################################################################################
## scikit-learn으로 다항 회귀 문제 만들기

# 다항회귀를 하기위해서는 먼저 우리 데이터셋의 가상의열(가상의 속성)을 만들어야 한다
from sklearn.datasets import load_boston

# 보스턴 데이터셋 불러오기
boston_dataset = load_boston()




'''
다중 선형 회귀를 할 때, 설계행렬에 x0의 값을 따로 설정해줬는데
PolynomialFeatures() 메소드를 사용해서 데이터를 가공하면
저절로 bias가 추가되서 같은 기능을 하는게 아닐까 싶어요.
ex)
X = np.array([
  np.ones(16), # 이 부분과 같은 기능을 bias 가 하는 것 같습니다.
  house_size,
  distance_from_station,
  number_of_rooms
  ]).T '''


## scikit-learn으로 다항 회귀 하기(다항회귀 데이터를 만들었다면 다중회귀 하듯이 하면 된다.)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 목표변수 데이터 프레임으로 저장
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])

# train vs test셋 나누기
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 선형회귀모델 가져와서 학습시키기
model = LinearRegression()
model.fit(x_train, y_train)

# 학습된 결과 보기
model.coef_  # 세타값들
model.intercept_ # 세타0(상수항 들어있다)

## 모델 성능 평가(test셋)
# 에측값 도출
y_test_prediction = model.predict(x_test)

# 실제값과 예측값 얼마나 괴리가 있는지 (평균 제곱근 오차)
mean_squared_error(y_test, y_test_prediction) ** 0.5 
# 이모델로 집가격을 예측하면 오차가 3200달러정도
# 다중선형회귀는 4500달러 
# 이차다항선형회귀는 3200달러


#################################################################################################################
## scikit-learn 로지스틱 회귀 데이터 준비
from sklearn.datasets import load_iris
import pandas as pd

iris_data = load_iris()

# 내용 확인
print(iris_data.DESCR)

x = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
x

y = pd.DataFrame(iris_data.target, columns=['class'])
y

## scikit-learn으로 로지스틱 회귀 쉽게하기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# 로지스틱회귀를 할 때 이코드 추가하면 좋다(오류방지)
y_train = y_train.values.ravel()

# 로지스틱회귀 모델 생성
# 선형회귀 할 때는 괄호안을 비웠다 -> LogisticRegression()
# 로지스틱회귀는 다양한 설정 가능
# solver파라미터 -> 모델을 최적화할 때 어떤 알고리즘을 선택할지 결정할 때 사용하는 알고리즘
# max_iter 파라미터 -> 최적화할 때 그과정을 몇 번 반복할지(충분히 최적화가 됐다고판단하면 알아서 멈춘다)
# 학습률알파는 자동으로 최적화돼있기 때문에 따로 설정할 필요가 없다
model = LogisticRegression(solver='saga', max_iter=2000)

# 모델 학습시키기
model.fit(x_train, y_train)

# 모델을 이용해 예측
model.predict(x_test)

# 모델의 예측력 평가
# 예측한거 중 몇 퍼센트가 올바르게 분류되었는지 확인하면 된다
model.score(x_test, y_test) # 정답률 96퍼센트 제대로 분류한다 




##################################################################################################################
#################################################################################################################
## 머신러닝을 위한 전처리
# scikit-learnd으로 Normalization해보기
import pandas as pd
import os
import numpy as np

from sklearn import preprocessing

os.getcwd()
os.chdir("C:\\Users\\wel27")

nba_player_of_the_week_df = pd.read_csv("NBA_player_of_the_week.csv")

# 데이터 요약
nba_player_of_the_week_df.head()

# 각 열에 대한 통계
nba_player_of_the_week_df.describe()

# Feature Scailing
# 키를 cm로 나타낸 (Height CM), 몸무게를 kg으로 나타낸 (Weight KG), 나이를 나타낸 (Age)
# 데이터프레임으로 따로 저장
height_weight_age_df = nba_player_of_the_week_df[['Height CM', 'Weight KG', 'Age']]
height_weight_age_df.head()

# sklearn에서 feature scaling 도구 가져오기
# min max normaliazation
# MinMaxScaler() -> min max normaliazation을 사용해서 데이터를 0과1사이로 만들어준다
scaler = preprocessing.MinMaxScaler()
normalization_data = scaler.fit_transform(height_weight_age_df)

normalization_data # height_weight_age -> 속성모두 0과1사이로 확인

# 이데이터를 다시 데이터프레임으로 만들어주기
normalized_df = pd.DataFrame(normalization_data, columns=['Height', 'Weight', 'Age'])

# 데이터 요약 통계량 확인
normalized_df.describe()



################################################################################################################
##Feature Scaling: 표준화(Standardization) 
from sklearn import preprocessing
import pandas as pd
import numpy as np
    
NBA_FILE_PATH = '../datasets/NBA_player_of_the_week.csv'
# 소수점 5번째 자리까지만 출력되도록 설정
pd.set_option('display.float_format', lambda x: '%.5f' % x)
    
nba_player_of_the_week_df = pd.read_csv(NBA_FILE_PATH)
    
# 데이터를 standardize 함
scaler = preprocessing.StandardScaler()
standardized_data = scaler.fit_transform(height_weight_age_df)
    
standardized_df = pd.DataFrame(standardized_data, columns=['Height', 'Weight', 'Age'])

'''사실 데이터를 Normalize할 때와 비교해서 바뀌는 부분은 한 줄밖에 없는데요. 
Min-Max Normalization을 할 때는 preprocessing 모듈에서 MinMaxScaler을 갖고 왔던 거 기억나시나요?
그 부분을 그냥 MinMaxScaler 대신 StandardScaler로 바꿔주기만 하면 됩니다.
StandardScaler는 이미 데이터를 Standardize 하는 방법을 알고 있기 때문에 
똑같이 fit_transform 메소드를 사용해 주면 파라미터로 넘기는 데이터가 자동으로 변환 되죠.'''



################################################################################################################
## pandas로 one-hat incoding 하기

import pandas as pd

TITANIC_FILE_PATH = "C:/Users/wel27/Downloads/titanic (1).csv"
titanic_df = pd.read_csv(TITANIC_FILE_PATH)
titanic_df.head()

# 두열데이터 뽑기
titanic_sex_embarked = titanic_df[['Sex', 'Embarked']]
titanic_sex_embarked.head()

# one-hat incoding -> pd.get_dummies(변환하고 싶은 데이터프레임)
one_hot_encoded_df = pd.get_dummies(titanic_sex_embarked)
one_hot_encoded_df.head()


# 전체 데이터프레임에서 원하는 열들만 one-hot incoding
one_hot_encoded_df = pd.get_dummies(data=titanic_df, columns=['Sex', 'Embarked'])
one_hot_encoded_df.head()


#########################################################################################################################
### 정규화(Regularization)
# scikit-learn으로 과적합 문제 직접 보기
# 과소적합 방지(이차 이상 회귀모델)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from math import sqrt

import numpy as np
import pandas as pd

ADMISSION_FILE_PATH = "C:/Users/wel27/Downloads/admission_data.csv"
admission_df = pd.read_csv(ADMISSION_FILE_PATH)

# 안쓰는 데이터프레임 삭제
admission_df = pd.read_csv(ADMISSION_FILE_PATH).drop('Serial No.', axis=1) # axis=1 의미는 해당 columns를 제거
admission_df.head()

# 학생이 지원한 학교에 합격할 확률

# 입력변수 따로 저장
X = admission_df.drop(['Chance of Admit '], axis=1)

# 6차항 변형기 정의
polynomial_transformer = PolynomialFeatures(6)

# 변형기에 input변수를 데이터를 넣어서 변수를 좀더 높은 차항으로 바꿔준다
# 직선대신 6차항의 다항회귀 모델 데이터를 준비
polynomial_features = polynomial_transformer.fit_transform(X.values)

# 변수 이름 생성
features = polynomial_transformer.get_feature_names(X.columns)

# 변환한 내용들을 입력변수 데이터프레임에 다시 저장(입력변수를 6차항으로 잘 바꿈)
X = pd.DataFrame(polynomial_features, columns=features)
X.head()

# 목표변수 준비
y = admission_df[['Chance of Admit ']]
y.head()

# 만든 데이터로 다항회귀
# train-test set 나누기
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# 선형회귀 모델을 만들어 training set 학습
model = LinearRegression()
model.fit(X_train, y_train)

## train, test set 성능 비교
# training set으로 예측값 계산
y_train_predict = model.predict(X_train)

# test set으로 예측값 계산
y_test_predict = model.predict(X_test)

# traning set과 test set에서의 평균 제곱근 오차로 평가
mse = mean_squared_error(y_train, y_train_predict)

print("training set에서의 성능")
print("----------------------")
print(sqrt(mse)) # 0에 가까운 좋은결과, 모델이 training set데이터의 관계를 아주 잘 학습(복잡한 모델로 과소적합 방지)

mse = mean_squared_error(y_test, y_test_predict)

print("test set에서의 성능")
print("----------------------")
print(sqrt(mse)) # 5가 넘는 높은 오차, 우리모델이 트레이닝 데이터에 과적합

# 보통 복잡한 모델을 그대로 학습시키면 과적합이된다.



##############################################################################
## scikit-learn으로 과적합 문제 해결해 보기
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from math import sqrt

import numpy as np
import pandas as pd

ADMISSION_FILE_PATH = "C:/Users/wel27/Downloads/admission_data.csv"
admission_df = pd.read_csv(ADMISSION_FILE_PATH)

# 안쓰는 데이터프레임 삭제
admission_df = pd.read_csv(ADMISSION_FILE_PATH).drop('Serial No.', axis=1) # axis=1 의미는 해당 columns를 제거
admission_df.head()

# 학생이 지원한 학교에 합격할 확률

# 입력변수 따로 저장
X = admission_df.drop(['Chance of Admit '], axis=1)

# 6차항 변형기 정의
polynomial_transformer = PolynomialFeatures(6)

# 변형기에 input변수를 데이터를 넣어서 변수를 좀더 높은 차항으로 바꿔준다
# 직선대신 6차항의 다항회귀 모델 데이터를 준비
polynomial_features = polynomial_transformer.fit_transform(X.values)

# 변수 이름 생성
features = polynomial_transformer.get_feature_names(X.columns)

# 변환한 내용들을 입력변수 데이터프레임에 다시 저장(입력변수를 6차항으로 잘 바꿈)
X = pd.DataFrame(polynomial_features, columns=features)
X.head()

# 목표변수 준비
y = admission_df[['Chance of Admit ']]
y.head()

# 만든 데이터로 다항회귀
# train-test set 나누기
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Lasso 모델을 만들어 training set 학습
# 람다 -> 세타값이 커지는 것에 대해 얼만큼의 패널티를 줄것인가에 대한 변수 : 여기서는 alpha 파라미터
# 손실함수를 최소화 하기위해 경사하강법을 사용하는데 경사하강법을 최대한 몇 번 할지 정해줄 수 있다 : max_iter 파라미터
# feature scaling을 하면 경사하강법으로 극소점을 좀더 빨리 찾을수 있는데 
# lasso,ridge 모델은 자체적으로 가능 : normalize=True -> 모델을 학습시키기전에 데이터를 자동으로 0과1사이로 만들어준다
model = Lasso(alpha=0.001, max_iter=1000, normalize=True) # L1 정규화
# model = Ridge(alpha=0.001, max_iter=1000, normalize=True) # L2 정규화

model.fit(X_train, y_train)

## train, test set 성능 비교
# training set으로 예측값 계산
y_train_predict = model.predict(X_train)

# test set으로 예측값 계산
y_test_predict = model.predict(X_test)

# traning set과 test set에서의 평균 제곱근 오차로 평가
mse = mean_squared_error(y_train, y_train_predict)

print("training set에서의 성능")
print("----------------------")
print(sqrt(mse))

mse = mean_squared_error(y_test, y_test_predict)

print("test set에서의 성능")
print("----------------------")
print(sqrt(mse)) # lasso모델을 사용하니 test모델에서도 성능 굿

############################################################################################################3
############################################################################################################
##scikit-learn으로 k겹 교차 검증 해보기
# train test 나누지 않고 바로 k겹 교차검증
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # 경고 메세지 막는 출력

iris_data = datasets.load_iris()

X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['Class'])

# 로지스틱 회귀모델 
logistic_model = LogisticRegression(max_iter=2000)

# 보통의경우 train-test-split 모듈을 사용해 데이터를 나눠줘야하지만 k겹 교차검증은 그럴필요 x
# k겹 교차검증
# cross_val_score(교차검증할 모델, 입력변수, 목표변수.values.ravel(), cv=5)
# .values.ravel() -> 경고 메세지 막아줌, cv = -> k를 정해주는 파라미터
cross_val_score(logistic_model, X, y.values.ravel(), cv=5)
# cross_val_score는 k겹 교차검증을 한번에 해주는 함수, 주어진 데이터셋을 k개로 나눈후
# 각 데이셋을 사용해서 모델을 학습하고 평가해준다
#5개의 데이터셋에 대한 성능이 리턴

# 성능의 평균
np.average(cross_val_score(logistic_model, X, y.values.ravel(), cv=5))


###################################################################################################################
#################################################################################################################
## scikit-learn으로 그리드 서치 해보기
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from math import sqrt

import numpy as np
import pandas as pd

# 데이터 준비
ADMISSION_FILE_PATH = "C:/Users/wel27/Downloads/admission_data.csv"
admission_df = pd.read_csv(ADMISSION_FILE_PATH)

# 입력변수 따로 저장
X = admission_df.drop(['Chance of Admit '], axis=1)

# 2차항 변형기 정의
polynomial_transformer = PolynomialFeatures(2) # 2 차식 변형기를 정의한다

# 변형기에 input변수를 데이터를 넣어서 변수를 좀더 높은 차항으로 바꿔준다
# 직선대신 2차항의 다항회귀 모델 데이터를 준비
polynomial_features = polynomial_transformer.fit_transform(X.values)

# 변수 이름 생성
features = polynomial_transformer.get_feature_names(X.columns)

# 변환한 내용들을 입력변수 데이터프레임에 다시 저장(입력변수를 6차항으로 잘 바꿈)
X = pd.DataFrame(polynomial_features, columns=features)
y = admission_df[['Chance of Admit ']]

# 최적화할 하이퍼파라미터들과 하이퍼파라미터에 넣어볼 후보값들이 들어가 있는 파이썬 딕셔너리를 만든다
hyper_parameter = {
    'alpha' : [0.01, 0.1, 1, 10],
    'max_iter' : [100, 500, 1000, 1500, 2000]
} 

# 사용할 모델
lasso_model = Lasso()

# 그리드서치 도구 만들기
hyper_parameter_tunner = GridSearchCV(lasso_model, hyper_parameter, cv=5)
hyper_parameter_tunner.fit(X, y)
# 주어진 데이터로 그리드 서치를 해라 경고문 -> max_iter값이 충분하지 않은 모델들이 있었다
# 작은값들도 확인해보고 싶었던 거니까 경고 메세지는 무시

# 최적의 파라미터 확인
hyper_parameter_tunner.best_params_



###############################################################################################################
#############################################################################################################
### 결정트리
## if-else 문으로 간단한 결정트리 만들기
def survival_classifier(seat_belt, highway, speed, age):
    # 코드를 
    if seat_belt == True:
        return 0
    else:
        if highway == False:
            return 0
        else:
            if speed < 100:
                return 0
            else:
                if age < 50:
                    return 0
                else:
                    return 1
        
print(survival_classifier(False, True, 110, 55))
print(survival_classifier(True, False, 40, 70))
print(survival_classifier(False, True, 80, 25))
print(survival_classifier(False, True, 120, 60))
print(survival_classifier(True, False, 30, 20))


## 정답
def survival_classifier(seat_belt, highway, speed, age):
    # 질문 노드: 안전 벨트를 했나요?
    if seat_belt:
        return 0  # 했으면 생존 리턴
    else:
        # 질문 노드: 사고가 고속도로였나요?
        if highway:
            # 질문 노드: 시속 100km를 넘었나요?
            if speed > 100:
                # 질문 노드: 사고자 나이가 50을 넘었나요?
                if age > 50:
                    return 1  # 사고자 나이가 50을 넘었으면 사망 리턴
                else:
                    return 0  # 사고자 나이가 50을 넘지 않았다면 생존 리턴
            else:
                return 0  # 시속 100km를 넘지 않았다면 생존 리턴
        else:  
            return 0  # 고속도로가 아니였다면 생존 리턴

## 정답
def survival_classifier(seat_belt, highway, speed, age):
  if not seat_belt:
    if highway:
      if speed > 100:
        if age > 50:
          return 1
  return 0


################################################################################################################################
###################################################################################################################
### 결정트리(Decision Tree)
## scikit-learn 데이터 준비하기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


iris_data = load_iris()
print(iris_data.DESCR)

# 데이터 셋 dataframe에 저장
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['class'])



## scikit-learn으로 결정 트리 쉽게 사용하기
# train-test 데이터 분리하기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 결정트리 모델 생성
# 괄호안 옵셔널 파라미터 넣으면 다양하게 사용가능
model = DecisionTreeClassifier(max_depth=4) # max_depth -> 만들려는 노드의 최대깊이 얼마나 깊은 결정트리를 만들건가
model.fit(X_train, y_train)

model.predict(X_test) # 분류문제여서 예측값이 0, 1, 2 뿐이다
model.score(X_test, y_test) # 90퍼센트의 확률로 제대로 분류한다


## scikit-learn으로 속성 중요도 확인하기
# 모델을 학습시키면 변수가 자동으로 저장된다
model.feature_importances_ # 속성드리 순서대로 얼마나 중요한지 넘파이 배열안에 들어있다

# 속성 중요도 정렬 시각화
importances = model.feature_importances_

indices_sorted = np.argsort(importances)

plt.figure()
plt.title('Feature importances')
plt.bar(range(len(importances)), importances[indices_sorted])
plt.xticks(range(len(importances)), X.columns[indices_sorted], rotation=90)
plt.show()


########################################################################################################
######################################################################################################
## scikit-learn으로 랜덤 포레스트 쉽게 사용하기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

iris_data = load_iris()

# 데이터 셋 dataframe에 저장
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel() # 랜덤포레스트 모델 학습시킬때 경고메세지가 안나오게 하는 코드

# 랜덤포레스트 모델 정의
RandomForestClassifier() # 비워두면 기본옵션들을 사용

# n_estimators -> 랜덤포레스트모델이 결정트리 몇 개를 만들어서 예측을 할지 정해주는 파라미터
# 안쓰면 기본값이 10이다
RandomForestClassifier(n_estimators=100)  # 100개의 결정트리를 사용하라

# max_depth -> 만들 트리들의 최대 깊이 정해주는 파라미터
RandomForestClassifier(max_depth=4)  # 최대 깊이를 4로 설정

model = RandomForestClassifier(n_estimators=100, max_depth=4)

# 모델 학습시키기
model.fit(X_train, y_train)

# 모델을 통해 예측하기
model.predict(X_test) # 분류문제이기 때문에 0,1,2의 예측값이 나온다

# 분류모델 평가 -> 예측한 값들중에 몇 퍼센트가 맞게 분류됐는지 확인
model.score(X_test, y_test)

# 랜덤포레스트도 결정트리를 이용하기 때문에 평균 지니감소를 이용해서 속성중요도를 계산할 수 있다
# 모델을 학습시키면 feature_infortance_라는 변수에 자동으로 저장된다
importances = model.feature_importances_

# 속성중요도 시각화
indices_sorted = np.argsort(importances)

plt.figure()
plt.title('Feature importances')
plt.bar(range(len(importances)), importances[indices_sorted])
plt.xticks(range(len(importances)), X.columns[indices_sorted], rotation=90)
plt.show()


################################################################################################################
##############################################################################################################
## scikit-learn으로 에다 부스트 쉽게 사용하기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd

iris_data = load_iris()

# 데이터 셋 dataframe에 저장
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()

# 에다 부스트 모델 정의
# 괄호 비워두면 기본 파라미터 사용
# n_estimators -> 결정스텀프를 몇 개를 만들어서 사용할 지 결정 (기본값은 10)
model = AdaBoostClassifier(n_estimators=100) 

# 학습 
model.fit(X_train, y_train)

# 예측
model.predict(X_test)

# 분류 모델 평가 -> 예측값들 중에 몇 퍼센트를 맞게 예측했냐
model.score(X_test, y_test) # 데스트셋의 인풋, 테스트의 실제값

# 속성중요도 
# 에다부스트도 결정트리를 이용하기 때문에 평균지니감소를 이용해서 속성중요도 계산 가능
# 모델 학습할 때 feature_infortances_라는 변수에 자동저장 
model.feature_importances_ 

# 속성 중요도 시각화
import matplotlib.pyplot as plt
import numpy as np

importances = model.feature_importances_ 
 
indices_sorted = np.argsort(importances)

plt.figure()
plt.title('Feature importances')
plt.bar(range(len(importances)), importances[indices_sorted])
plt.xticks(range(len(importances)), X.columns[indices_sorted], rotation=90)
plt.show()


#####################################################################################################
#####################################################################################################
## 협업 클러스터링 numpy 기본 함수
import numpy as np

# np.sum 함수
# np.sum함수는 파라미터로 넘겨주는 행렬 안에 있는 모든 원소들의 합을 구해주는 함수입니다.
A = np.array([
    [3, 3, 2, 3, 1],
    [5, 2, 2, 3, 1],
    [3, 3, 2, 3, 1],
    [3, 1, 4, 3, 1],
])

np.sum(A)

# 그렇지만 행렬 안에 nan 값이 있으면 항상 np.sum 함수도 nan을 리턴하는데요. 
# nan 값들만 제외하고 계산을 하고 싶을 때는 np.nansum이라는 함수를 사용하면 됩니다.
A = np.array([
    [3, 3, 2, 3, 1],
    [5, 2, 2, 3, 1],
    [3, 3, np.nan, 3, 1],
    [3, 1, 4, 3, 1],
])

np.nansum(A)

# np.mean 함수
# np.mean 함수는 행렬의 모든 원소들의 평균 값을 계산해 주는 함수입니다.
A = np.array([
    [3, 3, 2, 3, 1],
    [5, 2, 2, 3, 1],
    [3, 3, 2, 3, 1],
    [3, 1, 4, 3, 1],
])

np.mean(A)

# 이렇게 말이죠. np.sum과 마찬가지로 원소 중 단 한 개라도 nan 값이면 결과도 nan이 되는데요. 
# 그럴 때는 똑같이, np.nanmean 함수를 사용할 수 있습니다. 
# nan인 원소를 제외한 다른 모든 원소들의 평균값을 계산할 수 있죠. 이렇게요:
A = np.array([
    [3, 3, 2, 3, 1],
    [5, 2, 2, 3, 1],
    [3, 3, np.nan, 3, 1],
    [3, 1, 4, 3, 1],
])

np.nanmean(A)


####
# numpy 행렬 접근법
# 기본 인덱스 접근법
# numpy 행렬 안에 있는 세부 데이터는 인덱스를 사용해서 접근합니다.
# 예를 들어 A의 0 번째 행에 접근하고 싶으면, A[0], 3 번째 행에 접근하고 싶으면 A[3] 이렇게요.

A = np.array([
    [3, 3, 2, 3, 1],
    [5, 2, 2, 3, 1],
    [3, 3, 2, 3, 1],
    [3, 1, 4, 3, 1],
])

A[0]  # array([3, 3, 2, 3, 1]) 
A[3]  # array([3, 1, 4, 3, 1]) 

# 열 접근법
# 그럼 열에 접근하고 싶으면 어떻게 할까요? 3열에 접근하고 싶을 때는 이렇게 하면 되죠. A[:, 3]
A[:, 3]  # array([3, 3, 3, 3])

# A의 3 행이 잘 리턴이 됩니다. 그러니까 : 뒤에 , 그리고 원하는 열을 선택하면 되는 거죠. 
# 사실 행도 이렇게 똑같이 접근할 수 있는데요. 행은 반대로 이렇게 하면 됩니다. A[3, :]
A[3, :]  # array([3, 1, 4, 3, 1]) 리턴







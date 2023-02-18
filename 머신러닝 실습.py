## numpy 행렬 연습

import numpy as np

A = np.array([
    [0, 1, -1],
    [1, 2, 3],
    [2, 1, 0],
    [-1, 2, -4]
])

B = np.array([
    [0, 2],
    [1, 1],
    [-1, -2]
])

# A행렬의 2행 2열 원소
A[1][1]

# B행렬의 4행 1열 원소
A[3][0]


## numpy로 행렬 연산 연습하기
import numpy as np

A = np.array([
    [1, -1, 2],
    [3, 2, 2]
])

B = np.array([
    [0, 1],
    [-1, 1],
    [5, 2]
])

C = np.array([
    [2, -1],
    [-3, 3]
])

D = np.array([
    [-5, 1],
    [2, 0]
])

# 행렬 연산 결과를 result에 저장하세요
result = 2*A @ -B @ (3*C + D)

result 

##
# 숫자/행렬 스칼라 곱은 
# 행렬 덧셈은 +
# 행렬 곱셈은 @


## numpy로 전치, 단위, 역행렬연습
# B(T) * (2*A(T)) * (3*C(-)+D(T))
import numpy as np

A = np.array([
    [1, -1, 2],
    [3, 2, 2]
])

B = np.array([
    [0, 1],
    [-1, 1],
    [5, 2]
])

C = np.array([
    [2, -1],
    [-3, 3]
])

D = np.array([
    [-5, 1],
    [2, 0]
])

# 정답
B.T @ (2*A.T) @ (3*np.linalg.pinv(C) + D.T)

## 오류
# B.T라는 행렬과 2라는 스칼라 값은 @으로 계산할 수 없어요!!
B.T @ 2*A.T @ (3*np.linalg.pinv(C) + D.T)
B.T @ 2*(A.T) @ (3*np.linalg.pinv(C) + D.T)



########################################################################################################
### 선형회귀 가설함수 구현하기
'''실습과제
이번 과제에서는 가설 함수를 사용해서 주어진 데이터를 예측하는 코드를 구현해보겠습니다.
prediction이라는 함수로 구현할 건데요. 이 함수에 대해서 설명드릴게요.

prediction 함수
prediction 함수는 주어진 가설 함수로 얻은 결과를 리턴하는 함수입니다. 
파라미터로는 세타제로를 나타내는 숫자형 변수 theta_0,
세타원을 나타내는 숫자형 변수 theta_1, 그리고 모든 입력 변수 벡터 x들을 나타내는 numpy 배열 x를 받죠.
 

가설 함수를 h세타 = 세타제로 + 세타원 * x 이렇게 정의했는데요.

prediction 함수는 x의 각 요소의 예측값에 대한 numpy 배열을 리턴합니다.
numpy 배열과 연산들을 이용해서 prediction 함수를 작성해보세요.

numpy 배열과 숫자형 덧셈
numpy 배열과 일반 숫자형을 더하면 numpy 배열의 모든 요소에 해당 숫자형이 더해집니다. 
이걸 사용해서 과제를 풀어보세요!'''

np_array = np.array([1, 2, 3, 4, 5])
    
5 + np_array  # [6, 7, 8, 9, 10]

''' 출력예시 
array([ -1.2,  -0.2,   1. ,   1.2,   2.2,   3.6,   3.7,   4.8,   5.8,
         6.4,   7.4,   8.5,  10.4,  10.8])'''


import numpy as np

def prediction(theta_0, theta_1, x):
    """주어진 학습 데이터 벡터 x에 대해서 예측 값을 리턴하는 함수"""
    # 코드를 쓰세요
    return theta_0 + theta_1 * x


# 테스트 코드

# 입력 변수(집 크기) 초기화 (모든 집 평수 데이터를 1/10 크기로 줄임)
house_size = np.array([0.9, 1.4, 2, 2.1, 2.6, 3.3, 3.35, 3.9, 4.4, 4.7, 5.2, 5.75, 6.7, 6.9])
theta_0 = -3
theta_1 = 2

prediction(theta_0, theta_1, house_size)



### 선형회귀 예측 오차 구현하기

import numpy as np

def prediction(theta_0, theta_1, x):
    """주어진 학습 데이터 벡터 x에 대해서 모든 예측 값을 벡터로 리턴하는 함수"""
    # 저번 과제에서 쓴 코드를 갖고 오세요
    return theta_0 + theta_1 * x
    

def prediction_difference(theta_0, theta_1, x, y):
    """모든 예측 값들과 목표 변수들의 오차를 벡터로 리턴해주는 함수"""
    # 코드를 쓰세요
    return prediction(theta_0, theta_1, x) - y
#   return(theta_0 + theta_1 * x) - y 같은 답이다
    
# 입력 변수(집 크기) 초기화 (모든 집 평수 데이터를 1/10 크기로 줄임)
house_size = np.array([0.9, 1.4, 2, 2.1, 2.6, 3.3, 3.35, 3.9, 4.4, 4.7, 5.2, 5.75, 6.7, 6.9])

# 목표 변수(집 가격) 초기화 (모든 집 값 데이터를 1/10 크기로 줄임)
house_price = np.array([0.3, 0.75, 0.45, 1.1, 1.45, 0.9, 1.8, 0.9, 1.5, 2.2, 1.75, 2.3, 2.49, 2.6])

theta_0 = -3
theta_1 = 2

prediction_difference(-3, 2, house_size, house_price)




### 선형 회귀 경사 하강법 구현하기
'''numpy 배열 원소들 평균
numpy 배열에 mean() 메소드를 사용하면 안에 들어 있는 원소들의 평균을 쉽게 구할 수 있습니다.'''
np_array = np.array([1, 2, 3, 4, 5])
    
np_array.mean()  # 3

###########################################
import numpy as np

def prediction(theta_0, theta_1, x):
    """주어진 학습 데이터 벡터 x에 대해서 모든 예측 값을 벡터로 리턴하는 함수"""
    # 지난 과제 코드를 갖고 오세요
    return theta_0 + theta_1 * x
    
def prediction_difference(theta_0, theta_1, x, y):
    """모든 예측 값들과 목표 변수들의 오차를 벡터로 리턴해주는 함수"""
    # 지난 과제 코드를 갖고 오세요
    return prediction(theta_0, theta_1, x) - y
    
def gradient_descent(theta_0, theta_1, x, y, iterations, alpha):
    """주어진 theta_0, theta_1 변수들을 경사 하강를 하면서 업데이트 해주는 함수"""
    for _ in range(iterations):  # 정해진 번만큼 경사 하강을 한다
        error = prediction_difference(theta_0, theta_1, x, y)  # 예측값들과 입력 변수들의 오차를 계산
        # 코드를 쓰세요
        theta_0 = theta_0 - alpha * error.mean()
        theta_1 = theta_1 - alpha * (error * x).mean()
        
    return theta_0, theta_1
    
    
# 입력 변수(집 크기) 초기화 (모든 집 평수 데이터를 1/10 크기로 줄임)
house_size = np.array([0.9, 1.4, 2, 2.1, 2.6, 3.3, 3.35, 3.9, 4.4, 4.7, 5.2, 5.75, 6.7, 6.9])

# 목표 변수(집 가격) 초기화 (모든 집 값 데이터를 1/10 크기로 줄임)
house_price = np.array([0.3, 0.75, 0.45, 1.1, 1.45, 0.9, 1.8, 0.9, 1.5, 2.2, 1.75, 2.3, 2.49, 2.6])

# theta 값들 초기화 (아무 값이나 시작함)
theta_0 = 2.5
theta_1 = 0

# 학습률 0.1로 200번 경사 하강
theta_0, theta_1 = gradient_descent(theta_0, theta_1, house_size, house_price, 200, 0.1)

theta_0, theta_1



#############################################################################################################
## 선형회귀 sklearn으로 구현하기
# 범죄율로 집 값 예측하기

# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

# 보스턴 집 데이터 갖고 오기
boston_house_dataset = datasets.load_boston()

# 입력 변수를 사용하기 편하게 pandas dataframe으로 변환
X = pd.DataFrame(boston_house_dataset.data, columns=boston_house_dataset.feature_names)

# 목표 변수를 사용하기 편하게 pandas dataframe으로 변환
y = pd.DataFrame(boston_house_dataset.target, columns=['MEDV'])

## 코드를 쓰세요

# 범죄율 열 선택하기
x = X[['CRIM']]

# traing-test 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# 모델 학습시키기
# 선형회귀 함수 
linear_regression_model = LinearRegression()

# trainning set 학습시키기
linear_regression_model.fit(x_train, y_train)

# test 데이터로 예측
y_test_predict = linear_regression_model.predict(x_test) # -> test셋에 대한 예측값(y 예측값)


# 테스트 코드 (평균 제곱근 오차로 모델 성능 평가)
mse = mean_squared_error(y_test, y_test_predict)

mse ** 0.5




##########################################################################################################
## 다중 선형 회귀 가설 함수 구현하기
# 다중 선형 회귀의 가정 함수를 prediction 함수로 구현

import numpy as np

def prediction(X, theta):
    """다중 선형 회귀 가정 함수. 모든 데이터에 대한 예측 값을 numpy 배열로 리턴한다"""
    # 코드를 쓰세요
    return X @ theta
    
    
# 입력 변수
house_size = np.array([1.0, 1.5, 1.8, 5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0])  # 집 크기
distance_from_station = np.array([5, 4.6, 4.2, 3.9, 3.9, 3.6, 3.5, 3.4, 2.9, 2.8, 2.7, 2.3, 2.0, 1.8, 1.5, 1.0])  # 지하철역으로부터의 거리 (km)
number_of_rooms = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])  # 방 수

# 설계 행렬 X 정의
X = np.array([
    np.ones(16),
    house_size,
    distance_from_station,
    number_of_rooms
]).T

# 파라미터 theta 값 정의
theta = np.array([1, 2, 3, 4])

prediction(X, theta)



##  다중 선형 회귀 경사 하강법 구현하기
import numpy as np

def prediction(X, theta):
    """다중 선형 회귀 가정 함수. 모든 데이터에 대한 예측 값을 numpy 배열로 리턴한다"""
    # 전 과제 코드를 갖고 오세요
    return X @ theta

def gradient_descent(X, theta, y, iterations, alpha):
    """다중 선형 회귀 경사 하강법을 구현한 함수"""
    m = len(X)  # 입력 변수 개수 저장 -> 행렬 x의 행개수(데이터 개수)
    
    for _ in range(iterations):
        # 코드를 쓰세요
        #theta = theta - alpha/m*X.T@(prediction(X, theta)-y)
        error = prediction(X, theta) - y
        theta = theta - alpha / m * (X.T @ error)
        
    return theta
    

# 입력 변수
house_size = np.array([1.0, 1.5, 1.8, 5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0])  # 집 크기
distance_from_station = np.array([5, 4.6, 4.2, 3.9, 3.9, 3.6, 3.5, 3.4, 2.9, 2.8, 2.7, 2.3, 2.0, 1.8, 1.5, 1.0])  # 지하철역으로부터의 거리 (km)
number_of_rooms = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])  # 방 수

# 목표 변수
house_price = np.array([3, 3.2, 3.6 , 8, 3.4, 4.5, 5, 5.8, 6, 6.5, 9, 9, 10, 12, 13, 15])  # 집 가격

# 설계 행렬 X 정의
X = np.array([
    np.ones(16),
    house_size,
    distance_from_station,
    number_of_rooms
]).T

# 입력 변수 y 정의
y = house_price

# 파라미터 theta 초기화
theta = np.array([0, 0, 0, 0])

# 학습률 0.01로 100번 경사 하강
theta = gradient_descent(X, theta, y, 100, 0.01)

theta



###########################################################################################################
## 다중 선형 회귀 정규 방정식 구현하기
# normal_equation 함수는 파라미터로 설계 행렬 X, 모든 목표 변수 벡터 
# y를 받아서 정규 방정식을 계산해 최적의 theta 값들을 numpy 배열로 리턴합니다.
import numpy as np

def normal_equation(X, y):
    """설계 행렬 X와 목표 변수 벡터 y를 받아 정규 방정식으로 최적의 theta를 구하는 함수"""
    # 코드를 쓰세요
    return np.linalg.pinv(X.T @ X) @ X.T @ y
    
# 입력 변수
house_size = np.array([1.0, 1.5, 1.8, 5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0])  # 집 크기
distance_from_station = np.array([5, 4.6, 4.2, 3.9, 3.9, 3.6, 3.5, 3.4, 2.9, 2.8, 2.7, 2.3, 2.0, 1.8, 1.5, 1.0])  # 지하철역으로부터의 거리 (km)
number_of_rooms = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])  # 방 수

# 목표 변수
house_price = np.array([3, 3.2, 3.6 , 8, 3.4, 4.5, 5, 5.8, 6, 6.5, 9, 9, 10, 12, 13, 15])  # 집 가격

# 입력 변수 파라미터 X 정의
X = np.array([
    np.ones(16),
    house_size,
    distance_from_station,
    number_of_rooms
]).T

# 입력 변수 y 정의
y = house_price

# 정규 방적식으로 theta 계산
theta = normal_equation(X, y)
theta


##########################################################################################################
## 다중 선형회귀 scikit-learn으로 당뇨 수치 예측하기
'''실습과제
이번 과제에서는 scikit-learn 라이브러리에 있는 또 다른 데이터 셋인 당뇨병 수치 데이터를 이용해서 선형 모델을 학습시켜보겠습니다.

당뇨병 데이터 셋을 가지고 와서 각각 입력 변수와 목표 변수를 dataframe으로 바꾸는 코드까지는 이미 작성돼 있는데요. 배웠던 것처럼 여기서부터 한 번

1. 데이터를 training/test 셋으로 나누고
2. 선형 회귀 모델을 학습시키고
3. 학습시킨 데이터를 이용해서 예측을 해보세요!
(과제를 하기 전 꼭 print(diabetes_dataset.DESCR)을 써서 데이터 셋 내용을 살펴보세요!)

조건
1. train_test_split 함수의 옵셔널 파라미터는 test_size=0.2, random_state=5 이렇게 설정해 주세요.
2. testing set의 예측 값들은 꼭 변수 y_test_predict에 저장해 주세요.
3. 정답 확인은 모델의 성능으로 합니다. (템플렛 가장 아래 줄에 출력 코드 있음)'''

# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

# 당뇨병 데이터 갖고 오기
diabetes_dataset = datasets.load_diabetes()

# 입력 변수를 사용하기 편하게 pandas dataframe으로 변환
X = pd.DataFrame(diabetes_dataset.data, columns=diabetes_dataset.feature_names)

# 목표 변수를 사용하기 편하게 pandas dataframe으로 변환
y = pd.DataFrame(diabetes_dataset.target, columns=['diabetes'])

# train_test_split를 사용해서 주어진 데이터를 학습, 테스트 데이터로 나눈다
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

linear_regression_model = LinearRegression()  # 선형 회귀 모델을 가지고 오고 
linear_regression_model.fit(X_train, y_train)  # 학습 데이터를 이용해서 모델을 학습 시킨다

y_test_predict = linear_regression_model.predict(X_test)  # 학습시킨 모델로 예측

# 평균 제곱 오차의 루트를 통해서 테스트 데이터에서의 모델 성능 판단
mse = mean_squared_error(y_test, y_test_predict)

mse ** 0.5
 



#####################################################################################################
## 다항 회귀로 당뇨병 예측하기 : 문제만들기
'''
이번 과제에서는 다항 회귀를 한 번 직접 만들어 보겠습니다. 
전에 사용했던 당뇨 데이터를 갖고 오는 부분 코드가 작성돼 있습니다.
여기에서부터 이어서 데이터를 바꿔서 당뇨 수치 예측 문제를 2차 다항 문제로 변환해보세요.

조건
1. 2차 다항 회귀 문제로 바꾼 입력 변수 데이터는 변수 X에 pandas dataframe으로 저장합니다.
2. 데이터 열 이름도 변환한 거에 맞게 바꿔줍니다.'''

# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

diabetes_dataset = datasets.load_diabetes()  # 데이터 셋 갖고오기

# 코드를 쓰세요
# 데이터 확인
diabetes_dataset.data

# 행렬의 모양 확인
diabetes_dataset.data.shape # 442행 10열

# 어떤 속성있는지 확인
diabetes_dataset.feature_names

# 가상의 열 추가(다항속성 만들어 주는 툴)
from sklearn.preprocessing import PolynomialFeatures

# PolynomialFeatures(몇차함수인지 숫자)
polynomial_transformer = PolynomialFeatures(2) # 다항변형기 -> 다항 회귀를위해 데이터 가공

# polynomial_transformer.fit_transform(가공할 데이터)
polynomial_data = polynomial_transformer.fit_transform(diabetes_dataset.data) # 넘파이 행렬

polynomial_data.shape # 442행 66열 기존 10열을 조합하여 총 66열이 된것

# 변수 이름들도 다항 회귀에 맞게 다 바꿔줄게요.
# polynomial_transformer.get_feature_names(기존 데이터셋의 속성의 이름)
# 가능한 모든 2차 조합이 다있다.
polynomial_feature_names = polynomial_transformer.get_feature_names(diabetes_dataset.feature_names)


# 다항회귀를 위한 데이터 프레임 만들기
import pandas as pd
X = pd.DataFrame(polynomial_data, columns = polynomial_feature_names)

# 테스트 코드
X.head()

## 꺨끔 정답
# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

diabetes_dataset = datasets.load_diabetes()  # 데이터 셋 갖고오기

# 코드를 쓰세요
polynomial_transformer = PolynomialFeatures(2)  # 2 차식 변형기를 정의한다
polynomial_features = polynomial_transformer.fit_transform(diabetes_dataset.data)  # 당뇨 데이터를 2차항 문제로 변환

features = polynomial_transformer.get_feature_names(diabetes_dataset.feature_names)  # 입력 변수 이름들도 맞게 바꾼다

X = pd.DataFrame(polynomial_features, columns=features)

# 테스트 코드
X.head()


############################################################################################################
## 다항 회귀로 당뇨병 예측하기 II: 모델 학습하기
'''
저번 과제에서는 당뇨병 데이터를 다항 회귀 문제로 변환했는데요.
이번엔 바꾼 데이터를 사용해서 다항 회귀를 직접해보겠습니다.

데이터를 training/test set으로 나누고,
선형 회귀 모델을 갖고와서, training set 데이터를 사용해서 학습시킨 후
test set 데이터를 이용해서 학습시킨 모델로 예측까지 해보세요.

조건
train_test_split 함수의 옵셔널 파라미터는 test_size=0.2, random_state=5 이렇게 설정해주세요.
예측 값 벡터 변수 이름은 꼭 y_test_predict를 쓰세요!
정답 확인은 모델의 성능으로 합니다. (템플렛 가장 아래 줄에 출력 코드 있음)'''

# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

diabetes_dataset = datasets.load_diabetes()

# 지난 과제 코드를 가지고 오세요.
polynomial_transformer = PolynomialFeatures(2)  # 2 차식 변형기를 정의한다
polynomial_features = polynomial_transformer.fit_transform(diabetes_dataset.data)  # 당뇨 데이터를 2차항 문제로 변환

features = polynomial_transformer.get_feature_names(diabetes_dataset.feature_names)  # 입력 변수 이름들도 맞게 바꾼다

X = pd.DataFrame(polynomial_features, columns=features)

# 목표 변수
# diabetes_dataset의 메소드 target으로 데이터가 존재
# 이에 대한 컬럼명을 diabetes로 하는 데이터 프레임을 생성
# 이를 y 변수에 할당
y = pd.DataFrame(diabetes_dataset.target, columns=['diabetes'])

## 코드를 쓰세요

# train vs test셋 나누기
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 선형회귀모델 가져와서 학습시키기
model = LinearRegression()
model.fit(x_train, y_train)

# 학습된 결과 보기
model.coef_  # 세타값들
model.intercept_ # 세타0(상수항 들어있다)

## 모델 성능 평가(test셋)
# 예측값 도출
y_test_predict = model.predict(x_test)

# 실제값과 예측값 얼마나 괴리가 있는지 (평균 제곱근 오차)
mse = mean_squared_error(y_test, y_test_predict)
mse ** 0.5


### 꺨끔정답
# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd  

diabetes_dataset = datasets.load_diabetes()  # 데이터 셋 갖고오기

# 코드를 쓰세요
polynomial_transformer = PolynomialFeatures(2)  # 2 차식 변형기를 정의한다
polynomial_features = polynomial_transformer.fit_transform(diabetes_dataset.data)  # 당뇨 데이터를 2차항 문제로 변환

features = polynomial_transformer.get_feature_names(diabetes_dataset.feature_names)  # 입력 변수 이름들도 맞게 바꾼다

X = pd.DataFrame(polynomial_features, columns=features)

# 목표 변수
y = pd.DataFrame(diabetes_dataset.target, columns=['diabetes'])

# train_test_split를 사용해서 주어진 데이터를 학습, 테스트 데이터로 나눈다
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

# 선형 회귀 모델을 가지고 오고 
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)  # 학습 데이터를 이용해서 모델을 학습 시킨다

# 평균 제곱 오차의 루트를 통해서 테스트 데이터에서의 모델 성능 판단
y_test_predict = linear_regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_test_predict)

mse ** 0.5



##################################################################################################################
## 로지스틱 회귀 가정 함수 구현하기
import numpy as np

def sigmoid(x):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-x))
    

def prediction(X, theta):
    """로지스틱 회귀 가정 함수"""
    # 코드를 쓰세요
    return sigmoid(X @ theta)
    

# 입력 변수
hours_studied = np.array([0.2, 0.3, 0.7, 1, 1.3, 1.8, 2, 2.1, 2.2, 3, 4, 4.2, 4, 4.7, 5.0, 5.9])  # 공부 시간 (단위: 100시간)
gpa_rank = np.array([0.9, 0.95, 0.8, 0.82, 0.7, 0.6, 0.55, 0.67, 0.4, 0.3, 0.2, 0.2, 0.15, 0.18, 0.15, 0.05]) # 학년 내신 (백분률)
number_of_tries = np.array([1, 2, 2, 2, 4, 2, 2, 2, 3, 3, 3, 3, 2, 4, 1, 2])  # 시험 응시 횟수

# 설계 행렬 X 정의
X = np.array([
    np.ones(16),
    hours_studied,
    gpa_rank,
    number_of_tries
]).T

# 파라미터 theta 정의
theta = [0.5, 0.3, -2, 0.2]  

prediction(X, theta)

###################################################################################################################3

## 로지스틱 회귀 경사하강법 구현하기
import numpy as np

def sigmoid(x):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-x))
    
    
def prediction(X, theta):
    """로지스틱 회귀 가정 함수"""
    # 지난 과제에서 작성한 코드를 갖고 오세요
    return sigmoid(X @ theta)
    

def gradient_descent(X, theta, y, iterations, alpha):
    """로지스틱 회귀 경사 하강 알고리즘"""
    m = len(X)  # 입력 변수 개수 저장

    for _ in range(iterations):
        # 코드를 쓰세요
        error = prediction(X, theta) - y
        theta = theta - alpha / m * (X.T @ error)
        
            
    return theta
    
    
# 입력 변수
hours_studied = np.array([0.2, 0.3, 0.7, 1, 1.3, 1.8, 2, 2.1, 2.2, 3, 4, 4.2, 4, 4.7, 5.0, 5.9])  # 공부 시간 (단위: 100시간)
gpa_rank = np.array([0.9, 0.95, 0.8, 0.82, 0.7, 0.6, 0.55, 0.67, 0.4, 0.3, 0.2, 0.2, 0.15, 0.18, 0.15, 0.05]) # 학년 내신 (백분률)
number_of_tries = np.array([1, 2, 2, 2, 4, 2, 2, 2, 3, 3, 3, 3, 2, 4, 1, 2])  # 시험 응시 횟수

# 목표 변수
passed = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # 시험 통과 여부 (0: 탈락, 1:통과)

# 설계 행렬 X 정의
X = np.array([
    np.ones(16),
    hours_studied,
    gpa_rank,
    number_of_tries
]).T

# 입력 변수 y 정의
y = passed

theta = [0, 0, 0, 0]  # 파라미터 초기값 설정
theta = gradient_descent(X, theta, y, 300, 0.1)  # 경사 하강법을 사용해서 최적의 파라미터를 찾는다
theta
    

################################################################################################################
## 로지스틱 회귀로 와인 종류 분류하기
'''
이번 과제에서는 scikit-learn을 이용해서 다양한 와인 데이터를 분류해볼게요. 
데이터는 scikit-learn에서 기본적으로 사용할 수 있는 와인 데이터를 사용할 거고요. 
이걸 가지고 와서 입력과 목표 변수로 나누는 부분까지는 코드가 작성돼있습니다. 여러분이 한 번 직접

1. 데이터를 training/test set으로 나눈다.
2. scikit-learn의 LogisticRegression 모델을 학습한다.
3. 학습시킨 모델을 이용해서 test set 데이터에 대한 예측한다.
이 세 가지를 해보세요.

(본격적으로 시작하기 전에 print(wine_data.DESCR)을 실행해서 데이터 셋을 살펴보는 걸 잊지마세요!)

조건

1. train_test_split 함수의 옵셔널 파라미터는 test_size=0.2, random_state=5 이렇게 설정해주세요.
2. 경고 메시지가 나오지 않게 학습시키기 전에 y_train = y_train.values.ravel() 이 코드를 추가하는 걸 잊지마세요.
3. LogisticRegression 모델의 옵셔널 파라미터는 solver='saga', max_iter=7500 이렇게 설정해주세요.
4. test set 데이터 예측 값들을 저장하는 numpy 배열 변수 이름은 y_test_predict으로 정의하세요.'''

# 필요한 라이브러리 import
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd  

wine_data = datasets.load_wine()
""" 데이터 셋을 살펴보는 코드
print(wine_data.DESCR)
"""

# 입력 변수를 사용하기 편하게 pandas dataframe으로 변환
X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# 목표 변수를 사용하기 편하게 pandas dataframe으로 변환
y = pd.DataFrame(wine_data.target, columns=['Y/N'])

# 코드를 쓰세요
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
y_train = y_train.values.ravel()

logistic_model = LogisticRegression(solver='saga', max_iter=7500)  # sci-kit learn에서 로지스틱 모델을 가지고 온다
logistic_model.fit(X_train, y_train)  # 학습 데이터를 이용해서 모델을 학습 시킨다

# 로지스틱 회귀 모델를 이용해서 각 와인 데이터 분류를 예측함
y_test_predict = logistic_model.predict(X_test)

# 로지스틱 회귀 모델의 성능 확인 (정확성 %를 리턴함)
score = logistic_model.score(X_test, y_test)
y_test_predict, score

#######################################################################################################
#########################################################################################################
## 데이터 전처리
## Normalization 직접 해보기

# 필요한 도구 임포트
from sklearn import preprocessing
import pandas as pd

PATIENT_FILE_PATH = 'D:/2021년 11월/liver_patient_data.csv'
# 데이터가 저장돼있는 파일 경로를 변수에 저장하고, 소숫점 몇 자리까지 출력할 건지 설정해 줍니다. 
# (출력 내용은 소숫점 5 자리까지 출력합니다)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# 데이터 파일을 pandas dataframe으로 가지고 온다
liver_patients_df = pd.read_csv(PATIENT_FILE_PATH)

# 이 데이터 셋에 어떤 내용이 있는지 열 정보를 통해서 살펴볼게요.
liver_patients_df.columns

# Normalization할 열 이름들
features_to_normalize = ['Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase']

# 코드를 쓰세요
new_df = liver_patients_df[features_to_normalize]

scaler = preprocessing.MinMaxScaler()

normalization_data = scaler.fit_transform(new_df)

normalized_df = pd.DataFrame(normalization_data, columns=features_to_normalize)
# 체점용 코드
normalized_df.describe()


## 꺨끔 정답
from sklearn import preprocessing
import pandas as pd

PATIENT_FILE_PATH = './datasets/liver_patient_data.csv'
pd.set_option('display.float_format', lambda x: '%.5f' % x)

liver_patients_df = pd.read_csv(PATIENT_FILE_PATH)

features_to_normalize = ['Total_Bilirubin','Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase']

scaler = preprocessing.MinMaxScaler()

normalized_data = scaler.fit_transform(liver_patients_df[features_to_normalize])
normalized_df = pd.DataFrame(normalized_data, columns = features_to_normalize)

normalized_df.describe()



##############################################################################################################
## one-hot encoding 직접해보기
import pandas as pd

GENDER_FILE_PATH = 'C:/Users/wel27/Downloads/gender.csv'

gender_df = pd.read_csv(GENDER_FILE_PATH)
input_data = gender_df.drop(['Gender'], axis=1)

# 여기 코드를 쓰세요
X = pd.get_dummies(input_data)

# 체점용 코드
X.head()


#################################################################################################################
## 정규화
# L1 정규화 직접 해보기
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

import numpy as np
import pandas as pd

# 데이터 파일 경로 정의
INSURANCE_FILE_PATH = 'C:/Users/wel27/Downloads/insurance.csv'

insurance_df = pd.read_csv(INSURANCE_FILE_PATH)  # 데이터를 pandas dataframe으로 갖고 온다 (insurance_df.head()를 사용해서 데이터를 한 번 살펴보세요!)
insurance_df = pd.get_dummies(data=insurance_df, columns=['sex', 'smoker', 'region'])  # 필요한 열들에 One-hot Encoding을 해준다

# 입력 변수 데이터를 따로 새로운 dataframe에 저장
X = insurance_df.drop(['charges'], axis=1)

polynomial_transformer = PolynomialFeatures(4)  # 4 차항 변형기를 정의
polynomial_features = polynomial_transformer.fit_transform(X.values)  #  4차 항 변수로 변환

features = polynomial_transformer.get_feature_names(X.columns)  # 새로운 변수 이름들 생성

X = pd.DataFrame(polynomial_features, columns=features)  # 다항 입력 변수를 dataframe으로 만들어 준다
y = insurance_df[['charges']]  # 목표 변수 정의

# 여기 코드를 쓰세요
# train-test set 나누기
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Lasso 모델을 만들어 training set 학습
# 람다 -> 세타값이 커지는 것에 대해 얼만큼의 패널티를 줄것인가에 대한 변수 : 여기서는 alpha 파라미터
# 손실함수를 최소화 하기위해 경사하강법을 사용하는데 경사하강법을 최대한 몇 번 할지 정해줄 수 있다 : max_iter 파라미터
# feature scaling을 하면 경사하강법으로 극소점을 좀더 빨리 찾을수 있는데 
# lasso,ridge 모델은 자체적으로 가능 : normalize=True -> 모델을 학습시키기전에 데이터를 자동으로 0과1사이로 만들어준다
model = Lasso(alpha=0.01, max_iter=2000, normalize=True) # L1 정규화
# model = Ridge(alpha=0.001, max_iter=1000, normalize=True) # L2 정규화

model.fit(X_train, y_train)

## train, test set 성능 비교
# training set으로 예측값 계산
y_train_predict = model.predict(X_train)

# test set으로 예측값 계산
y_test_predict = model.predict(X_test)

# 체점용 코드
mse = mean_squared_error(y_train, y_train_predict)

print("training set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

mse = mean_squared_error(y_test, y_test_predict)

print("testing set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

# 4차 항의 높은 회귀 모델을 사용해도 성능이 training과 test 셋에서 큰 차이가 없는걸 확인할 수 있습니다.



#############################################################################################################
##############################################################################################################
## k겹 교차 검증 직접 해보기
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd

GENDER_FILE_PATH = 'C:/Users/wel27/Downloads/gender.csv'

# 데이터 셋을 가지고 온다
gender_df = pd.read_csv(GENDER_FILE_PATH)

X = pd.get_dummies(gender_df.drop(['Gender'], axis=1)) # 입력 변수를 one-hot encode한다
y = gender_df[['Gender']].values.ravel()

# 여기에 코드를 쓰세요
logistic_model = LogisticRegression(solver='saga', max_iter=2000) # saga도 경사 하강법을 변형한 하나의 최적화 방법

k_fold_score = np.average(cross_val_score(logistic_model, X, y, cv=5))
# 체점용 코드
k_fold_score


################################################################################################################
################################################################################################################
## 그리드 서치 직접 해보기
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

# 경고 메시지 출력 억제 코드
import warnings
warnings.simplefilter(action='ignore')

GENDER_FILE_PATH = 'C:/Users/wel27/Downloads/gender.csv'

# 데이터 셋을 가지고 온다
gender_df = pd.read_csv(GENDER_FILE_PATH)

X = pd.get_dummies(gender_df.drop(['Gender'], axis=1)) # 입력 변수를 one-hot encode한다
y = gender_df[['Gender']].values.ravel()

# 여기 코드를 쓰세요
hyper_parameter = {
    'penalty' : ['l1', 'l2'],
    'max_iter' : [500, 1000, 1500, 1500, 2000]
} 

model = LogisticRegression()

hyper_parameter_tunner = GridSearchCV(model, hyper_parameter, cv=5)
hyper_parameter_tunner.fit(X, y)

best_params = hyper_parameter_tunner.best_params_

# 체점용 코드
best_params



#########################################################################################################
#########################################################################################################
## scikit-learn 유방암 데이터 준비하기
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()
# 데이터 셋을 살펴보기 위한 코드
print(cancer_data.DESCR)

# 코드를 쓰세요
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()

# 실행 코드
X_train.head()


###############################################################################################################
##############################################################################################################
## 결정 트리로 악성/양성 유방암 분류하기
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()

# 저번 과제에서 쓴 데이터 준비 코드를 갖고 오세요
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()
# 코드를 쓰세요
model = DecisionTreeClassifier(max_depth=5, random_state=42) 
model.fit(X_train, y_train)

predictions = model.predict(X_test) 
score = model.score(X_test, y_test)
# 출력 코드
predictions, score



########################################################################################################
#######################################################################################################
## 랜덤 포레스트로 악성/양성 유방암 분류하기
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()

# 저번 챕터 유방암 데이터 준비하기 과제에서 쓴 코드를 갖고 오세요
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()
# 코드를 쓰세요
model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 모델을 통해 예측하기
predictions = model.predict(X_test) # 분류문제이기 때문에 0,1,2의 예측값이 나온다

# 분류모델 평가 -> 예측한 값들중에 몇 퍼센트가 맞게 분류됐는지 확인
score = model.score(X_test, y_test)

# 출력 코드
predictions, score


##########################################################################################################
#########################################################################################################
# 에다 부스트로 악성/양성 유방암 분류하기
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()

# 챕터 1 유방암 데이터 준비하기 과제에서 쓴 코드를 갖고 오세요
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()

# 코드를 쓰세요
model = AdaBoostClassifier(n_estimators=50, random_state=5) 
model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = model.score(X_test, y_test)

# 출력 코드
predictions, score



##############################################################################################################
#############################################################################################################
## 내용기반 추천 sklearn으로 유저 평점 예측하기
# 필요한 도구들을 가지고 오는 코드
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

# 유저 평점 + 영화 속성 데이터 경로 정의
MOVIE_DATA_PATH = 'C:/Users/wel27/Downloads/movie_rating.csv'

# pandas로 데이터 불러 오기
movie_rating_df = pd.read_csv(MOVIE_DATA_PATH)

features =['romance', 'action', 'comedy', 'heart-warming'] # 사용할 속성들 이름

# 입력 변수와 목표 변수 나누기
X = movie_rating_df[features]
y = movie_rating_df[['rating']]

# 입력 변수와 목표 변수들을 각각의 training/test 셋으로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 코드를 쓰세요.
model = LinearRegression()
model.fit(X_train, y_train)

y_test_predict = model.predict(X_test)
# 실행 코드
y_test_predict


##################################################################################################
##################################################################################################
## numpy로 유저 간 거리 구하기
import numpy as np
from math import sqrt


def distance(user_1, user_2):
    """유클리드 거리를 계산해주는 함수"""
    # 코드를 쓰세요 
    distance = sqrt(np.sum((user_1 - user_2) ** 2))
    return distance
    
    
# 실행 코드
user_1 = np.array([0, 1, 2, 3, 4, 5])
user_2 = np.array([0, 1, 4, 6, 1, 4])

distance(user_1, user_2)


## 필요한 메소드 
###
# np.sum 메소드
# numpy의 sum 메소드를 사용하면, numpy 배열 안에 있는 모든 원소들의 합을 리턴해줍니다. 예를 들어
vector_1 = np.array([1, 2, 3, 4, 5])
np.sum(vector_1)  # 15 리턴

# sqrt 함수
# 파이썬 math 모듈의 sqrt 함수는 주어진 숫자의 제곱근을 리턴해줍니다. 예를 들어:
sqrt(4)  # 2.0 리턴
sqrt(9)  # 3.0 리턴




###################################################################################################
### 협업필터링 이웃들 구하기
## 템플릿 코드 설명
'''
이번 레슨에서는 평점 데이터가 주어졌을 때 그 안에서 특정 유저와 비슷한 k 명의 이웃들을 구하는 함수, 
get_k_neighbors를 구현해보겠습니다.'''
import pandas as pd
import numpy as np
from math import sqrt

RATING_DATA_PATH = './ratings.csv'  # 받아올 평점 데이터 경로 정의

np.set_printoptions(precision=2)  # 소수점 둘째 자리까지만 출력

'''
distance 함수
지난 과제에서 구현했던, 두 유저 간의 유클리드 거리를 계산해주는 함수입니다. 
파라미터로는 두 유저의 평점 벡터를 받아서 거리를 리턴합니다.'''
def distance(user_1, user_2):
    """유클리드 거리를 계산해주는 함수"""
    return sqrt(np.sum((user_1 - user_2)**2))

'''
filter_users_without_movie
filter_users_without_movie 함수는 파라미터로 평점 데이터 행렬과 영화 번호를 받아서 
평점 데이터 행렬에서 해당 영화를 평가하지 않은 유저 정보를 미리 다 제거해 주는 함수입니다. 
유저의 이웃을 구하는데 이웃들이 원하는 영화에 평점을 안 줬으면 어짜피 사용할 수 없으니까 
미리 없애주기 위해 있습니다.'''
def filter_users_without_movie(rating_data, movie_id):
    """movie_id 번째 영화를 평가하지 않은 유저들은 미리 제외해주는 함수"""
    return rating_data[~np.isnan(rating_data[:,movie_id])]

'''
fill_nan_with_user_mean함수
평점 데이터 행렬의 빈값들을 각 유저의 평균 평점으로 채워 넣어주는 함수입니다. 
이 함수는 파라미터로 평점 데이터 행렬을 받고, 빈값들이 유저의 평균 평점으로 채워진 새로운 행렬을 리턴합니다.'''
def fill_nan_with_user_mean(rating_data):
    """평점 데이터의 빈값들을 각 유저 평균 값으로 체워주는 함수"""
    filled_data = np.copy(rating_data)  # 평점 데이터를 훼손하지 않기 위해 복사
    row_mean = np.nanmean(filled_data, axis=0)  # 유저 평균 평점 계산
    
    inds = np.where(np.isnan(filled_data))  # 비어 있는 인덱스들을 구한다
    filled_data[inds] = np.take(row_mean, inds[1])  #빈 인덱스를 유저 평점으로 채운다
    
    return filled_data


'''
과제: get_k_neighbors 함수'''
def get_k_neighbors(user_id, rating_data, k):
    """user_id에 해당하는 유저의 이웃들을 찾아주는 함수"""
    distance_data = np.copy(rating_data)  # 평점 데이터를 훼손하지 않기 위해 복사
    # 마지막에 거리 데이터를 담을 열 추가한다
    distance_data = np.append(distance_data, np.zeros((distance_data.shape[0], 1)), axis=1)
    
    # 코드를 쓰세요.
    
    # 데이터를 거리 열을 기준으로 정렬한다
    distance_data = distance_data[np.argsort(distance_data[:, -1])]
    
    # 가장 가까운 k개의 행만 리턴한다 + 마지막(거리) 열은 제외한다
    return distance_data[:k, :-1]

'''
과제로는 위 함수, get_k_neighbors를 구현합니다. 
get_k_neighbors 함수는 파라미터로 몇 번째 유저인지를 user_id로, 
평점 데이터를 rating_data로, 그리고 몇 명의 이웃들을 찾을지를 k로 받습니다. 
user_id는 그냥 각 행렬 안에서의 순서라고 생각하시면 됩니다. 
그리고 user_id의 유저와 가장 가까운 k 명의 유저 평점 데이터를 리턴하죠.

이미 작성된 코드를 설명드리면'''
distance_data = np.copy(rating_data)  # 평점 데이터를 훼손하지 않기 위해 복사
'''
distance_data에는 빈값이 없는 평점 데이터의 복사본이 저장돼있습니다.'''

# 맨 뒤 위치에 거리 데이터를 담을 열 추가한다
distance_data = np.append(distance_data, np.zeros((distance_data.shape[0], 1)), axis=1)

'''
그리고 이 복사본의 가장 뒤 열에는 각 행까지의 거리 정보를 저장할 새로운 열을 추가시켜줬죠. 
여러분이 작성하실 부분은, 이 새로운 열을 반복분을 통해서 체워넣는 겁니다. 
주의하셔야 될 점은 각 행의 마지막 열은 거리 정보를 저장하는 열이기 때문에 거리 계산에서 제외해 줘야 합니다.
반복문을 돌면서 user_id 번째 유저가 나올 때는, 해당 데이터의 거리 정보는 무한대, np.inf로 저장해 주시면 됩니다.
과제는 이 마지막 열들을 모두 user_id 번째 유저와의 거리로 채우는 거에서 끝나는데요. 이걸 채워 넣으면: 
그 뒤에서는'''

# 데이터를 거리 열을 기준으로 정렬한다
distance_data = distance_data[np.argsort(distance_data[:, -1])]
    
# 가장 가까운 k개의 행만 리턴한다 + 마지막(거리) 열은 제외한다
return distance_data[:k, :-1]

'''
평점 데이터를 거리 열을 기준으로 정렬한 후, 마지막 열은 제외하고 가장 가까운 k개의 행, 
그러니까 user_id 유저의 이웃들을 리턴해줍니다.'''




####################################################################################
####################################################################################
####### 과제
### 협업필터링 이웃들 구하기
import pandas as pd
import numpy as np
from math import sqrt

RATING_DATA_PATH = 'C:/Users/wel27/Downloads/ratings.csv'  # 받아올 평점 데이터 경로 정의

np.set_printoptions(precision=2)  # 소수점 둘째 자리까지만 출력

def distance(user_1, user_2):
    """유클리드 거리를 계산해주는 함수"""
    return sqrt(np.sum((user_1 - user_2)**2))
    
    
def filter_users_without_movie(rating_data, movie_id):
    """movie_id 번째 영화를 평가하지 않은 유저들은 미리 제외해주는 함수"""
    return rating_data[~np.isnan(rating_data[:,movie_id])]

    
def fill_nan_with_user_mean(rating_data):
    """평점 데이터의 빈값들을 각 유저 평균 값으로 체워주는 함수"""
    filled_data = np.copy(rating_data)  # 평점 데이터를 훼손하지 않기 위해 복사
    row_mean = np.nanmean(filled_data, axis=1)  # 유저 평균 평점 계산
    
    inds = np.where(np.isnan(filled_data))  # 비어 있는 인덱스들을 구한다
    filled_data[inds] = np.take(row_mean, inds[0])  #빈 인덱스를 유저 평점으로 채운다
    # np.take 함수는 인덱스를 이용해서 어레이의 요소를 가져옵니다.
    # np.where 함수는 조건에 맞는 인덱스 반환
    return filled_data

    
def get_k_neighbors(user_id, rating_data, k):
    """user_id에 해당하는 유저의 이웃들을 찾아주는 함수"""
    distance_data = np.copy(rating_data)  # 평점 데이터를 훼손하지 않기 위해 복사
    # 마지막에 거리 데이터를 담을 열 추가한다
    distance_data = np.append(distance_data, np.zeros((distance_data.shape[0], 1)), axis=1)

    '''
    # 코드를 쓰세요.
    for i in range(9):
        
        if user_id == i:
            distance_data[i,20] = np.inf
        else:
            distance_data[i,20] = distance(distance_data[user_id][:-1], distance_data[i][:-1])
            # distance_data[i][:-1] -> i행 전체열개수-1열'''
    # 코드를 쓰세요.
    for i in range(len(distance_data)):
        row = distance_data[i]
    
        if i == user_id:  # 같은 유저면 거리를 무한대로 설정
            row[-1] = np.inf
        
        else: 
            row[-1] = distance(distance_data[user_id][:-1], row[:-1])
      
    # 데이터를 거리 열을 기준으로 정렬한다
    distance_data = distance_data[np.argsort(distance_data[:, -1])]
    
    # 가장 가까운 k개의 행만 리턴한다 + 마지막(거리) 열은 제외한다
    return distance_data[:k, :-1]
  

# 실행 코드
# 영화 3을 본 유저들 중, 유저 0와 비슷한 유저 5명을 찾는다
rating_data = pd.read_csv(RATING_DATA_PATH, index_col='user_id').values  # 평점 데이터를 불러온다
filtered_data = filter_users_without_movie(rating_data, 3)  # 3 번째 영화를 보지 않은 유저를 데이터에서 미리 제외시킨다
filled_data = fill_nan_with_user_mean(filtered_data)  # 빈값들이 채워진 새로운 행렬을 만든다
user_0_neighbors = get_k_neighbors(0, filled_data, 5)  # 유저 0과 비슷한 5개의 유저 데이터를 찾는다
user_0_neighbors



###########################################################################################################
###########################################################################################################
## 협업 필터링 유저 평점 예측하기
import pandas as pd
import numpy as np
from math import sqrt

RATING_DATA_PATH = 'C:/Users/wel27/Downloads/ratings.csv'  # 받아올 평점 데이터 경로 정의

np.set_printoptions(precision=2)  # 소수점 둘째 자리까지만 출력

def distance(user_1, user_2):
    """유클리드 거리를 계산해주는 함수"""
    return sqrt(np.sum((user_1 - user_2)**2))
    
    
def filter_users_without_movie(rating_data, movie_id):
    """movie_id 번째 영화를 평가하지 않은 유저들은 미리 제외해주는 함수"""
    return rating_data[~np.isnan(rating_data[:,movie_id])]
    
    
def fill_nan_with_user_mean(rating_data):
    """평점 데이터의 빈값들을 각 유저 평균 값으로 체워주는 함수"""
    filled_data = np.copy(rating_data)  # 평점 데이터를 훼손하지 않기 위해 복사
    row_mean = np.nanmean(filled_data, axis=1)  # 유저 평균 평점 계산
    
    inds = np.where(np.isnan(filled_data))  # 비어 있는 인덱스들을 구한다
    filled_data[inds] = np.take(row_mean, inds[0])  #빈 인덱스를 유저 평점으로 채운다
    
    return filled_data
    
    
def get_k_neighbors(user_id, rating_data, k):
    """user_id에 해당하는 유저의 이웃들을 찾아주는 함수"""
    distance_data = np.copy(rating_data)  # 평점 데이터를 훼손하지 않기 위해 복사
    # 마지막에 거리 데이터를 담을 열 추가한다
    distance_data = np.append(distance_data, np.zeros((distance_data.shape[0], 1)), axis=1)
    
    # 코드를 쓰세요.
    for i in range(len(distance_data)):
        row = distance_data[i]
        
        if i == user_id:  # 같은 유저면 거리를 무한대로 설정
            row[-1] = np.inf
        else:  # 다른 유저면 마지막 열에 거리 데이터를 저장
            row[-1] = distance(distance_data[user_id][:-1], row[:-1])
    
    # 데이터를 거리 열을 기준으로 정렬한다
    distance_data = distance_data[np.argsort(distance_data[:, -1])]
    
    # 가장 가까운 k개의 행만 리턴한다 + 마지막(거리) 열은 제외한다
    return distance_data[:k, :-1]
    
def predict_user_rating(rating_data, k, user_id, movie_id,):
    """예측 행렬에 따라 유저의 영화 평점 예측 값 구하기"""
    # movie_id 번째 영화를 보지 않은 유저를 데이터에서 미리 제외시킨다
    filtered_data = filter_users_without_movie(rating_data, movie_id)
    # 빈값들이 채워진 새로운 행렬을 만든다
    filled_data = fill_nan_with_user_mean(filtered_data)
    # 유저 user_id와 비슷한 k개의 유저 데이터를 찾는다
    neighbors = get_k_neighbors(user_id, filled_data, k)
    
    # 코드를 쓰세요
    return np.mean(neighbors[:, movie_id])
    
    
# 실행 코드   
# 평점 데이터를 불러온다
rating_data = pd.read_csv(RATING_DATA_PATH, index_col='user_id').values
# 5개의 이웃들을 써서 유저 0의 영화 3에 대한 예측 평점 구하기
predict_user_rating(rating_data, 5, 0, 3)  

            

## 오류
'''
neighbors[:][movie_id]와 같이 대괄호를 연달아 인덱싱하면 neighbors[:]로 먼저 인덱싱 하고, 
그 결과를 다시 [movie_id]로 인덱싱합니다. 
즉 A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 에서 A[:][1]을 쓴다면 먼저 A[:]가 먼저 실행되어 결국 
A전체가 결과가 되고(어차피 모든 행 = 자기 자신), 이를 다시 A[1]로 수행하여 ([4, 5, 6])을 반환합니다. 

이에 반해 neighbors[:, movie_id]와 같이 콤마를 사용할 경우, 
원하는 방식대로 모든 행의 movie_id열의 원소의 array를 반환합니다.  위의 A에 실행하면
A[:, 1]은 모든 행의 1열의 array를 반환하니 ([2, 5, 8])로 원하는 결과를 얻습니다'''


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

A[:][1]
A[:,1]



############################################################################################################
############################################################################################################
## 행렬 인수분해 손실 함수 구현해보기
import numpy as np
'''
알아야 할 함수
np.nansum
np.nansum 함수는 np.nan을 제외하고 행렬에 있는 모든 원소의 합을 리턴합니다. 예를 들면'''
data = np.array([
    [0, 1, 1, 2, 3, np.nan],
    [np.nan, 1, 1, 2, 1, -1],
])
np.nansum(data)  # 11을 리턴합니다


'''
과제: cost 함수
cost 함수는 파라미터로 에측 값 행렬 데이터 prediction과 실제 평점 데이터 rating을 받습니다. 
그리고 두 행렬의 제곱 오차 합을 리턴합니다.'''
import numpy as np

def cost(prediction, R):
    """행렬 인수분해 알고리즘의 손실을 계산해주는 함수"""
    # 코드를 쓰세요
    error = prediction - R
    squre_error = (error)**2
    cost = np.nansum(squre_error)
    return cost

   # return np.nansum((prediction - R)**2)
                
    
# 실행 코드


# 예측 값 행렬
prediction = np.array([
    [4, 4, 1, 1, 2, 2],
    [4, 4, 3, 1, 5, 5],
    [2, 2, 1, 1, 3, 4],
    [1, 3, 1, 4, 2, 2],
    [1, 2, 4, 1, 2, 5],
    ])

# 실제 값 행렬
R = np.array([
    [3, 4, 1, np.nan, 1, 2],
    [4, 4, 3, np.nan, 5, 3],
    [2, 3, np.nan, 1, 3, 4],
    [1, 3, 2, 4, 2, 2],
    [1, 2, np.nan, 2, 2, 4],
    ])

cost(prediction, R)


############################################################################################################
############################################################################################################
## 유저/속성 행렬 초기화 해보기
import numpy as np
'''
미리 알아야 할 함수
np.random.rand
np.random.rand 함수는 원하는 차원의 행렬을 임의로 생성해 주는 함수입니다. 
차원이 3x5 (3행 5열이 있는)인 행렬을 임의로 생성하고 싶다면'''
np.random.rand(3, 5)

random_matrix = np.random.rand(3, 5) # 변수에 저장

'''
템플릿 코드
필요한 도구 임포트, 임의성 컨트롤'''
import numpy as np

# 채점을 위해 numpy에서 임의성 도구들의 결과가 일정하게 나오도록 해준다
np.random.seed(5)

# 먼저 이번 과제에서 사용할 numpy를 임포트 해옵니다. 
# 그다음은 np.random.seed 함수를 사용해서 임의성을 사용하는 
# numpy의 모든 기능들이 코드를 돌릴 때마다 똑같이 나오게 합니다. 
# 실제로 돌릴 때마다 값이 다르게 나오면 채점을 할 수 없겠죠?

'''
initialize 함수'''
def initialize(R, num_features):
    """임의로 유저 취향과 상품 속성 행렬들을 만들어주는 함수"""
    num_users, num_items = R.shape  # 유저 데이터 개수와 영화 개수를 변수에 저장
    
    # 코드를 쓰세요.
    
    return Theta, X

'''
그다음은 initialize 함수를 살펴봅시다. 
initialize 함수는 저희가 작성하고 싶은 함수인데요. 
파라미터로는 평점 행렬과 속성 개수를 받아서, 임의로 만든 유저 취향과 영화 속성 행렬을 리턴합니다. 
살펴보면 이미 코드가 조금 있죠? 위 부분은 생성시킬 행렬들의 차원 데이터를 변수에 저장하는 코드입니다. 
그리고 여기서 유저 취향 행렬은 변수 Theta에, 영화 속성 행렬은 X에 저장시켜주셔야 하는데요. 
마지막에는 이 두 행렬을 리턴합니다.'''

#####################################################################################################3
'''
과제: initialize 함수 채우기
유저 취향 행렬: 유저 데이터 개수 x 속성 개수 차원의 임의 행렬로 만든다 (변수 Theta에 저장)
영화 속성 행렬: 속성 개수 x 영화 데이터 개수 차원의 임의 행렬로 만든다 (변수 X에 저장)'''

import numpy as np

# 체점을 위해 numpy에서 임의성 도구들의 결과가 일정하게 나오도록 해준다
np.random.seed(5)

def initialize(R, num_features):
    """임의로 유저 취향과 상품 속성 행렬들을 만들어주는 함수"""
    num_users, num_items = R.shape  # 유저 데이터 개수와 영화 개수를 변수에 저장
    
    # 코드를 쓰세요.
    Theta = np.random.rand(num_users, num_features)
    X = np.random.rand(num_features, num_items)

    return Theta, X
    
    
# 실제 값 행렬
R = np.array([
    [3, 4, 1, np.nan, 1, 2],
    [4, 4, 3, np.nan, 5, 3],
    [2, 3, np.nan, 1, 3, 4],
    [1, 3, 2, 4, 2, 2],
    [1, 2, np.nan, 2, 2, 4],
    ])
    
    
Theta, X = initialize(R, 2)
Theta, X


#######################################################################################################
########################################################################################################
### 행렬 인수분해 경사 하강법 구현해보기

## 템플릿 코드 설명
# 임포트 & 임의성
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 체점을 위해 임의성을 사용하는 numpy 도구들의 결과가 일정하게 나오도록 해준다
np.random.seed(5)

'''
먼저 사용할 도구들을 임포트합니다. 
구현을 위해서 pandas와 numpy를, 손실 시각화를 위해서 matplotlib을 가지고 옵니다.
저번 과제와 마찬가지로 채점을 위해서 numpy의 임의성 도구들의 결과를 일정하게 나오게 하는
np.random.seed 함수도 포함합니다.'''

# 필요한 함수들
def predict(Theta, X):
    """유저 취향과 상품 속성을 곱해서 예측 값을 계산하는 함수"""
    return Theta @ X


def cost(prediction, R):
    """행렬 인수분해 알고리즘의 손실을 계산해주는 함수"""
    return np.nansum((prediction - R)**2)


def initialize(R, num_features):
    """임의로 유저 취향과 상품 속성 행렬들을 만들어주는 함수"""
    num_users, num_items = R.shape
    
    Theta = np.random.rand(num_users, num_features)
    X = np.random.rand(num_features, num_items)
    
    return Theta, X

'''나머지는 저희가 경사 하강 함수를 작성하면서 사용하는 함수들인데요. 
cost는 손실을 계산해 주는 함수, initialize는 임의로 유저 취향과 영화 속성 행렬을 초기화해주는 함수입니다.
predict 함수는 처음 보셨을 텐데요. 
그냥 유저 취향과 영화 속성 행렬들을 받아서 곱해주는 (예측 값들을 계산해 주는) 함수입니다.'''


# gradient_descent 함수
'''
마지막은 저희가 구현할 gradient_descent 함수입니다. 
파라미터로는 평점 데이터 행렬 R, 유저 취향 행렬 Theta, 영화 속성 행렬 X, 경사 하강 횟수 iteration, 학습률 alpha, 
그리고 정규화 상수 lambda_를 받습니다. 
이 파라미터들을 이용해서 유저 취향 행렬과 영화 속성 행렬을 업데이트하죠'''

def gradient_descent(R, Theta, X, iteration, alpha, lambda_):
    """행렬 인수분해 경사 하강 함수"""
    num_user, num_items = R.shape
    num_features = len(X)
    costs = []

'''가장 먼저 유저 데이터, 영화 데이터, 속성의 개수를 파악합니다. 
그리고 경사 하강을 할 때마다 손실을 계산할 건데요. 이걸 저장할 파이썬 리스트를 만들어 줍니다 (costs).'''
    for _ in range(iteration):
        prediction = predict(Theta, X)
        error = prediction - R
        costs.append(cost(prediction, R))

'''
그다음은 경사 하강을 하고 싶은 만큼 반복문을 돕니다. 
한 번 경사 하강을 할 때마다 예측 값을 계산하고 (prediction), 
원소 별 예측 값과 실제 값의 오차를 저장하는 행렬을 계산하고 (error) 
마지막으로는 손실을 costs 리스트에 추가해 줍니다.'''
        for i in range(num_user):
            for j in range(num_items):
                if not np.isnan(R[i][j]):
                    for k in range(num_features):
                        # 아래 코드를 채워 넣으세요.
                        Theta[i][k] -= 
                        X[k][j] -= 
                        
    return Theta, X, costs
'''
마지막 부분은 그냥 모든 유저, 모든 영화 데이터, 모든 속성에 대해서 다 도는 반복문인데요. 
if not np.isnan(R[i][j]): 실제 데이터 평점이 없는 값들에 대해서는 건너뛰어 주는 거 보이시죠'''

#----------------------실행(채점) 코드----------------------
# 평점 데이터를 가지고 온다
ratings_df = pd.read_csv(RATING_DATA_PATH, index_col='user_id')

# 평점 데이터에 mean normalization을 적용한다
for row in ratings_df.values:
    row -= np.nanmean(row)
       
R = ratings_df.values
        
Theta, X = initialize(R, 5)  # 행렬들 초기화
Theta, X, costs = gradient_descent(R, Theta, X, 200, 0.001, 0.01)  # 경사 하강
    
# 손실이 줄어드는 걸 시각화 하는 코드 (디버깅에 도움이 됨)
# plt.plot(costs)

Theta, X

'''
실행 코드의 대부분은 그냥 평점 데이터를 읽어 오고, 데이터에 mean normalization을 적용하고, 
행렬들을 초기화해서 경사 하강법을 초기화하는 내용인데요. 채점 자체는 Theta와 X를 통해서 됩니다. 
(경사 하강은 학습률 0.001, 정규화 항 0.01로 실행합니다)
마지막 줄 plt.plot(costs)을 주피터 환경에서 실행하시면 경사 하강을 하면서 저장한 손실이 그래프로 출력되는데요.'''





#######################################################################################################
########################################################################################################
### 행렬 인수분해 경사 하강법 구현해보기(실제 코드)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 체점을 위해 임의성을 사용하는 numpy 도구들의 결과가 일정하게 나오도록 해준다
np.random.seed(5)
RATING_DATA_PATH = 'C:/Users/wel27/Downloads/ratings.csv'  # 데이터 파일 경로 정의
# numpy 출력 옵션 설정
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def predict(Theta, X):
    """유저 취향과 상품 속성을 곱해서 예측 값을 계산하는 함수"""
    return Theta @ X


def cost(prediction, R):
    """행렬 인수분해 알고리즘의 손실을 계산해주는 함수"""
    return np.nansum((prediction - R)**2)


def initialize(R, num_features):
    """임의로 유저 취향과 상품 속성 행렬들을 만들어주는 함수"""
    num_users, num_items = R.shape
    
    Theta = np.random.rand(num_users, num_features)
    X = np.random.rand(num_features, num_items)
    
    return Theta, X


def gradient_descent(R, Theta, X, iteration, alpha, lambda_):
    """행렬 인수분해 경사 하강 함수"""
    num_user, num_items = R.shape
    num_features = len(X)
    costs = []
        
    for _ in range(iteration):
        prediction = predict(Theta, X)
        error = prediction - R
        costs.append(cost(prediction, R))
                          
        for i in range(num_user):
            for j in range(num_items):
                if not np.isnan(R[i][j]):
                    for k in range(num_features):
                        # 아래 코드를 채워 넣으세요.
                        Theta[i][k] -= alpha * (np.nansum(error[i, :]*X[k, :]) + lambda_*Theta[i][k])
                        X[k][j] -= alpha * (np.nansum(error[:, j]*Theta[:, k]) + lambda_*X[k][j])
    return Theta, X, costs


#----------------------실행(채점) 코드----------------------
# 평점 데이터를 가지고 온다
ratings_df = pd.read_csv(RATING_DATA_PATH, index_col='user_id')

# 평점 데이터에 mean normalization을 적용한다
for row in ratings_df.values:
    row -= np.nanmean(row)
       
R = ratings_df.values
        
Theta, X = initialize(R, 5)  # 행렬들 초기화
Theta, X, costs = gradient_descent(R, Theta, X, 200, 0.001, 0.01)  # 경사 하강
    
# 손실이 줄어드는 걸 시각화 하는 코드 (디버깅에 도움이 됨)
# plt.plot(costs)

Theta, X


# 파라미터로는 평점 데이터 행렬 R, 유저 취향 행렬 Theta, 영화 속성 행렬 X, 경사 하강 횟수 iteration,
# 학습률 alpha, 그리고 정규화 상수 lambda_를 받습니다. 
    
    
    
    
    
    
    
a = error
error[0, :]    # 에러 한줄(행)
    
error[:, 0]    # 에러 한줄(열)
    



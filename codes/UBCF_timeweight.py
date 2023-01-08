import pandas as pd
import numpy as np
from tqdm import tqdm

# Read rating data
ratings = pd.read_csv('/home/sjkim/recommendSystem/finalproject/Amazon_ratings.csv', encoding='latin-1')
ratings

#ratings['timestamp'] = ratings['timestamp'].astype('int16')
#ratings['rating'] = ratings['rating'].astype('int16')

# Rating 데이터를 test, train으로 나누고 train을 full matrix로 변환
from sklearn.model_selection import train_test_split
x = ratings.copy()
y = ratings['user_id']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
rating_matrix = x_train.pivot(values='rating', index='user_id', columns='item_id') 

# RMSE 계산을 위한 함수
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model, neighbor_size=20):
    id_pairs = zip(x_test['user_id'], x_test['item_id'])
    y_pred = np.array([model(user, item, neighbor_size) for (user, item) in tqdm(id_pairs)])
    y_true = np.array(x_test['rating'])
    return RMSE(y_true, y_pred)

# 모든 가능한 사용자 pair의 Cosine similarities 계산
from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# 모든 user의 rating 평균 계산 
rating_mean = rating_matrix.mean(axis=1) # axis=1은 유저별 평균 

def ubcf_sig_weighting(user_id, item_id, neighbor_size=20):
    import numpy as np
    # 현 user의 평균 가져오기
    user_mean = rating_mean[user_id]
    if item_id in rating_matrix:
        sim_scores = user_similarity[user_id]           # 현 user와 다른 사용자 간의 유사도 가져오기
        t_gap = time_gap[user_id]                       # 현 user와 다른 사용자 간의 time gap 가져오기
        item_ratings = rating_matrix[item_id]           # 현 item의 rating 가져오기. 즉, rating_matrix의 열(크기: 3706)을 추출
        others_mean = rating_mean                       # 모든 사용자의 rating 평균 가져오기
        common_counts = sig_counts[user_id]             # 현 user와 다른 사용자 간의 공통 rating개수 가져오기
        no_rating = item_ratings.isnull()               # 현 item에 대한 rating이 없는 user 선택
        low_significance = common_counts < SIG_LEVEL    # 공통으로 평가한 영화의 수가 SIG_LEVEL보다 낮은 사람 선택
        too_far = t_gap > TIME_GAP                      # 영화의 평가시점이 너무 먼 사람을 선택
        none_rating_idx = item_ratings[no_rating | low_significance | too_far].index  # 평가를 안했거나, SIG_LEVEL, 평가시점이 기준 이하인 user 제거. 3가지 중 하나라도 걸리면 제거
        item_ratings = item_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        others_mean = others_mean.drop(none_rating_idx)
        if len(item_ratings) > MIN_RATINGS:     # 충분한 rating이 있는지 확인
            if neighbor_size == 0:              # Neighbor size가 지정되지 않은 경우
                item_ratings = item_ratings - others_mean                       # 편차로 예측치 계산
                prediction = np.dot(sim_scores, item_ratings) / sim_scores.sum()
                prediction = prediction + user_mean                             # 예측값에 현 사용자의 평균 더하기
            else:                               # Neighbor size가 지정된 경우
                neighbor_size = min(neighbor_size, len(sim_scores))             # 지정된 neighbor size 값과 해당 영화를 평가한 총사용자 수 중 작은 것으로 결정
                sim_scores = np.array(sim_scores)                               # array로 바꾸기 (argsort를 사용하기 위함)
                item_ratings = np.array(item_ratings)
                others_mean = np.array(others_mean)
                user_idx = np.argsort(sim_scores)                               # 유사도를 순서대로 정렬 
                sim_scores = sim_scores[user_idx][-neighbor_size:]              # 유사도, rating, 평균값을 neighbor size만큼 받기
                item_ratings = item_ratings[user_idx][-neighbor_size:]
                others_mean = others_mean[user_idx][-neighbor_size:]
                item_ratings = item_ratings - others_mean                   # 편차로 예측치 계산
                if sim_scores.sum() > 0.0000001:
                    prediction = np.dot(sim_scores, item_ratings) / sim_scores.sum() + user_mean
                else:
                    prediction= user_mean
        else:
            prediction = user_mean
    else:
        prediction = user_mean
    if prediction > 5:
        prediction = 5
    if prediction < 1:
        prediction = 1
    return prediction

    # 각 rating의 "시간"을 기록한 full matrix 생성 (3706x6368)
time_matrix = x_train.pivot(values='timestamp', index='user_id', columns='item_id')

# 각 사용자 쌍의 공통 rating 수(significance level)를 집계하기 위한 함수
def count_num():       # matrix 연산 이용
    rating_flag1 = np.array((rating_matrix > 0).astype(float))      # 각 user의 rating 영화를 1로 표시
    rating_flag2 = rating_flag1.T
    counts = np.dot(rating_flag1, rating_flag2)                     # 사용자별 공통 rating 수 계산
    return counts
    
# 각 사용자의 rating "시간의 차이의 평균"을 계산
# for loop 사용
from tqdm import tqdm 
def time_gap_calc1():
    time_gap = np.zeros(np.shape(user_similarity))
    tg_matrix = time_matrix.T                  # 평가 시점 데이터 가져오기, time_matrix를 transpose 시킨 것
    for i, user in tqdm(enumerate(tg_matrix)):
        for j, other in enumerate(tg_matrix):  # i,j로 가능한 모든 조합이 뽑힘
            #두 사용자 간에 공통으로 평가한 영화에 대한 time stamp 차이의 평균 계산
            time_gap[i,j] = np.nanmean(abs((tg_matrix[user] - tg_matrix[other]))) # 현재user에서 다른 user를 뺀 것
            # nan인건 nan으로 냅두기 0으로 해버리면 그건 0으로 빼버리는거니까 값이 달라짐 그래서 nanmean을 사용(nan 무시하고 빼버리기)
    return time_gap

# Numpy matrix 연산 사용
def time_gap_calc2():
    tg_matrix = np.array(time_matrix)
    return np.nanmean(np.abs(tg_matrix[np.newaxis,:,:] - tg_matrix[:,np.newaxis,:]), axis=2)

print('Running sig_counts')
sig_counts = count_num()
sig_counts = pd.DataFrame(sig_counts, index=rating_matrix.index, columns=rating_matrix.index).fillna(0)

print('Running time_gap calculation')
time_gap = time_gap_calc1()
time_gap = pd.DataFrame(time_gap, index=time_matrix.index, columns=time_matrix.index).fillna(0)
time_gap.to_csv('time_gap.csv')
#time_gap = pd.read_csv('time_gap.csv')

SIG_LEVEL = 3       # minimum significance level 지정. 공통적으로 평가한 아이템의 수
MIN_RATINGS = 2     # 예측치 계산에 사용할 minimum rating 수 지정
TIME_GAP = 16000000 # 평가한 시점이 얼마 이상 차이가 날때 제외할지에 대한 기준

print('Running model')
print(score(ubcf_sig_weighting, 45))
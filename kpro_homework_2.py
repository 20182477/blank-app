'''#1. 패키지 호출
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import streamlit as st
import matplotlib.font_manager as fm
import platform

script_dir = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(script_dir, 'NanumGothic (1).ttf')
fm.fontManager.addfont(FONT_PATH)
font_name = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

#2. 데이터 호출 및 일데이터 추출, 프레임화
df = pd.read_csv('data_울산_2024.csv', encoding='cp949', parse_dates=True)
date_col = next(c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]))
df.set_index(date_col, inplace=True)
daily = df.resample('D').first().dropna()

# 자동으로 날짜 컬럼 찾기
date_col = None
for col in df.columns:
    try:
        df[col] = pd.to_datetime(df[col])
        date_col = col
        break
    except Exception:
        continue

if date_col is None:
    raise ValueError("날짜 컬럼을 찾을 수 없습니다. 파일의 날짜 컬럼명을 직접 지정해주세요.")

df.set_index(date_col, inplace=True)

daily = df.resample('D').first()
print(daily)

#3.일별(00:00) 리샘플링 및 결측 제거
daily = df.resample('D').first()
daily_clean = daily.dropna()
print(daily_clean)

#4.히트맵 그리기 (matplotlib만 사용)
corr = daily_clean.corr()
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, aspect='auto')        # imshow 반환값 저장
cbar = fig.colorbar(im, ax=ax)             # fig.colorbar에 im과 ax 지정
ax.set_xticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr)))
ax.set_yticklabels(corr.columns)
ax.set_title('Variable Correlation Heatmap')
plt.tight_layout()
plt.show()
st.pyplot(fig1) 

#5. 학습 데이터셋 준비
target = '울산권_온산(정) 배수지 탁도'
X = daily_clean.drop(columns=[target])
y = daily_clean[target]
test_size = st.sidebar.slider(
    '테스트 세트 비율(test_size)', 
    min_value=0.1, 
    max_value=0.5, 
    value=0.2, 
    step=0.05
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
st.write(f"선택된 test_size = {test_size:.2f}")
st.write(f"학습 데이터: {len(X_train)}, 테스트 데이터: {len(X_test)}")

#6.모델 학습(XGboost, Random forest) 및 성능(R²) 확인
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    #'LGBM': LGBMRegressor(random_state=42)
}

#7.실제 배수지 탁도 vs 예측 탁도값 비교 그래프 시각화
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} R2 score: {r2:.4f}')
    predictions[name] = y_pred

    plt.figure()
plt.plot(y_test.values, label='Actual')
for name, y_pred in predictions.items():
    plt.plot(y_pred, label=name)
plt.legend()
plt.title('Actual vs Predicted Turbidity')
plt.xlabel('타임테이블')
plt.ylabel('탁도')
plt.tight_layout()
plt.show()
st.pyplot(fig2)'''

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

st.title("울산 탁도 예측 대시보드")

# --- 1. 데이터 로드 & 날짜 컬럼 탐색 ---
df = pd.read_csv('data_울산_2024.csv', encoding='cp949')

date_col = None
for col in df.columns:
    converted = pd.to_datetime(df[col], errors='coerce')
    # 80% 이상이 유효 날짜인 컬럼을 날짜로 간주
    if converted.notna().mean() > 0.8:
        df[col] = converted
        date_col = col
        break

if date_col is None:
    st.error("날짜 컬럼을 자동으로 찾지 못했습니다. CSV를 확인해주세요.")
    st.stop()

st.write(f"**날짜 컬럼**으로 지정된: `{date_col}`")
df.set_index(date_col, inplace=True)

# --- 2. 일별 리샘플링 & 결측 제거 ---
daily_clean = df.resample('D').first().dropna()
st.write("일별 리샘플링 후 데이터", daily_clean.shape)

# --- 3. 타겟(탁도) 컬럼 자동 감지 ---
turb_cols = [c for c in daily_clean.columns if '탁도' in c]
if not turb_cols:
    st.error("탁도 컬럼을 찾을 수 없습니다.")
    st.stop()
target = turb_cols[0]
st.write(f"**타겟 컬럼**: `{target}`")

# --- 4. 사이드바: test_size 슬라이더 ---
test_size = st.sidebar.slider(
    '테스트 세트 비율', 
    min_value=0.1, max_value=0.5, value=0.2, step=0.05
)

# --- 5. 학습/테스트 분할 & 개수 출력 ---
X = daily_clean.drop(columns=[target])
y = daily_clean[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
st.write(f"- 학습 샘플 수: {len(X_train)}")  
st.write(f"- 테스트 샘플 수: {len(X_test)}")

# --- 6. 히트맵 그리기 & 표시 ---
corr = daily_clean.corr()
fig1, ax1 = plt.subplots(figsize=(12, 10))
im = ax1.imshow(corr, aspect='auto')
fig1.colorbar(im, ax=ax1)
ax1.set_xticks(range(len(corr))); ax1.set_xticklabels(corr.columns, rotation=90)
ax1.set_yticks(range(len(corr))); ax1.set_yticklabels(corr.columns)
ax1.set_title("Variable Correlation Heatmap")
plt.tight_layout()
st.pyplot(fig1)

# --- 7. 모델 정의, 학습, 예측 & R² ---
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LGBM': LGBMRegressor(
        num_leaves=64,
        max_depth=10,
        min_data_in_leaf=10,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42
    )
}

st.write("### 모델별 R² 점수")
preds = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    score = r2_score(y_test, y_pred)
    st.write(f"- **{name}**: R² = {score:.4f}")
    preds[name] = y_pred

# --- 8. 실제 vs 예측 그래프 & 표시 ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(y_test.values, label='Actual', marker='o')
for name, y_pred in preds.items():
    ax2.plot(y_pred, label=name, marker='x')
ax2.legend()
ax2.set_title("Actual vs Predicted Turbidity")
ax2.set_xlabel("샘플 인덱스")
ax2.set_ylabel("탁도")
plt.tight_layout()
st.pyplot(fig2)

#1. 패키지 호출
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

#2. rawdata 호출
df = pd.read_csv('data_울산_2024.csv', encoding='cp949')

date_col = None
for col in df.columns:
    converted = pd.to_datetime(df[col], errors='coerce')
    if converted.notna().mean() > 0.8:
        df[col] = converted
        date_col = col
        break

if date_col is None:
    st.error("날짜 컬럼을 자동으로 찾지 못했습니다. CSV를 확인해주세요.")
    st.stop()

st.write(f"**날짜 컬럼**으로 지정된: `{date_col}`")
df.set_index(date_col, inplace=True)

#3. 이상데이터 제거
daily_clean = df.resample('D').first().dropna()
st.write("일별 리샘플링 후 데이터", daily_clean.shape)

#4. 타겟 데이터 확인
turb_cols = [c for c in daily_clean.columns if '배수지 탁도' in c]
if not turb_cols:
    st.error("탁도 컬럼을 찾을 수 없습니다.")
    st.stop()
target = turb_cols[0]
st.write(f"**타겟 컬럼**: `{target}`")

#5.테스트세트 슬라이더 생성 및 학습데이터 분할
test_size = st.sidebar.slider(
    '테스트 세트 비율', 
    min_value=0.1, max_value=0.5, value=0.15, step=0.05)
X = daily_clean.drop(columns=[target])
y = daily_clean[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42)
st.write(f"- 학습 샘플 수: {len(X_train)}")  
st.write(f"- 테스트 샘플 수: {len(X_test)}")

#6. 히트맵
corr = daily_clean.corr()
fig1, ax1 = plt.subplots(figsize=(12, 10))
im = ax1.imshow(corr, aspect='auto')
fig1.colorbar(im, ax=ax1)
ax1.set_xticks(range(len(corr))); ax1.set_xticklabels(corr.columns, rotation=90)
ax1.set_yticks(range(len(corr))); ax1.set_yticklabels(corr.columns)
ax1.set_title("Variable Correlation Heatmap")
plt.tight_layout()
st.pyplot(fig1)

#7.학습모델 정의, 학습, 예측 & R² 성능지표
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

#8.타겟 데이터 예측값vs실측값
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

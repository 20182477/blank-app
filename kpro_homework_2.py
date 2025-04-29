#1. 패키지 호출
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import streamlit as st
import platform

if platform.system() == "Darwin":  #
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")

fe = fm.FontEntry(
    fname=r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # ttf 파일이 저장되어 있는 경로
    name="NanumGothic",
)  # 원하는 폰트 설정
fm.fontManager.ttflist.insert(0, fe)  # Matplotlib에 폰트 추가

plt.rcParams.update({"font.size": 18, "font.family": "NanumGothic"})  # 폰트 설정

plt.rcParams["axes.unicode_minus"] = False

#2. 데이터 호출 및 일데이터 추출, 프레임화
df = pd.read_csv('data_울산_2024.csv', encoding="cp949")
print(df)

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

#3. 결측치 제거
daily_clean = daily.dropna()
print(daily_clean)

#4. Correalation Heatmap
corr = daily_clean.corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Variable Correlation Heatmap')
plt.tight_layout()
plt.show()

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

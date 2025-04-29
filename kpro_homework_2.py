import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
import streamlit
plt.rcParams['font.family'] = 'Malgun Gothic' 


df = pd.read_csv('data_울산_2024.csv', encoding="cp949")

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

daily_clean = daily.dropna()

corr = daily_clean.corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr, aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Variable Correlation Heatmap')
plt.tight_layout()
plt.show()

target = '울산권_온산(정) 배수지 탁도'
X = daily_clean.drop(columns=[target])
y = daily_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LGBM': LGBMRegressor(random_state=42)
}

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
plt.xlabel('타임테이블블')
plt.ylabel('탁도')
plt.tight_layout()
plt.show()
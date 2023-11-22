import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import xgboost as xgb
import time

def read_data(file_path):
    # 使用 pandas 讀取資料
    data = pd.read_csv(file_path)  # 假設資料以空格分隔
    return data

def handle_missing_values(data):
    # 刪除包含缺失值的資料列
    data = data.dropna()
    return data

def split_attributes(data):
    # 將資料分成 class attribute 和 feature attribute
    class_attribute_name = data.columns[13]
    class_attribute = data[class_attribute_name]
    feature_attributes = data.drop(columns=class_attribute_name)

    return class_attribute_name, class_attribute, feature_attributes

def transfer(data):
    # 正規化
    # 創建MinMaxScaler對象
    regular = MinMaxScaler()
    # 將數據進行最小-最大正規化
    data = pd.DataFrame(regular.fit_transform(data))

    return data

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":

    # 訓練集資料前處理------------------------------------------------------------------------------------------------- #
    file_path = './BostonHousing.csv'
    data = read_data(file_path)

    # 處理缺失值
    data = handle_missing_values(data)

    # 分離類別屬性和特徵屬性
    class_attribute_name, class_attribute, feature_attributes = split_attributes(data)

    feature_attributes = transfer(feature_attributes)

    # print("Class Attribute Name:", class_attribute_name)
    # print("Class Attribute:\n", class_attribute)
    # print("Feature Attributes:")
    # print(feature_attributes)
    # print(data)


    # XGBoost方法------------------------------------------------------------------------------------------------- #
    start = time.time()

    # 將數據轉換為 XGBoost 的 DMatrix 格式
    data_dmatrix = xgb.DMatrix(data=feature_attributes, label=class_attribute)

    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    end = time.time()

    # 創建 KFold 對象，指定 k 值
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # 使用 cross_val_score 函數進行 K-fold 交叉驗證
    # estimator 是你的模型，X 和 y 是數據和標籤，cv 是 KFold 對象
    scores_rmse = cross_val_score(xg_reg, feature_attributes, class_attribute, scoring='neg_mean_squared_error',
                                  cv=kfold)

    rmse_scores = [(-score) ** 0.5 for score in scores_rmse]
    rmse = np.mean(rmse_scores)

    mape_scorer = make_scorer(mape, greater_is_better=False)
    scores_mape = cross_val_score(xg_reg, feature_attributes, class_attribute, scoring=mape_scorer, cv=kfold)
    scores_mape = scores_mape * -1
    mape_s = np.mean(scores_mape)

    scores_r2 = cross_val_score(xg_reg, feature_attributes, class_attribute, scoring='r2', cv=kfold)
    r2 = np.mean(scores_r2)


    # print(rmse)
    # print(mape_s)
    # print(r2)
    for i in range(5):
        print("fold-" + str(i+1) + " Mean Absolute Percentage Error: " + str(scores_mape[i]) + "%")
        print("fold-" + str(i+1) + " Root Mean Square Error: " + str(rmse_scores[i]))
        print("fold-" + str(i+1) + " R-squared: " + str(scores_r2[i]))

    print(f"Average Mean Absolute Percentage Error: {mape_s} %")
    print(f"Average Root Mean Square Error: {rmse}")
    print(f"Average R-squared: {r2}")
    print(f"Training Time: {end-start} second")
    print()

    # 適配模型
    xg_reg.fit(feature_attributes, class_attribute)

    # 取得特徵的重要性
    feature_importances = xg_reg.feature_importances_

    # 將特徵的重要性與特徵名稱配對起來
    feature_importance_dict = dict(zip(feature_attributes.columns, feature_importances))

    # 依照重要性降序排列
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # 輸出排好序的特徵重要性
    for feature, importance in sorted_feature_importance:
        column = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat',
                  'medv']
        # print(f"{feature}: {importance}")
        for i in range(13):
            if feature == i:
                name = column[i]
        print(f"{name}: {importance}")

    print()
    # 刪除特徵值
    feature_name = feature_attributes.columns[1]
    new_feature_attributes = feature_attributes.drop(columns=feature_name)

    # print(new_feature_attributes)

    start = time.time()

    new_data_dmatrix = xgb.DMatrix(data=new_feature_attributes, label=class_attribute)

    new_xg_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    end = time.time()

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores_rmse = cross_val_score(new_xg_reg, new_feature_attributes, class_attribute, scoring='neg_mean_squared_error',
                                  cv=kfold)

    rmse_scores = [(-score) ** 0.5 for score in scores_rmse]
    rmse = np.mean(rmse_scores)

    mape_scorer = make_scorer(mape, greater_is_better=False)
    scores_mape = cross_val_score(new_xg_reg, new_feature_attributes, class_attribute, scoring=mape_scorer, cv=kfold)
    scores_mape = scores_mape * -1
    mape_s = np.mean(scores_mape)

    scores_r2 = cross_val_score(new_xg_reg, new_feature_attributes, class_attribute, scoring='r2', cv=kfold)
    r2 = np.mean(scores_r2)

    for i in range(5):
        print("fold-" + str(i+1) + " Mean Absolute Percentage Error: " + str(scores_mape[i]) + " %")
        print("fold-" + str(i+1) + " Root Mean Square Error: " + str(rmse_scores[i]))
        print("fold-" + str(i+1) + " R-squared: " + str(scores_r2[i]))
    
    print()
    print(f"New Average Mean Absolute Percentage Error: {mape_s} %")
    print(f"New Average Root Mean Square Error: {rmse}")
    print(f"New Average R-squared: {r2}")
    print(f"Training Time: {end - start} second")
    print()




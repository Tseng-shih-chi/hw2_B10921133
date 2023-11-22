import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import time


def read_data(file_path):
    # 使用 pandas 讀取資料
    data = pd.read_csv(file_path, header=None)  # 假設資料以空格分隔
    data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', \
                    'income']
    data.replace(' ?', pd.NA, inplace=True)

    return data


def handle_missing_values(data):
    # 刪除包含缺失值的資料列
    data = data.dropna()
    return data


def split_attributes(data):
    # 將資料分成 class attribute 和 feature attribute
    class_attribute_name = data.columns[12]
    class_attribute = data[class_attribute_name]
    feature_attributes = data.drop(columns=class_attribute_name)

    return class_attribute_name, class_attribute, feature_attributes


def read_test_data(file_path):
    test_data = pd.read_csv(file_path)
    test_data = test_data.reset_index()
    test_data.replace(' ?', pd.NA, inplace=True)

    return test_data


def transfer(data):
    # encoder
    label_encoder = LabelEncoder()
    data['workclass'] = label_encoder.fit_transform(data['workclass'])
    data['education'] = label_encoder.fit_transform(data['education'])
    data['marital-status'] = label_encoder.fit_transform(data['marital-status'])
    data['occupation'] = label_encoder.fit_transform(data['occupation'])
    data['relationship'] = label_encoder.fit_transform(data['relationship'])
    data['race'] = label_encoder.fit_transform(data['race'])
    data['sex'] = label_encoder.fit_transform(data['sex'])
    data['native-country'] = label_encoder.fit_transform(data['native-country'])
    data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

    # 正規化
    # 創建MinMaxScaler對象
    regular = MinMaxScaler()
    # 將數據進行最小-最大正規化
    data = pd.DataFrame(regular.fit_transform(data))

    return data


if __name__ == "__main__":
    # 訓練集資料前處理------------------------------------------------------------------------------------------------- #
    train_file_path = './adult.data'
    train_data = read_data(train_file_path)

    # 處理缺失值
    train_data = handle_missing_values(train_data)

    # 分離類別屬性和特徵屬性
    train_class_attribute_name, train_class_attribute, train_feature_attributes = split_attributes(train_data)

    train_feature_attributes = transfer(train_feature_attributes)

    # print(train_data)
    # print(train_data.iloc[14,13])

    # print("Class Attribute Name:", train_class_attribute_name)
    # print("Class Attribute:\n", train_class_attribute)
    # print("Feature Attributes:")
    # print(train_feature_attributes)

    # 測試集資料前處理------------------------------------------------------------------------------------------------ #
    test_file_path = './adult.test'
    test_data = read_test_data(test_file_path)

    test_data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country', 'income']

    # 處理缺失值
    test_data = handle_missing_values(test_data)

    # print(test_data)
    # print(test_data.iloc[4, 1])
    # 分離類別屬性和特徵屬性
    test_class_attribute_name, test_class_attribute, test_feature_attributes = split_attributes(test_data)

    test_feature_attributes = transfer(test_feature_attributes)

    # print("Class Attribute Name:", test_class_attribute_name)
    # print("Class Attribute:\n", test_class_attribute)
    # print("Feature Attributes:")
    # print(test_feature_attributes)

    # KNN方法------------------------------------------------------------------------------------------------------- #
    start = time.time()

    knn = KNeighborsRegressor(n_neighbors=5)

    knn.fit(train_feature_attributes, train_class_attribute)

    predicted = knn.predict(test_feature_attributes)

    # print(train_data.index)
    # print(test_data.index)
    #
    # print(test_feature_attributes)
    # print(test_class_attribute)
    # print(train_feature_attributes)
    # print(train_class_attribute)

    end = time.time()

    mae = mean_absolute_error(test_class_attribute, predicted)
    mape = (mae / np.mean(test_class_attribute)) * 100

    mse = mean_squared_error(test_class_attribute, predicted)
    # 計算均方根誤差（RMSE）
    rmse = np.sqrt(mse)

    r2 = r2_score(test_class_attribute, predicted)

    print(f"KNN Mean Absolute Percentage Error: {mape} %")
    print(f"KNN Root Mean Square Error: {rmse}")
    print(f"KNN R-squared: {r2}")
    print(f"KNN Predictions Time: {end - start} second")
    print()

    # SVR方法------------------------------------------------------------------------------------------------------- #
    start = time.time()
    svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1)

    svr_model.fit(train_feature_attributes, train_class_attribute)

    y_pred = svr_model.predict(test_feature_attributes)

    end = time.time()

    mae = mean_absolute_error(test_class_attribute, y_pred)
    # 計算 MAPE
    mape = (mae / np.mean(test_class_attribute)) * 100

    mse = mean_squared_error(test_class_attribute, y_pred)
    # 計算均方根誤差（RMSE）
    rmse = np.sqrt(mse)

    r2 = r2_score(test_class_attribute, y_pred)

    print(f"SVR Mean Absolute Percentage Error: {mape} %")
    print(f"SVR Root Mean Square Error: {rmse}")
    print(f"SVR R-squared: {r2}")
    print(f"SVR Predictions Time: {end - start} second")
    print()

    # RandomForest方法----------------------------------------------------------------------------------------------- #
    start = time.time()
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_regressor.fit(train_feature_attributes, train_class_attribute)

    y_pred = rf_regressor.predict(test_feature_attributes)

    end = time.time()

    mae = mean_absolute_error(test_class_attribute, y_pred)
    # 計算 MAPE
    mape = (mae / np.mean(test_class_attribute)) * 100

    mse = mean_squared_error(test_class_attribute, y_pred)
    # 計算均方根誤差（RMSE）
    rmse = np.sqrt(mse)

    r2 = r2_score(test_class_attribute, y_pred)

    print(f"RandomForest Mean Absolute Percentage Error: {mape} %")
    print(f"RandomForest Root Mean Square Error: {rmse}")
    print(f"RandomForest R-squared: {r2}")
    print(f"RandomForest Predictions Time: {end - start} second")
    print()

    # XGBoost方法------------------------------------------------------------------------------------------------- #
    start = time.time()

    # 將數據轉換為 XGBoost 的 DMatrix 格式
    dtrain = xgb.DMatrix(train_feature_attributes, label=train_class_attribute)
    dtest = xgb.DMatrix(test_feature_attributes, label=test_class_attribute)

    # 定義模型參數，可以根據需要進行調整
    params = {
        'objective': 'reg:squarederror',  # 回歸問題
        'eval_metric': 'rmse',  # 評估指標為均方根誤差
        'eta': 0.1,  # 學習率
        'max_depth': 3,  # 決策樹的最大深度
        'subsample': 0.8,  # 用於訓練每棵樹的樣本比例
    }

    # 訓練 XGBoost 模型
    num_round = 100  # 迭代次數
    bst = xgb.train(params, dtrain, num_round)

    # 在測試集上進行預測
    y_pred = bst.predict(dtest)

    end = time.time()

    mae = mean_absolute_error(test_class_attribute, y_pred)
    # 計算 MAPE
    mape = (mae / np.mean(test_class_attribute)) * 100

    mse = mean_squared_error(test_class_attribute, y_pred)
    # 計算均方根誤差（RMSE）
    rmse = np.sqrt(mse)

    r2 = r2_score(test_class_attribute, y_pred)

    print(f"XGBoost Mean Absolute Percentage Error: {mape} %")
    print(f"XGBoost Root Mean Square Error: {rmse}")
    print(f"XGBoost R-squared: {r2}")
    print(f"XGBoost Predictions Time: {end - start} second")

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# # Đọc dữ liệu và quan sát nhanh dữ liệu California Housing Prices đã được chia thành 3 tập Train, Dev, và Test
# * Lệnh ``pd.read_csv()`` tham khảo tại trang 47 trong sách Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow
# * Lệnh ``df.head()`` tham khảo tại trang 47 trong sách Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow
# * Lệnh ``df.info()`` tham khảo tại trang 47 trong sách Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

# %%
df_train = pd.read_csv('housing_train.csv') # Đọc dữ liệu tập Train (Tập dữ liệu việc huấn luyện mô hình)
df_dev = pd.read_csv('housing_dev.csv')     # Đọc dữ liệu tập Dev   (Tập dữ liệu dành cho việc phát triển/tinh chỉnh mô hình)
df_test = pd.read_csv('housing_test.csv')   # Đọc dữ liệu tập Test  (Tập dữ liệu dành cho việc kiểm tra mô hình)


# %%
df_train.info() # Kiểm tra thông tin ban đầu dữ liệu (số dòng, số cột, kiểu dữ liệu của các cột)


# %%
df_train.head() # Quan sát nhanh 5 dòng dữ liệu đầu tiên


# %%
df_dev.info() # Kiểm tra thông tin ban đầu dữ liệu (số dòng, số cột, kiểu dữ liệu của các cột)


# %%
df_dev.head() # Quan sát nhanh 5 dòng dữ liệu đầu tiên


# %%
df_test.info() # Kiểm tra thông tin ban đầu dữ liệu (số dòng, số cột, kiểu dữ liệu của các cột)


# %%
df_test.head() # Quan sát nhanh 5 dòng dữ liệu đầu tiên

# %% [markdown]
# # Tách thuộc tính "median_house_value" để làm thuộc tính cần dự đoán  cho bài toán hồi quy tuyến tính (Y), các thuộc tính còn lại là dữ kiện (X).
# * Lệnh ``df.drop()`` tham khảo tại trang sách số 63 sách tham khảo Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

# %%
train_X = df_train.drop("median_house_value", axis=1)
train_Y = df_train["median_house_value"].copy()
del df_train

dev_X = df_dev.drop("median_house_value", axis=1)
dev_Y = df_dev["median_house_value"].copy()
del df_dev

test_X = df_test.drop("median_house_value", axis=1)
test_Y = df_test["median_house_value"].copy()
del df_test

# %% [markdown]
# # Kiểm tra các thuộc tính bị khuyết giá trị trong tập Train
# * Chúng ta có thể cả các lệnh ``df.isnull()`` hoặc ``df.isna()`` để chỉ giá trị bị khuyết (``True``) / không bị khuyết (``False``)
# * Chúng ta dùng tiếp lệnh ``arr.any(axis=1)`` để trả về các dòng chỉ chứa một cột có giá trị ``True``/``False`` nếu các cột chỉ cần tồn tại một thuộc tính ``True``
# * Chúng ta tính tổng các dòng kết quả trên bằng lệnh ``arr.sum(axis=0)`` sẽ biết được số trường hợp bị khuyết của mỗi thuộc tính

# %%
train_X.isnull()


# %%
train_X.isnull().any(axis=1)
# train_X.isna().any(axis=1)


# %%
train_X[train_X.isnull().any(axis=1)]
# train_X[train_X.isna().any(axis=1)]


# %%
train_X.isnull().sum(axis = 0)
# train_X.isna().sum(axis = 0)

# %% [markdown]
# # Xử lý các cột bị khuyết dữ liệu trên tập Train
# 1. Phương án 1: Xóa toàn bộ một dòng có cột dữ liệu bị khuyết
# 2. Phương án 2: Thay thế giá trị số bằng cách dùng trung bình cộng (``mean``) của toàn cột dữ liệu đó
#     * Cách xử lý 1: Dùng lớp ``SimpleImputer`` từ thư viện ``sklearn.impute``
#     * Cách xử lý 2: Cách hai thực hiện một số thao tác trên cột
# 
# Các bạn có thể tham khảo tại trang số 63 sách Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

# %%
# Cách 1:
imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_mean.fit(train_X["total_bedrooms"].values.reshape(-1, 1))
tmp_total_bedrooms = imputer_mean.transform(train_X['total_bedrooms'].values.reshape(-1, 1))


# %%
# Cách 2:
idx_null = train_X["total_bedrooms"].isnull() # kiểm tra giá trị khuyết
mean_total_bedrooms = train_X["total_bedrooms"][train_X["total_bedrooms"].isna() == False].mean() # tính trung bình cộng các giá trị không bị khuyết
train_X["total_bedrooms"].fillna(mean_total_bedrooms, inplace=True) # thay thế các dòng không bị khuyết bởi giá trị trung bình
print(train_X["total_bedrooms"][idx_null == True]) # in ra màn hình giá trị các dòng bị khuyết ban đầu để kiểm tra


# %%
print((train_X["total_bedrooms"] == tmp_total_bedrooms.squeeze()).all()) # Kiểm tra kết quả Cách 1 và Cách 2 có giống nhau hay không?

print((train_X["total_bedrooms"].isna() == False).all()) # Kiểm tra kết quả sau khi áp dụng Cách 2 còn giá trị nào bị khuyết hay không?

# %% [markdown]
# # Yêu cầu 1: Xử lý các cột bị khuyết dữ liệu trên tập Dev/Test

# %% [markdown]
# ## Xử lí dữ liệu bị khuyết trên tập Dev
# ### 1. Kiểm tra các thuộc tính bị khuyết giá trị
dev_X.isnull()

# %% [markdown]
# ### 2. Tìm các giá trị bị khuyết
dev_X_Null_Index = dev_X.isnull().any(axis=1)
dev_X[dev_X_Null_Index]
# %%
dev_X.isnull().sum(axis = 0) # Tính tổng các giá trị bị khuyết

# %% [markdown]
# ### 3. Xử lí các cột bị khuyết dữ liệu
# * Lọc tất cả các phần tử bị khuyết và thay thế các phần tử đó bằng giá trị trung bình của cột đó
dev_X_imputer_mean = SimpleImputer() # Default SimpleImputer(missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True, add_indicator=False)
dev_X_imputer_mean.fit(dev_X["total_bedrooms"].values.reshape(-1,1))
dev_X["total_bedrooms"] = dev_X_imputer_mean.transform(dev_X["total_bedrooms"].values.reshape(-1,1))
# %% [markdown]
# ## Xử lí dữ liệu bị khuyết trên tập Test
test_X_Null_Index = test_X.isnull().any(axis=1)
test_X[test_X_Null_Index]

# %%
test_X.isnull().sum(axis = 0)

#%%
test_X_imputer_mean = SimpleImputer() # Default SimpleImputer(missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True, add_indicator=False)
test_X_imputer_mean.fit(test_X["total_bedrooms"].values.reshape(-1,1))
test_X["total_bedrooms"] = test_X_imputer_mean.transform(test_X["total_bedrooms"].values.reshape(-1,1))
# %% [markdown]
# # Yêu cầu 2: Thực hiện các thí nghiệm với việc thêm các thuộc tính tích lũy dẫn
# Chúng ta có danh sách các thuộc tính: ``longitude``, ``latitude``, ``housing_median_age``, ``total_rooms``, ``total_bedrooms``, ``population``, ``households``, ``median_income``, ``ocean_proximity``
# 
# 1. Thí nghiệm 1: ``longitude``
# 2. Thí nghiệm 2: ``longitude``, ``latitude``
# 3. Thí nghiêm 3: ``longitude``, ``latitude``, ``housing_median_age``
# 4. Thí nghiệm 4: v.v...
# 
# Lưu ý: Trong quá trình làm, các bạn sẽ tham khảo cách xử lý thuộc tính ``ocean_proximity`` bằng lớp ``LabelEncoder`` từ thư viện ``sklearn.preprocessing``

# %%
from math import sqrt
features_list_1 = ["longitude"]
model_1 = sm.OLS(train_Y, train_X[features_list_1]).fit()
rmse_train_1 = sqrt(mean_squared_error(train_Y, model_1.predict(train_X[features_list_1])))
# If True returns MSE value, if False returns RMSE value.
rmse_dev_1 = sqrt(mean_squared_error(dev_Y, model_1.predict(dev_X[features_list_1])))
rmse_test_1 = sqrt(mean_squared_error(test_Y, model_1.predict(test_X[features_list_1])))


# %%
features_list_2 = ["longitude", "latitude"]
model_2 = sm.OLS(train_Y, train_X[features_list_2]).fit()
rmse_train_2 = sqrt(mean_squared_error(train_Y, model_2.predict(train_X[features_list_2])))
rmse_dev_2 = sqrt(mean_squared_error(dev_Y, model_2.predict(dev_X[features_list_2])))
rmse_test_2 = sqrt(mean_squared_error(test_Y, model_2.predict(test_X[features_list_2])))


# %%
features_list_3 = ["longitude", "latitude", "housing_median_age"]
model_3 = sm.OLS(train_Y, train_X[features_list_3]).fit()
rmse_train_3 = sqrt(mean_squared_error(train_Y, model_3.predict(train_X[features_list_3])))
rmse_dev_3 = sqrt(mean_squared_error(dev_Y, model_3.predict(dev_X[features_list_3])))
rmse_test_3 = sqrt(mean_squared_error(test_Y, model_3.predict(test_X[features_list_3])))

# %%
features_list_4 = ["longitude", "latitude", "housing_median_age", "total_rooms"]
model_4 = sm.OLS(train_Y, train_X[features_list_4]).fit()
rmse_train_4 = sqrt(mean_squared_error(train_Y, model_4.predict(train_X[features_list_4])))
rmse_dev_4 = sqrt(mean_squared_error(dev_Y, model_4.predict(dev_X[features_list_4])))
rmse_test_4 = sqrt(mean_squared_error(test_Y, model_4.predict(test_X[features_list_4])))

# %%
features_list_5 = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms"]
model_5 = sm.OLS(train_Y, train_X[features_list_5]).fit()
rmse_train_5 = sqrt(mean_squared_error(train_Y, model_5.predict(train_X[features_list_5])))
rmse_dev_5 = sqrt(mean_squared_error(dev_Y, model_5.predict(dev_X[features_list_5])))
rmse_test_5 = sqrt(mean_squared_error(test_Y, model_5.predict(test_X[features_list_5])))


# %% [markdown]
# # Yêu cầu 3: Trình bày kết quả mô hình vào một bảng bằng thư viện Pandas

# %%
df_result = pd.DataFrame(data = {'RMSE_Train': [rmse_train_1, rmse_train_2, rmse_train_3, rmse_train_4, rmse_train_5],
                                 'RMSE_Dev': [rmse_dev_1, rmse_dev_2, rmse_dev_3, rmse_dev_4, rmse_dev_5],
                                 'RMSE_Test': [rmse_test_1, rmse_test_2, rmse_test_3, rmse_test_4, rmse_test_5]},
                         index = ['longitude',
                                  'longitude + latitude',
                                  'longitude + latitude + housing_median_age',
                                  'longitude + latitude + housing_median_age + total_rooms',
                                  'longitude + latitude + housing_median_age + total_rooms + total_bedrooms'
                                  ])

display(df_result.round(3))

# %% [markdown]
# # Yêu cầu 4: Nhận xét các kết quả trên
# Các bạn vui lòng nhận xét tại đây nha, ví dụ như giá trị hiệu suất ``RMSE`` thay đổi như thế nào?
# Các bạn nên dùng ``Markdown`` như thế này để nhận xét.
# %% [markdown]
# # Yêu cầu 5: Thực hiện lại các Yêu cầu 2, 3, 4 khi dùng mô hình ``Support Vector Machine regressor`` (``sklearn.svm.SVR``) để giải quyết bài toán hồi quy tuyến tính 

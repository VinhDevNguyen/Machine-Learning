# %% [markdown]
# # Bài tập thực hành tuần 2
# ## Bài tập 1:
# * Thực hành trên Python tính phương trình hồi quy đơn (trang 114 - 117 sách Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, 2nd Edition).

# ### 1.Tạo một cái data-set như hình ![Data Set](data_set.jpg)
# * Import numpy module
import numpy as np

# %% [markdown]
# * Theo mối tương quan với công thức Linear Regression là: y = $\theta_0$ + $\theta_1$$x1$ + $\epsilon$ 
# thì ta có thể tạo một biểu đồ với Intercept($\theta_0$) = 4 và Coefficients($\theta_1$) = 3
X = 2 * np.random.rand(100,1) 
y = 4 + 3 * X +np.random.randn(100,1)
# %% [markdown]
# ### 2. Plot đồ thị
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.show()

# %% [markdown]
# ### 3. Tính $\hat{\theta}$ dựa trên công thức Normal Equation: $\hat{\theta}$ = ($X^T$ X)$^-$$^1$ $X^T$ y

# %%
X_b = np.concatenate((np.ones((100,1)),X), axis=1) # add x0 = 1 to each instance
# X_b = np.c_[np.ones((100, 1)), X]

# %% [markdown]
# ### 4. Sử dụng sklearn package để tính $\hat{\theta}$
# Đầu tiên cần phải import sklearn
from sklearn import linear_model

# %% [markdown]
# Sau đó chúng ta model data
Regress = linear_model.LinearRegression()
Regress.fit(X, y)
print('Intercept (Theta0) = ', Regress.intercept_[0])
print('Coeficients (Theta1) = ', Regress.coef_[0][0])
# %%
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta0 = theta_best[0][0]
theta1 = theta_best[1][0]
print('Intercept (Theta0) = ', theta0)
print('Coeficients (Theta1) = ', theta1)

# %% [markdown]
# ### 5. Sử dụng $\hat{\theta}$ để dự đoán kết quả mới
# * Code trong sách
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
y_predict
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# %% [markdown]
# * Code tự viết: Áp dụng công thức y = $\theta_0$ + $\theta_1$$x1$ + $\epsilon$ và plot đồ thị.
plt.plot(X, theta0 + theta1*X, color='red')
# plt.plot(X, y, "b.")
plt.scatter(X, y)
plt.axis([0, 2, 0, 15])
plt.show()

# %% [markdown]
# ## Bài tập 2
# * Sử dụng chương trình Python trang 114 - 117 sách Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, 2nd Edition tính lại ví dụ mô phỏng ở slides 10 thuộc Tuần 2.

# %% [markdown]
# ### 1. Import numpy module
import numpy as np

# %% [markdown]
# ### 2. Create dataset: ($x_i$,$y_i$) = (147,49), (150,53), (153,51), (160,54)
X = np.array([147, 150, 153, 160])
y = np.array([49, 53, 51,54])

# %% [markdown]
# ### 3. Tính phương trình hồi quy:
# * Chuyển $x_i$ về dạng \begin{bmatrix}1 & 147 \\1 & 150 \\1 & 153 \\1 & 160 \end{bmatrix}
X_b = np.c_[np.ones((4, 1)), X]
X_b

# %% [markdown]
# * Tính $\hat\theta$
theta_hat = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_hat

# %% [markdown]
# ### 4. Summary
# Vậy phương trình hồi quy cần tìm là: $y$ = 5.02 + 0.31$x$

# %% [markdown]
# ## Bài tập 3
# * Thực hành với dữ liệu cho trong Sheet1 file Excel `demo-data.xls`.
# * Vẽ biểu đồ phân tán (biểu đồ Scatter) và nhận định về quan hệ giữa  $X$  và  $Y$ .
# * Thay đổi một giá trị của $Y$ sao cho thật khác biệt. Chạy chương trình và quan sát.
# %% [markdown]
# ### 1. Import some modules
import matplotlib.pyplot as plt # To plot data
import numpy as np
import pandas as pd # Import data from excel file to dataframe
from sklearn import linear_model

# %% [markdown]
# ### 2. Import data from Excel file to dataframe

# %%
df = pd.read_excel (r'demo_data.xls', 0) # Import data from Sheet1 to dataframe
df.head()

# ### 3. Data exploration
# * Summarize the data
df.describe()

# %% [markdown]
# * Select features
Feature = df[['X','y']]

# %% [markdown]
# ### 4. Plot data
plt.scatter(Feature.X, Feature.y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


# %% [markdown]
# ### 5. Correlation
Feature.X.corr(Feature.y, method = "pearson")

# %% [markdown]
# ### 6. Change $y$ element and run program again
RanIndex = np.random.randint(0,Feature.y.shape[0])
Feature.y[RanIndex] = Feature.y[RanIndex] + 999

plt.scatter(Feature.X, Feature.y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

Feature.X.corr(Feature.y, method = "pearson")

# %% [markdown]
# ### 7. Summary
# Sau khi thay đổi một giá trị $y$ bất kì sao cho thật khác 
# biệt thì chúng ta đã tạo một điểm Outliner trên đồ thị 
# và mối tương quan giữa $X$ và $y$ cũng bị thay đổi rõ rệt

# %% [markdown]
# ## Bài tập 4
# * Phân tích hồi quy đa biến, cho dữ liệu về giá cổ phiếu demo-data-mul.xls.
# * Vẽ biểu đồ Scatter giữa giá cổ phiếu và các thuộc tính Interest Rate và Unemployment Rate
# * Tìm phương trình hồi quy của giá cổ phiếu theo hai thuộc tính (biến) Interest Rate và Unemployment Rate.

# %% [markdown]
# ### 1. Import some modules
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# %% [markdown]
# ### 2. Import data from Excel file to dataframe
df = pd.read_excel('demo_data_mul.xls')
df.head()

# %% [markdown]
# ### 3. Correlation
df.corr(method='pearson')
# %% [markdown]
# * Visualize using heat-map 

# %% [markdown]
# ### 3. Data Exploration
Feature = df[[
    'Interest_Rate',
    'Unemployment_Rate',
    'Stock_Index_Price'
]]

# %% [markdown]
# ### 4. Vẽ biểu đồ scatter giữa `Interest_Rate` và `Stock_Index_Price`
plt.scatter(Feature.Interest_Rate, Feature.Stock_Index_Price)
plt.xlabel("Interest_Rate")
plt.ylabel("Stock_Index_Price")
plt.show()

# %% [markdown]
# ### 5. Vẽ biểu đồ scatter giữa `Unemployment Rate` và `Stock_Index_Price`
plt.scatter(Feature.Interest_Rate, Feature.Stock_Index_Price)
plt.xlabel("Interest_Rate")
plt.ylabel("Stock_Index_Price")
plt.show()

# %% [markdown]
# ### 6. Tính Intercept và Coeficients để tìm ra phương trình hồi quy
Regress = LinearRegression()
x_train = df[['Interest_Rate', 'Unemployment_Rate']]
Regress.fit(x_train, Feature.Stock_Index_Price)
print('Intercept: ', Regress.intercept_)
print('Coeficients: ', Regress.coef_)

# %% [markdown]
# ### 7. Summary
# * Ta rút ra được phương trình hồi quy: $y_hat$ = 1798.4039 + 345.5400 * `Interest_Rate` - 250.1465 * `Unemployment_Rate`

# %% [markdown]
# ## Bài tập 5:
# * Thực hành với dữ liệu cho trong Sheet2 file Excel demo-data.xls.
# * Thực hiện các yêu cầu tương tự Bài tập 4.
# * Bỏ bớt biến và quan sát sự thay đổi các thông số trong bảng phân tích hồi quy.
# * Xác định phương trình hồi quy có chứa  $X^{2}_2$  và nhận xét các thông số (Tạo thêm 1 cột $X^{2}_2$).
# ### 1. Import some modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# %% [markdown]
# ### 2. Read file and import to dataframe
df = pd.read_excel('demo_data.xls', 1)
df.head()

# %%
# ###  3. Plot data
plt.scatter(df[['X1']],df[['y']])
plt.xlabel('X1')
plt.ylabel('y')


# %%
plt.scatter(df[['X2']],df[['y']])
plt.xlabel('X2')
plt.ylabel('y')


# %% [markdown]
# ### 4. Linear Regression
x_train = df[['X1', 'X2']]
y_train = df[['y']]

Regress = LinearRegression()
Regress.fit(x_train,y_train)

# %% [markdown]
# ### 5. Calculate $\hat{\theta}$
print('Intercept: ', Regress.intercept_)
print('Coeficients: ', Regress.coef_)

# %% [markdown]
# ### 6. Ta bỏ bớt X1 để xem sự thay đổi của $\hat{\theta}$
Regress.fit(df[['X2']], y_train)
print('Intercept: ', Regress.intercept_)
print('Coeficients: ', Regress.coef_)

# %% [markdown]
# * Ta thấy rằng khi bỏ bớt X1 thì thì $\hat{\theta}$ thay đổi
# ### 7. Thêm vào cột $X^{2}_2$ để xem sự thay đổi của $\hat{\theta}$
X22 = np.full((10,1),8.0)
df['X22'] = X22
df.head()

# %% 
x_train = df[['X1', 'X2', 'X22']]
y_train = df[['y']]
Regress = LinearRegression()
Regress.fit(x_train,y_train)
print('Intercept: ', Regress.intercept_)
print('Coeficients: ', Regress.coef_)

# %% [markdown]
# ### 8. Summary
# * Khi thêm cột X22 vào thì ta thấy phương trình hồi quy không có sự thay đổi


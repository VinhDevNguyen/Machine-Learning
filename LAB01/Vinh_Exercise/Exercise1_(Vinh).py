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
# * Sử dụng chương trình Python trang 114 - 117 sách Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, 2nd Edition tính lại ví dụ mô phỏng ở slides 11 thuộc Tuần 2.


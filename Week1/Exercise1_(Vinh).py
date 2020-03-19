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

# %%
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best

# %% [markdown]
# ### 4. Sử dụng $\hat{\theta}$ để dự đoán kết quả mới


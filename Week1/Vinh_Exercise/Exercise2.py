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
# %%

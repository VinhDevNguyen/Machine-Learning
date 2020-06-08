# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Bài tập 1.
#  - <ins>Yêu cầu</ins>: Ý tưởng cơ bản của thuật toán ``Support Vector Machine`` (``SVM``) là gì? Ý tưởng của thuật toán biên mềm (``soft margin``) ``SVM``. Nêu ý nghĩa của siêu tham số ``C`` trong bài toán cực tiểu hàm mất mát.
# 
#   1. Ý tưởng cơ bản của SVM là đưa toàn bộ dataset vào không gian nhiều chiều (n chiều), từ đó tìm ra mặt phẳng thích hợp nhất (hyperplane) để phân chia
#   2. Support Vector Machine thuần (hard margin) thì gặp hai vấn đề chính đó là nó chỉ hoạt động trên dataset ``Linearly Separable`` và thứ 2 đó là nó khá nhạy cảm với biến nhiễu (sensitive to noise). Để tránh vấn đề này, chúng ta cần sử dụng một mô hình linh hoạt 
# hơn. Nhiệm vụ của nó là tìm được mặt phẳng vẫn phân loại tốt nhưng chấp nhận sai lệch ở một mức độ chấp nhận được.
#   3. Tham số `C` giúp ta điều chỉnh độ rộng của margin, `C` càng lớn thì sai lệch càng nhiều nhưng margin thì càng lớn, và ngược lại.
#  
# %% [markdown]
# ### Bài tập 2.
#  - <ins>Yêu cầu</ins>: Sử dụng mô hình ``SVM`` thuộc thư viện ``sklearn`` để xây dựng mô hình phân loại dựa trên tập dữ liệu huấn luyện ``X_train``, ``y_train``. Hãy nhận xét về tỉ lệ nhãn ``0`` và ``1`` trong bộ dữ liệu đã cho như đoạn code bên dưới. Hãy thử thay đổi giá trị của tham số ``C`` và nhận xét các độ đo ``Recall``, ``Precison``, ``F1-score``, và ``Accuracy`` của mô hình thu được trên tập dữ liệu kiểm tra ``X_test``, ``y_test``.
#  - Nguồn tham khảo dữ liệu ``thyroid_sick.csv``: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease

# %%
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('thyroid_sick.csv')
X = df[[column_name for column_name in df.columns if column_name != 'classes']]
y = df[['classes']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# %% [markdown]
# ### Bài tập 3.
#  - <ins>Yêu cầu</ins>: Ý tưởng của hàm ``kernel`` $K(\dots, \dots)$ là gì? Khi nào chúng ta áp dụng hàm ``kernel``? Chúng ta có cần biết biểu thức của hàm $\Phi(x)$ không?
# %% [markdown]
# ### Bài tập 4.
#  - <ins>Yêu cầu</ins>: Cho điểm dữ liệu trong không gian hai chiều $x = [x_1, x_2]^T$ và hàm biến đổi sang không gian năm chiều $\Phi(x) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1x_2, x_2^2]^T$. Hãy tính hàm ``kernel`` $K(a, b)$.
# %% [markdown]
# ### Bài tập 5.
#  - <ins>Yêu cầu</ins>: Giả sử bạn dùng bộ phân loại ``SVM`` với hàm ``kernel`` (radial basis function) ``RBF`` cho tập huấn luyện và thấy mô hình phân loại chưa tốt. Để cải thiện, bạn sẽ giảm hay tăng tham số $\gamma$ trong công thức hàm ``kernel``, tham số ``C`` trong hàm mất mát.
# %% [markdown]
# ### Bài tập 6. (Exercise 9 trang 174, Chapter 5: Support Vector Machines)
#  - <ins>Yêu cầu</ins>: Huấn luyện một bộ phân lớp ``SVM`` dựa trên bộ dữ liệu ``MNIST`` (dùng để phân loại hình ảnh các ký tự số có cùng kích thước). Bởi vì bộ phân loại ``SVM`` là bộ phân lớp nhị phân, chúng ta sẽ cần sử dụng chiến thuật ``one-versus-the-rest`` để phân loại tất cả ``10`` ký tự số (trong thực tế chúng ta chỉ dùng chiến thuật ``one-versus-one`` trong các trường hợp dữ liệu nhỏ). Bạn hãy báo cáo độ chính xác (``accuracy``) của mô hình đã huấn luyện trên tập test.

# %%
import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

# %% [markdown]
# ### Bài tập 7. (Exercise 10 trang 174, Chapter 5: Support Vector Machines)
#  - <ins>Yêu cầu</ins>: Hãy huấn luyện một mô hình hồi quy tuyến tính với dữ liệu giá nhà ``California housing dataset``.

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Bài tập 1.
#  - <ins>Yêu cầu</ins>: Ý tưởng cơ bản của thuật toán ``Support Vector Machine`` (``SVM``) là gì? Ý tưởng của thuật toán biên mềm (``soft margin``) ``SVM``. Nêu ý nghĩa của siêu tham số ``C`` trong bài toán cực tiểu hàm mất mát.
# 
#   1. Ý tưởng cơ bản của SVM là đưa toàn bộ dataset vào không gian nhiều chiều (n chiều), từ đó tìm ra mặt phẳng thích hợp nhất (hyperplane) để phân chia
#   2. Support Vector Machine thuần (hard margin) thì gặp hai vấn đề chính đó là nó chỉ hoạt động trên dataset ``Linearly Separable`` và thứ 2 đó là nó khá nhạy cảm với biến nhiễu (sensitive to noise). Để tránh vấn đề này, chúng ta cần sử dụng một mô hình linh hoạt 
# hơn. Nhiệm vụ của nó là tìm được mặt phẳng vẫn phân loại tốt nhưng chấp nhận sai lệch ở một mức độ chấp nhận được.
#   3. Tham số `C` là hằng số dương giúp cân đối độ lớn của margin và sự hy sinh của các điểm nằm trong vùng không an toàn. Khi $C = \infty $ hoặc rất lớn, Soft Margin SVM trở thành Hard Margin SVM.
# %% [markdown]
# ### Bài tập 2.
#  - <ins>Yêu cầu</ins>: Sử dụng mô hình ``SVM`` thuộc thư viện ``sklearn`` để xây dựng mô hình phân loại dựa trên tập dữ liệu huấn luyện ``X_train``, ``y_train``. Hãy nhận xét về tỉ lệ nhãn ``0`` và ``1`` trong bộ dữ liệu đã cho như đoạn code bên dưới. Hãy thử thay đổi giá trị của tham số ``C`` và nhận xét các độ đo ``Recall``, ``Precison``, ``F1-score``, và ``Accuracy`` của mô hình thu được trên tập dữ liệu kiểm tra ``X_test``, ``y_test``.
#  - Nguồn tham khảo dữ liệu ``thyroid_sick.csv``: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
import numpy as np
from sklearn import preprocessing

# %%
df = pd.read_csv('thyroid_sick.csv')
X = df[[column_name for column_name in df.columns if column_name != 'classes']]
y = df[['classes']]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# %%
df.pivot_table(index =['classes'], aggfunc='size')

# %% [markdown]
# * Nhận xét:

# %%
from sklearn import svm
C_parameter = 10.5
mean_acc = []
for n in np.arange(0.5, C_parameter, 0.5):
    
    #Train Model and Predict  
    clf = svm.SVC(C=n).fit(X_train,y_train)
    yhat=clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, yhat)

    mean_acc.append(float(accuracy_score(y_test, yhat)))
    print("Result with C = " + str(n))
    np.set_printoptions(precision=2)
    print (classification_report(y_test, yhat))

print( "The best accuracy was with", max(mean_acc), "with C=", n)

# %% [markdown]
# ### Bài tập 3.
#  - <ins>Yêu cầu</ins>: Ý tưởng của hàm ``kernel`` $K(\dots, \dots)$ là gì? Khi nào chúng ta áp dụng hàm ``kernel``? Chúng ta có cần biết biểu thức của hàm $\Phi(x)$ không?
#  1. Kernel SVM là việc đi tìm một hàm số biến đổi dữ liệu $x$ từ không gian feature ban đầu thành dữ liệu trong một không gian mới bằng hàm số $\Phi(\mathbf{x})$. Hàm số này cần thoả mãn mục đích đó là tronng không gian mới, dữ liệu giữa hai classes là phân biệt tuyến tính hoặc gần như phần biệt tuyến tính.
#  2. Chúng ta áp dụng hàm ``kernel`` khi dữ liệu không phân biệt tuyến tính, Với dữ liệu gần phân biệt tuyến tính, linear và poly kernels cho kết quả tốt hơn.
#  3.
# %% [markdown]
# ### Bài tập 4.
#  - <ins>Yêu cầu</ins>: Cho điểm dữ liệu trong không gian hai chiều $x = [x_1, x_2]^T$ và hàm biến đổi sang không gian năm chiều $\Phi(x) = [1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}x_1x_2, x_2^2]^T$. Hãy tính hàm ``kernel`` $K(a, b)$.
# 
# \begin{eqnarray}
# \Phi(\mathbf{x})^T\Phi(\mathbf{z}) &=& [1, \sqrt{2} x_1, \sqrt{2} x_2, x_1^2, \sqrt{2} x_1x_2, x_2^2] [1, \sqrt{2} z_1, \sqrt{2} z_2, z_1^2, \sqrt{2} z_1z_2, z_2^2]^T \\
# &=& 1 + 2x_1z_1 + 2x_2z_2 + x_1^2x_2^2 + 2x_1z_1x_2z_2 + x_2^2z_2^2 \\
# &=& (1 + x_1z_1 + x_2z_2)^2 = (1 + \mathbf{x}^T\mathbf{z})^2 = k(\mathbf{x}, \mathbf{z})
# \end{eqnarray}
# %% [markdown]
# ### Bài tập 5.
#  - <ins>Yêu cầu</ins>: Giả sử bạn dùng bộ phân loại ``SVM`` với hàm ``kernel`` (radial basis function) ``RBF`` cho tập huấn luyện và thấy mô hình phân loại chưa tốt. Để cải thiện, bạn sẽ giảm hay tăng tham số $\gamma$ trong công thức hàm ``kernel``, tham số ``C`` trong hàm mất mát.
# %% [markdown]
# ### Bài tập 6. (Exercise 9 trang 174, Chapter 5: Support Vector Machines)
#  - <ins>Yêu cầu</ins>: Huấn luyện một bộ phân lớp ``SVM`` dựa trên bộ dữ liệu ``MNIST`` (dùng để phân loại hình ảnh các ký tự số có cùng kích thước). Bởi vì bộ phân loại ``SVM`` là bộ phân lớp nhị phân, chúng ta sẽ cần sử dụng chiến thuật ``one-versus-the-rest`` để phân loại tất cả ``10`` ký tự số (trong thực tế chúng ta chỉ dùng chiến thuật ``one-versus-one`` trong các trường hợp dữ liệu nhỏ). Bạn hãy báo cáo độ chính xác (``accuracy``) của mô hình đã huấn luyện trên tập test.

# %%
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
mnist = fetch_openml('mnist_784', version=1, cache=True)

# %%
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]
lin_clf = LinearSVC(random_state=42)
lin_clf.fit(X_train, y_train)
y_pred = lin_clf.predict(X_train)
accuracy_score(y_train, y_pred)

# %%
y_test_predict =lin_clf.predict(X_test)
accuracy_score(y_test, y_test_predict)
# %%
Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = Scaler.fit_transform(X_test.astype(np.float32))
# %%
lin_clf = LinearSVC(random_state =42)
lin_clf.fit(X_train, y_train)
y_pred = lin_clf.predict(X_train)
accuracy_score(y_train, y_pred)
# %%
y_test_predict = lin_clf.predict(X_test_scaled)
accuracy_score(y_test, y_test_predict)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciproca, uniform
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.svm import SVC

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

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


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Bài Tập 1:
# * Linear Regression thì dùng để dự đoán giá trị liên tục (Như là dự đoán giá nhà, etc...), nhưng Linear Regression lại không phải cách tốt nhất để dự đoán tập hợp những điểm dữ liệu, để dự đoán được chúng ta cần phải dùng phương pháp nào đó giúp tìm tập hợp các điểm có khả năng cao nhất để phân loại, và phương pháp đó là dùng Logistic Regression
# * Linear Regression dùng đường thẳng để dự đoán kết quả, trong khi đó Logistic dùng đường cong để dự đoán (Ta còn có thể gọi nó là logistic curve)
# * Logistic Regression kết quả của nó trả ra các giá trị rời rạc còn Linear thì trả ra giá trị tuyến tính nên Linear Regression thì dành cho các bài toán hồi quy tuyến tính còn logistic thì dành cho các bài toán phân loại
# %% [markdown]
# # Bài Tập 2:
# 1.Import thư viện:

# %%
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# %% [markdown]
# ## 2.Read data from `csv` file

# %%
df_Cont = pd.read_csv("ML_Learning_Hobby.csv")
df_Cont.head()


# %%
X = np.asarray(df_Cont[['X']])
X[0:5]

# %% [markdown]
# ## 3. Chuyển Feature `y` từ `Yes` thành `1` và `No` thành `0`

# %%
LE_y = preprocessing.LabelEncoder()
LE_y.fit(['No','Yes'])
df_Cont['Y'] = LE_y.transform(df_Cont['Y'].values.reshape(-1,1))
df_Cont['Y'].head()


# %%
y = np.asarray(df_Cont[['Y']])
y[0:5]

# %% [markdown]
# ## 4. Modeling (Logistic Regression with Scikit-learn)

# %%
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X, y)
LR


# %%
X_new = 62
yhat = LR.predict([[X_new]])
yhat

# %% [markdown]
#  Với `X_new` = `62` kg thì sinh viên này thích học machine learning
# %% [markdown]
#  # Bài tập 3
#  ## Xây dựng mô hình hồi quy logistic dựa vào nhóm các thuộc tích liên tục FeatureName_Cont.
# %% [markdown]
# ### 1.Read data from `csv` file

# %%
df_Cont = pd.read_csv("Play_Tennis.csv")
df_Cont.head()

# %% [markdown]
# ### 2. Chuyển Feature `y` từ `Yes` thành `1` và `No` thành `0`

# %%
LE_y = preprocessing.LabelEncoder()
LE_y.fit(['No','Yes'])
df_Cont['Play_Tennis'] = LE_y.transform(df_Cont['Play_Tennis'].values.reshape(-1,1))
df_Cont['Play_Tennis'].head()


# %%
X = np.asarray(df_Cont[["Outlook_Cont", "Temp_Cont", "Humidity_Cont", "Wind_Cont"]])
X[0:5]


# %%
y = np.asarray(df_Cont[["Play_Tennis"]])
y[0:5]


# %%
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X, y)
LR


# %%
yhat = LR.predict(X)
yhat

# %% [markdown]
#  ## Xây dựng mô hình hồi quy logistic dựa vào nhóm các thuộc tích liên tục FeatureName_Cont và rời rạc FeatureName_Cat

# %%
df_All = df_Cont.copy()
df_All.head()

# %% [markdown]
# ### Biến tất cả các biến rời rạc thành biến giả

# %%
df_All.pivot_table(index =['Outlook_Cat'], aggfunc='size')


# %%
LE_Outlook_Cat = preprocessing.LabelEncoder()
LE_Outlook_Cat.fit(["Overcast", "Rain", "Sunny"])
df_All["Outlook_Cat"] = LE_Outlook_Cat.transform(df_All['Outlook_Cat'].values.reshape(-1,1))
df_All["Outlook_Cat"].head()


# %%
df_All.pivot_table(index =['Temp_Cat'], aggfunc='size')


# %%
LE_Temp_Cat = preprocessing.LabelEncoder()
LE_Temp_Cat.fit(["Cool", "Mild", "Hot"])
df_All["Temp_Cat"] = LE_Temp_Cat.transform(df_All['Temp_Cat'].values.reshape(-1,1))
df_All["Temp_Cat"].head()


# %%
df_All.pivot_table(index =['Humidity_Cat'], aggfunc='size')


# %%
LE_Humidity_Cat = preprocessing.LabelEncoder()
LE_Humidity_Cat.fit(["High", "Normal"])
df_All["Humidity_Cat"] = LE_Humidity_Cat.transform(df_All['Humidity_Cat'].values.reshape(-1,1))
df_All["Humidity_Cat"].head()


# %%
df_All.pivot_table(index =['Wind_Cat'], aggfunc='size')


# %%
LE_Wind_Cat = preprocessing.LabelEncoder()
LE_Wind_Cat.fit(["Strong", "Weak"])
df_All["Wind_Cat"] = LE_Wind_Cat.transform(df_All['Wind_Cat'].values.reshape(-1,1))
df_All["Wind_Cat"].head()


# %%
X_All = np.asarray(df_All[["Outlook_Cat", "Outlook_Cont", "Temp_Cat", "Temp_Cont", "Humidity_Cat", "Humidity_Cont", "Wind_Cat", "Wind_Cont"]])
X_All[0:5]


# %%
y_All = np.asarray(df_All[["Play_Tennis"]])
y_All[0:5]


# %%
from sklearn.linear_model import LogisticRegression
LR_All = LogisticRegression().fit(X_All, y_All)
LR_All


# %%
yhat_All = LR_All.predict(X_All)
yhat_All

# %% [markdown]
#  ## Đánh giá:
#  Classification Report của model chỉ có FeatureName_Cont

# %%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y, yhat))

# %% [markdown]
#  Classification Report của model có FeatureName_Cont và FeatureName_Cat

# %%
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_All, yhat_All))

# %% [markdown]
#  Ta thấy accuracy tăng khi ta kết hợp nhóm các thuộc tích liên tục FeatureName_Cont và rời rạc FeatureName_Cat

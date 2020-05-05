# %% [markdown]
# # Bài Tập 2:
# 1.Import thư viện:
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing
%matplotlib inline 
import matplotlib.pyplot as plt

# %% [markdown]
# 2.Read data from `csv` file
# %%
df_Cont = pd.read_csv("ML_Learning_Hobby.csv")
df_Cont.head()

# %%
X = np.asarray(df_Cont[['X']])
X[0:5]

# %% [markdown]
# 3. Chuyển Feature `y` từ `Yes` thành `1` và `No` thành `0`

# %%
LE_y = preprocessing.LabelEncoder()
LE_y.fit(['No','Yes'])
df_Cont['Y'] = LE_y.transform(df_Cont['Y'].values.reshape(-1,1))
df_Cont['Y'].head()

# %%
y = np.asarray(df_Cont[['Y']])
y[0:5]

# %%
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X, y)
LR

# %%
X_new = 62
yhat = LR.predict([[X_new]])
yhat

# %% [markdown]
# Với `X_new` = `62` kg thì sinh viên này thích học machine learning

# %% [markdown]
# # Bài tập 3
# ## Xây dựng mô hình hồi quy logistic dựa vào nhóm các thuộc tích liên tục FeatureName_Cont.

# %% [markdown]
# 1.Read data from `csv` file
# %%
df_Cont = pd.read_csv("Play_Tennis.csv")
df_Cont.head()


# %%
# %% [markdown]
# 2. Chuyển Feature `y` từ `Yes` thành `1` và `No` thành `0`

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

# # %%
# from sklearn.metrics import classification_report
# print(classification_report(y, yhat))

# # %%
# X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# y_proba = LR.predict_proba(X_new)
# plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
# plt.show()

# # %%

# %% [markdown]
# ## Xây dựng mô hình hồi quy logistic dựa vào nhóm các thuộc tích liên tục FeatureName_Cont và rời rạc FeatureName_Cat

# %%
df_All = df_Cont.copy()

# %%
df_All.pivot_table(index =['Outlook_Cat'], aggfunc='size')


# %%
LE_Outlook_Cat = preprocessing.LabelEncoder()
LE_Outlook_Cat.fit(["Overcast", "Rain", "Sunny"])
df_All["Outlook_Cat"] = LE_Outlook_Cat.transform(df_Cont['Outlook_Cat'].values.reshape(-1,1))
df_All["Outlook_Cat"].head()

# %%
df_All.pivot_table(index =['Temp_Cat'], aggfunc='size')

# %%
LE_Temp_Cat = preprocessing.LabelEncoder()
LE_Temp_Cat.fit(["Cool", "Mild", "Hot"])
df_All["Temp_Cat"] = LE_Temp_Cat.transform(df_Cont['Temp_Cat'].values.reshape(-1,1))
df_All["Temp_Cat"].head()


# %%
df_All.pivot_table(index =['Humidity_Cat'], aggfunc='size')

# %%
LE_Humidity_Cat = preprocessing.LabelEncoder()
LE_Humidity_Cat.fit(["High", "Normal"])
df_All["Humidity_Cat"] = LE_Humidity_Cat.transform(df_Cont['Humidity_Cat'].values.reshape(-1,1))
df_All["Humidity_Cat"].head()


# %%
df_All.pivot_table(index =['Wind_Cat'], aggfunc='size')

# %%
LE_Wind_Cat = preprocessing.LabelEncoder()
LE_Wind_Cat.fit(["Strong", "Weak"])
df_All["Wind_Cat"] = LE_Wind_Cat.transform(df_Cont['Wind_Cat'].values.reshape(-1,1))
df_All["Wind_Cat"].head()

# %%

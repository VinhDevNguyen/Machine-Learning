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
df = pd.read_csv("ML_Learning_Hobby.csv")
df.head()

# %%
X = np.asarray(df[['X']])
X[0:5]

# %% [markdown]
# 3. Chuyển Feature `y` từ `Yes` thành `1` và `No` thành `0`

# %%
LE_y = preprocessing.LabelEncoder()
LE_y.fit(['No','Yes'])
df['Y'] = LE_y.transform(df['Y'].values.reshape(-1,1))
df['Y'].head()

# %%
y = np.asarray(df[['Y']])
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
df = pd.read_csv("Play_Tennis.csv")
df.head()


# %%
# %% [markdown]
# 2. Chuyển Feature `y` từ `Yes` thành `1` và `No` thành `0`

# %%
LE_y = preprocessing.LabelEncoder()
LE_y.fit(['No','Yes'])
df['Play_Tennis'] = LE_y.transform(df['Play_Tennis'].values.reshape(-1,1))
df['Play_Tennis'].head()


# %%
X = np.asarray(df[["Outlook_Cont", "Temp_Cont", "Humidity_Cont", "Wind_Cont"]])
X[0:5]

# %%
y = np.asarray(df[["Play_Tennis"]])
y[0:5]

# %%
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X, y)
LR

# %%
yhat = LR.predict(X)
yhat


# %% [markdown]
# * Thực hành với dữ liệu cho trong Sheet2 file Excel demo-data.xls.
# * Thực hiện các yêu cầu tương tự Bài tập 4.
# * Bỏ bớt biến và quan sát sự thay đổi các thông số trong bảng phân tích hồi quy.
# * Xác định phương trình hồi quy có chứa  $X^{2}_2$  và nhận xét các thông số (Tạo thêm 1 cột $X^{2}_2$).
# ### 1. Import some modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf


# %% [markdown]
# ### 2. Read file and import to dataframe
df = pd.read_excel('demo_data.xls', 1)
df.head()

# %%
# ###  3. Plot data
plt.scatter(df[['X1']],df[['y']],color='blue')
plt.xlabel('X1')
plt.ylabel('y')


# %%
plt.scatter(df[['X2']],df[['y']], color = 'red')
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
# ### 6. Ta bỏ bớt X1 để xem sự thay đổi của nó
result = smf.ols('y ~ X2', df).fit()
result.summary()

# %% [markdown]
# ### 7. Ta bỏ bớt thử X2 để xem sự thay đổi của nó
result = smf.ols('y ~ X1', df).fit()
result.summary()
# * Ta thấy khi bỏ bớt X1 và X2 đi thì ta thấy hệ số $R^2$ bị giảm
# %% [markdown]
# * Ta thấy rằng khi bỏ bớt X1 thì thì $\hat{\theta}$ thay đổi
# ### 7. Thêm vào cột $X^{2}_2$ để xem sự thay đổi của $\hat{\theta}$
X22 = df['X2'].pow(2)
df['X22'] = X22
df.head()

X23 = df['X2'].pow(3)
df['X23'] = X23
df.head()
result23 = smf.ols('y ~ X1 + X2 + X22 + X23', data= df).fit()
result22 = smf.ols('y ~ X1 + X2 + X22', data = df).fit()
# %%
result22.summary()

# %%
result23.summary()

# %% [markdown]
# ### 8. Summary
# * Khi thêm cột X2_2 vào thì ta thấy hệ số $R^2$ đều tăng rất mạnh

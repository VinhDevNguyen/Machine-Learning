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

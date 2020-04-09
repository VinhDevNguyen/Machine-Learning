# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt 


# %%
data = pd.ExcelFile('demo_data_mul.xls')


# %%
df = pd.read_excel(data, 0, header = 0)
df


# %%
df.describe()


# %%
df.corr(method='pearson')


# %%
sns.heatmap(df.corr(method = 'pearson'), annot = True)
plt.show()


# %%
sns.pairplot(data = df, kind = 'scatter')
plt.show()


# %%
Y = df[['Year']]
M = df[['Month']]
IR = df[['Interest_Rate']]
UR = df[['Unemployment_Rate']]
SIP = df[['Stock_Index_Price']]
Y


# %%
M


# %%
IR


# %%
UR


# %%
SIP


# %%
plt.scatter(IR, SIP)
plt.show()


# %%
plt.scatter(UR, SIP)
plt.show()


# %%
lr = LinearRegression()


# %%
x_train = df[['Interest_Rate', 'Unemployment_Rate']]
x_train


# %%



# %%
lr.fit(x_train, SIP)
print(lr.intercept_)
print(lr.coef_)


# %%
plt.scatter(IR, SIP, color = 'blue')
plt.scatter(IR,lr.predict(x_train), color = 'red')


# %%
plt.scatter(UR, SIP, color = 'blue')
plt.scatter(UR,lr.predict(x_train), color = 'red')


# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x_grid = np.linspace(min(x_train['Interest_Rate']), max(x_train['Interest_Rate']), 20)
y_grid = np.linspace(min(x_train['Unemployment_Rate']), max(x_train['Unemployment_Rate']), 20)
x_grid, y_grid = np.meshgrid(x_grid, y_grid)
exog = pd.core.frame.DataFrame({'Interest_Rate' : x_grid.ravel(), 'Unemployment_Rate' : y_grid.ravel()})
out = lr.predict(exog)

ax.plot_surface(x_grid, y_grid, out.reshape(20, 20), rstride = 1, cstride = 1, alpha = 0.3)
ax.scatter(x_train['Interest_Rate'], x_train['Unemployment_Rate'], SIP, color = 'red', marker = 'o', alpha = 1)
ax.set_xlabel('Interest_Rate')
ax.set_ylabel('Unemployment_Rate')
ax.set_zlabel('Stock_Index_Price')

ax.view_init(20, 75)
plt.show()


# %%



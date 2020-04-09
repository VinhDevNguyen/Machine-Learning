# %% [markdown]
# ## Bài tập 3
# * Thực hành với dữ liệu cho trong Sheet1 file Excel `demo-data.xls`.
# * Vẽ biểu đồ phân tán (biểu đồ Scatter) và nhận định về quan hệ giữa  $X$  và  $Y$ .
# * Thay đổi một giá trị của $Y$ sao cho thật khác biệt. Chạy chương trình và quan sát.
# %% [markdown]
# ### 1. Import some modules
import matplotlib.pyplot as plt # To plot data
import numpy as np
import pandas as pd # Import data from excel file to dataframe
from sklearn import linear_model

# %% [markdown]
# ### 2. Import data from Excel file to dataframe

# %%
df = pd.read_excel (r'demo_data.xls', 0) # Import data from Sheet1 to dataframe
df.head()

# ### 3. Data exploration
# * Summarize the data
df.describe()

# %% [markdown]
# * Select features
Feature = df[['X','y']]

# %% [markdown]
# ### 4. Plot data
plt.scatter(Feature.X, Feature.y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


# %% [markdown]
# ### 5. Correlation
Feature.X.corr(Feature.y, method = "pearson")

# %% [markdown]
# ### 6. Change $y$ element and run program again
RanIndex = np.random.randint(0,Feature.y.shape[0])
Feature.y[RanIndex] = Feature.y[RanIndex] + 999

plt.scatter(Feature.X, Feature.y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

Feature.X.corr(Feature.y, method = "pearson")

# %% [markdown]
# ### 7. Summary
# Sau khi thay đổi một giá trị $y$ bất kì sao cho thật khác 
# biệt thì chúng ta đã tạo một điểm Outliner trên đồ thị 
# và mối tương quan giữa $X$ và $y$ cũng bị thay đổi rõ rệt

# %%

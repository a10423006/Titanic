#%%
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# %% # 檢查null
print(pd.isnull(train_data).any())

# %% # 處理 null # Age填上平均值
age_mean = train_data.Age.mean()
train_data.Age = train_data.Age.fillna(age_mean)

# Cabin 含有的 NAN 過多，該欄位直接 drop
print(len(np.where(pd.isnull(train_data.Cabin))[0]))
train_data = train_data.drop('Cabin', 1)

# Embarked 含有 NAN僅兩筆，將此兩筆 drop
print(len(np.where(pd.isnull(train_data.Embarked))[0]))
train_data = train_data.dropna()

#%% # 調整順序 # 處理類別變數 # 不納入 Name 和 Ticket
train_y = train_data['Survived'].astype('category')
train_x = train_data[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

#%% # Label encoding
labelencoder = LabelEncoder()
train_x['Sex'] = labelencoder.fit_transform(train_x['Sex'])
train_x['Embarked'] = labelencoder.fit_transform(train_x['Embarked'])

# %% # RFECV 透過 決策樹和 StratifiedKFold 5折交叉驗證尋找最佳特徵數
cols = train_x.columns[1:]
DTC = DecisionTreeClassifier()
rfecv = RFECV(estimator=DTC, cv=KFold(n_splits=5), scoring='accuracy').fit(train_x[cols], train_y)

list_rfecv = ['PassengerId']
for i, n in enumerate(cols):
    if rfecv.support_[i]:
        list_rfecv.append(n)

select_datas = pd.concat([train_x[list_rfecv], train_y], axis=1)
select_datas.to_csv('select_datas.csv', index=False)

# %%
# 特徵排名
list_rank_rfecv = []
for i, n in enumerate(cols):
    list_rank_rfecv.append((n, rfecv.ranking_[i]))
list_rank_rfecv = sorted(list_rank_rfecv, key = lambda x : x[1])

# 選擇特徵數x交叉驗證準確度
plt.figure(figsize=(15,10))
plt.xlabel("Number of features selected", fontsize = 25)
plt.ylabel("Cross validation score(Accuracy)", fontsize = 25)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
# 最高點
max_position = int(np.where(rfecv.grid_scores_ == rfecv.grid_scores_.max())[0][0])
# 曲線設定
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, '-o', lw=5, ms=20)
plt.plot([max_position + 1]*2, [rfecv.grid_scores_.min(), rfecv.grid_scores_.max()], '--', lw=5)
plt.plot(max_position + 1, rfecv.grid_scores_.max(), '-o', ms=20, mfc='darkorange', mec='darkorange')
# 標籤設定
plt.annotate(round(rfecv.grid_scores_.max(), 4), xy=(max_position, rfecv.grid_scores_.max()), xytext=(20, -10), textcoords='offset points', fontsize = 20, color='darkorange', weight='bold')
# 排名註解
plt.annotate('Features Rank\n'+str(list_rank_rfecv).replace('),', ')\n'), xy=(len(rfecv.grid_scores_)-1.7, rfecv.grid_scores_.min()), xytext=(0, 0), textcoords='offset points', bbox=dict(boxstyle='round'), fontsize = 25, color='w', weight='bold')
plt.savefig('.image/rfecv_cross_validation.png')
plt.show()

# %% # 生存比例查看
plt.bar(train_y.cat.categories, train_y.value_counts())
# 資料標籤
for a,b in zip(train_y.cat.categories, train_y.value_counts()):  
    plt.text(a, b+100, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
# y軸上限
plt.ylim(0, len(train_y))
# x軸label
plt.xticks(train_y.cat.categories,("Yes", "No"))
# x軸標題
plt.xlabel('the outcome for each passenger')
# y軸標題
plt.ylabel('Numbers')
plt.savefig('./image/survived_bar.png')
plt.show()
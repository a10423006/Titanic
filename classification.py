#%%
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import joblib

train = pd.read_csv('select_datas.csv', index_col='PassengerId')
print(Counter(train.Survived))

# %%
ros = RandomOverSampler(random_state=0)
resam_train_x, resam_train_y = ros.fit_sample(train[train.columns[:-1]], train.Survived)
sorted(Counter(resam_train_y).items())

train_x = pd.DataFrame(resam_train_x,columns=train.columns[:-1])
train_y = pd.DataFrame(resam_train_y,columns=['Survived'], dtype='category')

# %% # 生存比例查看
plt.bar(train_y['Survived'].cat.categories, train_y['Survived'].value_counts())
# 資料標籤
for a,b in zip(train_y['Survived'].cat.categories, train_y['Survived'].value_counts()):  
    plt.text(a, b+100, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
# y軸上限
plt.ylim(0, len(train_y))
# x軸label
plt.xticks(train_y['Survived'].cat.categories,("Yes", "No"))
# x軸標題
plt.xlabel('the outcome for each passenger')
# y軸標題
plt.ylabel('Numbers')
plt.savefig('./image/re_survived_bar.png')
plt.show()

#%% # SVM
svc_model = GridSearchCV(SVC(), param_grid={
                        "kernel":('linear', 'rbf', 'sigmoid')}, cv=KFold(n_splits=5),
                        scoring='accuracy').fit(train_x, train_y)
# svc_model.best_estimator_.get_params() 最佳參數
joblib.dump(svc_model, 'svc_model.pkl')

# %% # 讀取模型
svc_model = joblib.load('svc_model.pkl')

# 評估
mse = mean_squared_error(train_y, svc_model.predict(train_x))
mae = mean_absolute_error(svc_model.predict(train_x), train_y)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%svc_model.best_score_)

#%% #決策樹
dtree_model = GridSearchCV(tree.DecisionTreeClassifier(), param_grid={
                        "criterion":("gini", "entropy"),
                        "splitter": ("best", "random")}, cv=KFold(n_splits=5),
                        scoring='accuracy').fit(train_x, train_y)
# dtree_model.best_estimator_.get_params() 最佳參數
joblib.dump(dtree_model , 'dtree_model.pkl')

# %% # 讀取模型
dtree_model = joblib.load('dtree_model.pkl')

# 評估
mse = mean_squared_error(train_y, dtree_model.predict(train_x))
mae = mean_absolute_error(dtree_model.predict(train_x), train_y)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%dtree_model.best_score_)

#%% # 隨機森林
rforest_model = GridSearchCV(RandomForestClassifier(), param_grid={
                        "n_estimators":range(10, 100, 10),}, cv=KFold(n_splits=5),
                        scoring='accuracy').fit(train_x, train_y)
# rforest_model.best_estimator_.get_params() 最佳參數
joblib.dump(rforest_model , 'rforest_model.pkl')

# %% # 讀取模型
rforest_model= joblib.load('rforest_model.pkl')

# 評估
mse = mean_squared_error(train_y, rforest_model.predict(train_x))
mae = mean_absolute_error(rforest_model.predict(train_x), train_y)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%rforest_model.best_score_)

# %% # ROC
y_score = rforest_model.predict_proba(train_x)[:, 1]
fpr, tpr, threshold = roc_curve(train_y, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.png')
plt.show()

#%% # 邏輯迴歸
logistic_model = GridSearchCV(LogisticRegression(), 
                            param_grid={
                                'warm_start':('True', 'False'),
                                'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')},
                            cv=KFold(n_splits=5),
                            scoring='accuracy').fit(train_x, train_y)
# logistic_model.best_estimator_.get_params() 最佳參數
joblib.dump(logistic_model , 'logistic_model.pkl')

# %% # 讀取模型
logistic_model= joblib.load('logistic_model.pkl')

# 評估
mse = mean_squared_error(train_y, logistic_model.predict(train_x))
mae = mean_absolute_error(logistic_model.predict(train_x), train_y)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%logistic_model.best_score_)

#%% # KNN
knn_model = GridSearchCV(KNeighborsClassifier(),
                        param_grid={'n_neighbors':range(3, 10),
                                    'weights':('uniform', 'distance'),
                                    'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')},
                                cv=KFold(n_splits=5),
                                scoring='accuracy').fit(train_x, train_y)
# knn_model_model.best_estimator_.get_params() 最佳參數
joblib.dump(knn_model , 'knn_model.pkl')

# %% # 讀取模型
knn_model= joblib.load('knn_model.pkl')

# 評估
mse = mean_squared_error(train_y, knn_model.predict(train_x))
mae = mean_absolute_error(knn_model.predict(train_x), train_y)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%knn_model.best_score_)

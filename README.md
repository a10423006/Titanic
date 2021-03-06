# Predict survival on the Titanic with ML basics
### Kaggle Challenge: [Titanic-Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

## Tutorial Overview
> The sinking of the Titanic is one of the most infamous shipwrecks in history.
>
> On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
>
> While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
>
> In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

運用機器學習建立模型預測在鐵達尼號中可能生存的乘客。

## Data Set
原始訓練資料(train.csv) 891筆，測試資料(test.csv) 418筆
* Null 處理
```
pd.isnull(train_data).any()
```
| Columns | isNull | Null 數量 |處理方法 |
| ------- | ------ | :------: | :------: |
| PassengerId | False |
| Survived | False |
| Pclass | False |
| _**Name**_ | False | | _**Drop Column**_ |
| Sex | False |
| _**Age**_ | _**True*_ | _**177**_ | _**fillna(age_mean)**_ |
| SibSp | False |
| Parch | False |
| _**Ticket**_ | False | | _**Drop Column**_ |
| Fare | False |
| _**Cabin**_ | _**True**_ | _**687**_ | _**Drop Column**_ |
| _**Embarked**_ | _**True**_ | _**2**_ | _**Drop Rows**_ |

* Label encoding
```
labelencoder = LabelEncoder()
train_x['Sex'] = labelencoder.fit_transform(train_x['Sex'])
train_x['Embarked'] = labelencoder.fit_transform(train_x['Embarked'])
```

* 特徵選取

遞歸特徵消除法並使用決策樹和五折交叉驗證法
```
RFECV(estimator=DecisionTreeClassifier(), cv=KFold(n_splits=5), scoring='accuracy').fit(train_x, train_y)
```
![RFECV](https://github.com/a10423006/Titanic/blob/master/image/rfecv_cross_validation.png)

| Before | After |
| ------ | ----- |
| ![clean data](https://github.com/a10423006/Titanic/blob/master/image/pre_train_x.png) | ![clean data](https://github.com/a10423006/Titanic/blob/master/image/train_x.png) |

* Survived 比例 Bar Chart

有稍微類別失衡的現象，先建模看看情況吧！

![bar chart](https://github.com/a10423006/Titanic/blob/master/image/survived_bar.png)

## Classification
* Resample(RandomOverSampler)

### [imbalanced-learn API](https://imbalanced-learn.readthedocs.io/en/stable/index.html)
```
ros = RandomOverSampler(random_state=0)
resam_train_x, resam_train_y = ros.fit_sample(train[train.columns[:-1]], train.Survived)
```
![Resample bar](https://github.com/a10423006/Titanic/blob/master/image/re_survived_bar.png)

| | SVC | Decision Tree | Random Forest | Logistic Regression | KNN |
|:--------:|:-----:|:------:|:-----:|:-----:|:-----:|
| **Accuracy** | 0.769 | 0.850 | <font color="#dd0000">_0.864_</font> | 0.768 | 0.785 |
| **MSE** | 0.228 | 0.017 | _0.017_ | 0.209 | 0.017 |
| **MAE** | 0.228 | 0.017 | _0.017_ | 0.209 | 0.017 |

![ROC](https://github.com/a10423006/Titanic/blob/master/image/roc.png)

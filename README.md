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
---
原始訓練資料(train.csv) 891筆，測試資料(test.csv) 418筆
```
pd.isnull(train_data).any()
```
| Columns | isNull | 處理方法 |
| ------- | ------ | ------ |
| PassengerId | False |
| Survived | False |
| Pclass | False |
| Name | False |
| Sex | False |
| _*Age*_ | _*True*_ | fillna(age_mean) |
| SibSp | False |
| Parch | False |
| Ticket | False |
| Fare | False |
| _*Cabin*_ | _*True*_ | Drop |
| _*Embarked*_ | _*True*_ | Drop |

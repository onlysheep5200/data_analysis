import pandas as pd
import numpy as np
import xgboost as xgb

features = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
embark_map = {'S': 1, 'C': 2, 'Q': 3}
sex_map = {'mail': 1, 'female': 2}


def correct_data(df):
    df = df[features]
    df['family'] = df['Parch'] + df['SibSp'] + 1
    df['Sex'] = df['Sex'].map(sex_map).as_type(int)
    filled_embark = df['Embarked'].dropna().mode()[0]
    df['Embarked'] = df['Embarked'].fillna(filled_embark)
    df['Embarked'] = df['Embarked'].map(embark_map).as_type(int)
    filled_fare = df['Fare'].dropna().mode()[0]
    df['Fare'] = df['Fare'].fillna(filled_fare)
    return df


train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
test_df.info()
# train_df = correct_data(train_df)



import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('/Users/badyong/project/Fish.csv')
df = df[['Weight','Length1','Height','Width','Species']]
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777, stratify=y)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

pickle.dump(model, open('fis.pkl', 'wb'))
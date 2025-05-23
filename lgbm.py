# LGBM stands for Light Gradient Boosting Machine
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

cropdf = pd.read_csv("./crop_recommendation.csv")

X = cropdf.drop('label', axis=1)    # all columns except the last "label"
y = cropdf['label']                 # only the last column "label"

# splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 0)

# build the lightgbm model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
# X_new = pd.DataFrame([[90, 42, 43, 20.879744, 75, 5.5, 220]], columns=feature_names)
# newdata = model.predict(X_new)
# print(newdata)




# Import libraries
from sklearn.preprocessing import LabelBinarizer
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Data Reading(uploading)
data = pd.read_csv("myproject/iris.data")

# Data Exploration


def data_Explore(dataset):
    return dataset.shape, dataset.isnull().sum(), dataset.describe(), dataset.info(), data.head()


data_Explore(data)

# no missing values
# no outliers
# shape(150,5)
data["species"].value_counts()
# Multi-class classification(multi labels)
# Balanced(50:50:50)

# Dependent(target) AND  Independent Features(features)
features = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])

# Encoding Categorical Feature(target)
target = pd.get_dummies(target)

# Train_Test_Split
r = 1
np.random.seed(r)
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target, random_state=r)

# Create Our Model(RandomForestClassifier)
rfc = RandomForestClassifier()
rfc_model = rfc.fit(x_train, y_train)

# Save Our Model (writing mode)
pickle_out = open("myproject/iris.pkl", "wb")
pickle.dump(rfc_model, pickle_out)
pickle_out.close()

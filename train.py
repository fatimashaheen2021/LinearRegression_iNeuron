import numpy as np
import pandas as pd

import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

import pickle


boston = load_boston()
#### Loading the data
data=pd.DataFrame(boston.data,columns=boston.feature_names)
data["MDEV"]=boston.target

####Extracting Independent and dependent Variable  
X= data.drop(columns=["B","MDEV"],axis=1)
y= data["MDEV"]

model=LinearRegression()
model.fit(X,y)
print("model trained")

pickle.dump(model,open("lin_reg.sav", "wb"))
print("model saved")
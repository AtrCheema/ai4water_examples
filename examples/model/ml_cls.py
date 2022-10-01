"""
===================================
machine learning for classification
===================================
"""
import site
site.addsitedir("D:\\mytools\\AI4Water")

import numpy as np
import pandas as pd

from ai4water import Model
from ai4water.utils.utils import get_version_info
from sklearn.datasets import load_breast_cancer


for k,v in get_version_info().items():
    print(f"{k} version: {v}")

#%%

bunch = load_breast_cancer()

#%%

data = pd.DataFrame(np.column_stack([
    bunch['data'][0:1000, :], bunch['target'][0:1000,]
]),
    columns=bunch['feature_names'].tolist() + ['diagnostic'])

print(data.shape)

#%%

model = Model(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    model="XGBClassifier",
    split_random=True,
    x_transformation="zscore",
)

#%%
h = model.fit(data=data)

#%%
p = model.predict_on_validation_data(data=data)
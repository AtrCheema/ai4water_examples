"""
===============================================
Building Machine learning models for regression
===============================================
"""

import site
site.addsitedir("D:\\mytools\\AI4Water")

from ai4water import Model
from ai4water.utils.utils import get_version_info


for k,v in get_version_info().items():
    print(f"{k} version: {v}")
# %%

model = Model(model="RandomForestRegressor")

######################################

model = Model(model={"RandomForestRegressor": {"n_estimators": 200}})

#########################################

model = Model(model={"XGBRegressor": {"learning_rate": 0.01}})

#########################################

# model = Model(model={"CatBoostRegressor": {"learning_rate": 0.01}})


#########################################

#model = Model(model={"LGBMRegressor": {"num_leaves": 45}})


#########################################
# Custom model/estimator/algorithm
# --------------------------------

from sklearn.ensemble import RandomForestRegressor

class MyRF(RandomForestRegressor):
    pass

# uninitiated

model = Model(model=MyRF,
              ts_args={'lookback': 1},
              mode="regression")


#########################################
# uninitiated with arguments


model = Model(model={MyRF: {"n_estimators": 10}},
              ts_args={'lookback': 1},
              mode="regression")

#########################################
# initiated

model = Model(model=MyRF(), mode="regression")
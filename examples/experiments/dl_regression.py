"""
=========================================
Comparison of deep learning architectures
=========================================
"""

from ai4water.datasets import busan_beach
from ai4water.utils.utils import get_version_info
from ai4water.experiments import DLRegressionExperiments

########################################################
for k,v in get_version_info().items():
    print(f"{k} version: {v}")
# %%

data = busan_beach()
print(data)

#%%

comparisons = DLRegressionExperiments(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    val_fraction=0.0,
    epochs=20,
    ts_args={"lookback": 12},
    verbosity=0
)

#%%

comparisons.fit(data=data,
                include=['MLP',
                         'LSTM',
                         'CNNLSTM',
                         'TCN',
                         #"TFT",
                         "LSTMAutoEncoder",
                         ])

#%%

comparisons.compare_errors('r2', data=data)

###############################################

best_models = comparisons.compare_errors(
    'r2',
    data=data,
    cutoff_type='greater',
    cutoff_val=0.01)

################################################

comparisons.taylor_plot(data=data)

# %%
comparisons.compare_edf_plots(data=data)

# %%
comparisons.compare_regression_plots(data=data)

# %%
comparisons.compare_residual_plots(data=data)

# %%
comparisons.loss_comparison()

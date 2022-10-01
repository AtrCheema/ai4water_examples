"""
=========================================
Comparison of machine learning algorithms
=========================================
"""
import site
site.addsitedir("D:\\mytools\\AI4Water")
from ai4water.datasets import busan_beach
from ai4water.utils.utils import get_version_info
from ai4water.experiments import MLRegressionExperiments

for k,v in get_version_info().items():
    print(f"{k} version: {v}")

########################################################

data = busan_beach()

print(data)

# %%

comparisons = MLRegressionExperiments(
    input_features=data.columns.tolist()[0:-1],
    output_features=data.columns.tolist()[-1:],
    split_random=True,
    verbosity=0
)

# %%

comparisons.fit(data=data,
                run_type="dry_run")

# %%

comparisons.compare_errors('r2', data=data)

###############################################

best_models = comparisons.compare_errors(
    'r2',
    data=data,
    cutoff_type='greater',
    cutoff_val=0.01)

################################################

comparisons.taylor_plot(data=data)


#%%

comparisons.taylor_plot(data=data)

# %%
comparisons.compare_edf_plots(data=data)

# %%
comparisons.compare_regression_plots(data=data)

# %%
comparisons.compare_residual_plots(data=data)

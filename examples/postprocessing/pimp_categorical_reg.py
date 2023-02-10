"""
===============================================
Permutation Importance for categorical features
===============================================
"""


from ai4water import Model
from ai4water.datasets import busan_beach, mg_photodegradation
from ai4water.postprocessing.explain import PermutationImportance

# %%

cat_map = {'Catalyst': list(range(9, 24)), 'Anions': list(range(24, 30))}
mg_data, cat_enc, an_enc = mg_photodegradation(encoding="ohe")
print(mg_data.shape)

# %%

model = Model(model="XGBRegressor", verbosity=0)

# %%

model.fit(data=mg_data)

# %%
x, y = model.training_data(data=mg_data)

# %%
pimp = PermutationImportance(
    model.predict, x, y,
    save=False,
    cat_map=cat_map,
    feature_names=model.input_features,
    n_repeats=2)

# %%

pimp.plot_1d_pimp()

# %%

pimp.plot_1d_pimp("barchart")

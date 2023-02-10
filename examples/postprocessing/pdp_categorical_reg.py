"""
=========================
PDP for categorical data
=========================
"""

from ai4water import Model
from ai4water.datasets import mg_photodegradation
from ai4water.postprocessing.explain import PartialDependencePlot

# %%

data, cat_enc, an_enc = mg_photodegradation(encoding="ohe")
model = Model(model="XGBRegressor", verbosity=0)

# %%
model.fit(data=data)
# %%

x, _ = model.training_data(data=data)
# %%

pdp = PartialDependencePlot(model.predict, x, model.input_features,
                            num_points=14, save=False)

# %%

feature = [f for f in model.input_features if f.startswith('Catalyst_type')]
pdp.plot_1d(feature)

# %%

pdp.plot_1d(feature, show_dist_as="grid")

# %%

pdp.plot_1d(feature, show_dist=False)

# %%

pdp.plot_1d(feature, show_dist=False, ice=False)

# %%

pdp.plot_1d(feature, show_dist=False, ice=False, model_expected_value=True)

# %%

pdp.plot_1d(feature, show_dist=False, ice=False, feature_expected_value=True)

# %%

pdp.plot_1d(feature, ice_only=True, ice_color="red")

# %%

pdp.plot_1d(feature, ice_only=True, ice_color="Blues")
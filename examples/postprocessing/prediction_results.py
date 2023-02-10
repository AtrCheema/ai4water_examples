"""
====================================
postprocessing of prediction results
====================================
This file shows how to post-process prediction results
"""

from ai4water import Model
from ai4water.datasets import busan_beach
from ai4water.utils.utils import get_version_info

# sphinx_gallery_thumbnail_number = -1

#%%

for k,v in get_version_info().items():
    print(f"{k} version: {v}")
# %%

model = Model(model="XGBRegressor")
#%%

model.fit(data=busan_beach())

#%%
model.prediction_analysis(features="tide_cm", data=busan_beach(), show_percentile=True)

#%%

#%%

model.prediction_analysis(
    ['tide_cm', 'sal_psu'],
    data=busan_beach(),
    annotate_kws = {
        "annotate_counts":True,
        "annotate_colors":("black", "black"),
        "annotate_fontsize":10
    },
    custom_grid=[[-41.4, -20.0, 0.0, 20.0, 42.0],
                      [33.45, 33.7, 33.9, 34.05, 34.4]],
)

#%%

.. image:: https://readthedocs.org/projects/ai4water-examples/badge/?version=latest
    :target: https://ai4water.readthedocs.io/projects/Examples/en/latest/?badge=latest
    :alt: Documentation Status


Examples related to [ai4water](https://ai4water.readthedocs.io/) library.

Modle Building
===============

.. toctree::
   :maxdepth: 3

   Data Preparation for Regression task <model/data_prep_rgr.ipynb>
   Data splitting <model/data_splitting.ipynb>
   Building Machine Learning model for regression <model/ml_rgr.ipynb>
   Data preparation for classification task <model/data_prep_cls.ipynb>
   Building Machine Learning model for classification <model/ml_rgr.ipynb>
   Cross Validation <model/cross_val_rf.ipynb>
   Building Neural Network for regression <model/dl_rgr.ipynb>
   Building Neural Network for classification <model/dl_rgr.ipynb>
   Data preparation for time-series prediction <model/data_prep_ts.ipynb>
   LSTM for time series prediction <model/lstm_rgr.ipynb>
   Tab Transformer <model/tab_transformer.ipynb>
   FT Transformer <model/ft_transformer.ipynb>
   Input Attention LSTM model <model/interpretability_ia.ipynb>
   Temporal Fusion Transformer Model <model/tft.ipynb>
   Loading an existing model from config <model/from_config.ipynb>


Preprocessing
==============

.. toctree::
   :maxdepth: 3

   Feature engineering through transformations <preprocessing/transformations.ipynb>
   Missing data imputation <preprocessing/imputation.ipynb>
   HRU discretization <preprocessing/hru_discretization.ipynb>
   HRU discretization for Laos <preprocessing/hru_discretization_laos.ipynb>


Hyperparameter Optimization (HPO)
==================================
.. toctree::
   :maxdepth: 3

   Hpo for machine learning models (long) <hpo/hpo_ml_long.ipynb>
   Hpo for deep learning models (long) <hpo/hpo_nn_long.ipynb>
   Hpo for machine learning models (short) <hpo/hpo_ml_short.ipynb>
   Hpo for deep learning models (short) <hpo/hpo_nn_short.ipynb>
   Laoding results of hyperparameter optimization <hpo/load_hpo.ipynb>

Experiments
================
.. toctree::
   :maxdepth: 3

   Comparing machine learning algorithms for regression <experiments/ml_rgr_exp.ipynb>
   Comparing machine learning algorithms for classification <experiments/ml_cls_exp.ipynb>
   Comparing neural network architectures for regression <experiments/dl_rgr_exp.ipynb>
   Comparing neural network architectures for classification <experiments/dl_cls_exp.ipynb>
   Comparing performance of RF with data transformations <experiments/ml_transformation.ipynb>
   Effect of transformation on LSTM performance <experiments/dl_transformation.ipynb>

Postprocessing
================
.. toctree::
   :maxdepth: 3

   Analysis of prediction results <postprocessing/pred_analysis_rgr.ipynb>
   Partial dependence plot for regression task <postprocessing/pdp_rgr.ipynb>
   Partial dependence plot for regression task with categorical features <postprocessing/pdp_cat_rgr.ipynb>
   permutation importance for regression task <postprocessing/pimp_rgr.ipynb>
   permutation importance with categorical features <postprocessing/pimp_rgr_cat.ipynb>
   visualizing layers of neural networks <postprocessing/vis_nn_lyrs.ipynb>
   peeking inside LSTM <postprocessing/vis_lstm.ipynb>


Datasets
==============
.. toctree::
   :maxdepth: 3

   Beach Water Quality of Busan <datasets/busan_beach.ipynb>
   Mtropics dataset from Laos <datasets/mtropics_laos.ipynb>
   Global River Water Quality data <datasets/grqa.ipynb>
   Quadica <datasets/quadica.ipynb>
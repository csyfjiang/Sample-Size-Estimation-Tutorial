"""
sampsizeval -- sample-size / precision calculators for clinical prediction
models with a binary outcome, spanning the full study lifecycle:

    development  ->  external validation  ->  head-to-head model comparison

Modules
-------
development        Riley et al, BMJ 2020;368:m441        (is the training set big enough?)
validation_sim    Snell et al, J Clin Epidemiol 2021     (validation precision, simulation)
validation_closed Riley et al, BMJ 2023;383:e074821      (validation precision, closed form)
compare_auc       Jung, Pharm Stat 2024;23(4):557-569    (compare two correlated AUCs)
data              estimate the calculator inputs from a patient-level CSV
"""

from . import development, validation_closed, validation_sim, compare_auc, data

__version__ = "0.1.0"
__all__ = ["development", "validation_closed", "validation_sim", "compare_auc", "data"]

from config_sirius import CONFIG
from sirius import Sirius
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

base = Sirius(CONFIG)
lab = base.risk_lab()



from sirius import Sirius
from config_sirius import CONFIG

base = Sirius(CONFIG)
base.download_data()
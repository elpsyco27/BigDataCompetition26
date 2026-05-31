from lightgbm import LGBMRegressor

from config import MODEL_PARAMS


def build_model():
    return LGBMRegressor(**MODEL_PARAMS)

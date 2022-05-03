import os
import pathlib
import datetime

INITIAL_USDT_BALANCE = 100000
INITIAL_CRYPTO_BALANCE = 0

TRANSACTION_FEE_PERCENT = 1e-3

OPERATING_MODE = 'sharpe'

now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models"

if os.path.exists(TRAINED_MODEL_DIR):
    pass
else:
    os.makedirs(TRAINED_MODEL_DIR)
# models/__init__.py
from .base import BaseStockModel
from .gru_bi import GRUPredictor
# Future imports as we add more models
# from .lstm import LSTMPredictor
# from .transformers import TransformerPredictor

available_models = {
    "Bidirectional GRU": GRUPredictor,
    # Add more models here in the future:
    # "LSTM": LSTMPredictor,
    # "Transformer": TransformerPredictor,
}

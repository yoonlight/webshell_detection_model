from keras import layers, models
from keras.optimizers import adam_v2

MAX_DOCUMENT_LENGTH = 100


def load_model(lr: int = 0.001, n_words: int = 100) -> models.Model:
    input = layers.Input([MAX_DOCUMENT_LENGTH], name='input')
    network = layers.Embedding(input_dim=n_words+1, output_dim=30)(input)
    network = layers.LSTM(64)
    network = layers.Dense(1, "sigmoid")(network)
    model = models.Model(inputs=input, outputs=network)

    opt = adam_v2.Adam(lr)
    model.compile(opt, "binary_crossentropy")
    return model

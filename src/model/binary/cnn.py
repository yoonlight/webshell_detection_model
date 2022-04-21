from keras import layers, models
from keras.optimizers import adam_v2

MAX_DOCUMENT_LENGTH = 100


def load_model(lr: int = 0.001, n_words: int = 100) -> models.Model:
    input = layers.Input([MAX_DOCUMENT_LENGTH], name='input')
    network = layers.Embedding(input_dim=n_words+1, output_dim=30)(input)
    network = layers.Conv1D(64, 4, padding='valid', activation='relu',
                            kernel_regularizer="L2")(network)
    network = layers.MaxPooling1D()(network)
    network = layers.Conv1D(64, 4, padding='valid', activation='relu',
                            kernel_regularizer="L2")(network)
    network = layers.GlobalMaxPooling1D()(network)
    network = layers.Dropout(0.5)(network)
    network = layers.Dense(16)(network)
    network = layers.Dense(1, "sigmoid")(network)
    model = models.Model(inputs=input, outputs=network)

    opt = adam_v2.Adam(lr)
    model.compile(opt, "binary_crossentropy")
    return model

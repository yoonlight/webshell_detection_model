from keras import layers, models
from keras.optimizers import adam_v2

MAX_DOCUMENT_LENGTH = 100


def load_model(lr: int = 0.001, n_words: int = 100) -> models.Model:
    input = layers.Input([MAX_DOCUMENT_LENGTH], name='input')
    network = layers.Embedding(input_dim=n_words+1, output_dim=128)(input)
    branch1 = layers.Conv1D(128, 14, padding='valid', activation='relu',
                            kernel_regularizer="L2")(network)
    branch2 = layers.Conv1D(128, 15, padding='valid', activation='relu',
                            kernel_regularizer="L2")(network)
    branch3 = layers.Conv1D(128, 16, padding='valid', activation='relu',
                            kernel_regularizer="L2")(network)
    network = layers.Concatenate(axis=1)(
        [branch1, branch2, branch3])
    network = layers.GlobalMaxPooling1D()(network)
    network = layers.Dropout(0.5)(network)
    network = layers.Dense(2, "softmax")(network)
    model = models.Model(inputs=input, outputs=network)

    opt = adam_v2.Adam(lr)
    model.compile(opt, "categorical_crossentropy")
    return model

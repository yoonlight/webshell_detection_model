from model.softmax.cnn import load_model
from keras import metrics
import tensorflow_addons as tfa

metric = [metrics.Accuracy(), metrics.Recall(),
          metrics.Precision(), metrics.AUC(), tfa.metrics.F1Score()]

model = load_model()
model.compile(metrics=metric)
model.summary()
# model.fit()

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#get data/clean
path = Path("jsb_chorales/test/chorale_305.csv")
df = pd.read_csv(path)
df.columns = ["note0", "note1", "note2", "note3"]
print(df.head())


model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=(None, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(df["note0"], df["note1"], epochs=10)

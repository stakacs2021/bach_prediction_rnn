import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#load all chorales from a directory
def load_all_chorales(data_dir="jsb_chorales/train"):

    chorales = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(path)
        df.columns = ["note0", "note1", "note2", "note3"]
        #convert to numpy array
        chorale_array = df[["note0", "note1", "note2", "note3"]].values
        chorales.append(chorale_array)
    return chorales

#create sequences from chorales
def create_sequences(chorales, sequence_length, batch_size=32):
    #make long sequence of chorales
    all_sequences = []
    all_targets = []
    
    for chorale in chorales:
        #sliding windows
        for i in range(len(chorale) - sequence_length):
#4 sequence length
            seq = chorale[i:i + sequence_length] 
            target = chorale[i + sequence_length]  
            all_sequences.append(seq)
            all_targets.append(target)
    
    #convert to numpy arrays
    X = np.array(all_sequences, dtype=np.float32) 
    y = np.array(all_targets, dtype=np.float32)
    
    #create tf dataset 
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

#load data
print("Loading chorales...")
train_chorales = load_all_chorales("jsb_chorales/train")
valid_chorales = load_all_chorales("jsb_chorales/valid")
test_chorales = load_all_chorales("jsb_chorales/test")

print(f"Loaded {len(train_chorales)} training chorales")
print(f"Loaded {len(valid_chorales)} validation chorales")
print(f"Loaded {len(test_chorales)} test chorales")

# create datasets
#seuqence length
sequence_length = 32 
batch_size = 32
tf.random.set_seed(42)
np.random.seed(42)

#debug
print("Creating training sequences...")
train_set = create_sequences(train_chorales, sequence_length=sequence_length, 
                             batch_size=batch_size)
valid_set = create_sequences(valid_chorales, sequence_length=sequence_length, 
                            batch_size=batch_size)

#build model
print("Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length, 4)),
    tf.keras.layers.SimpleRNN(128, return_sequences=False),
    tf.keras.layers.Dense(64, activation="relu"),
    #output 4
    tf.keras.layers.Dense(4)  
])

model.compile(
    optimizer="adam",
    loss="mse", 
    metrics=["mae"]
)

model.summary()

#train
print("Training model...")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "bach_model_best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_set,
    validation_data=valid_set,
    epochs=20,
    callbacks=[model_checkpoint]
)

#load best model
try:
    model = tf.keras.models.load_model("bach_model_best.keras")
    print("Loaded best model from checkpoint.")
except Exception as e:
    print(f"Warning: Could not load saved model: {e}")
    print("Using the model from training instead.")

#generate chorale
def generate_chorale(model, seed_sequence, length=100, temperature=1.0):
    generated = seed_sequence.copy()
    current_sequence = seed_sequence.copy()
    
    for _ in range(length):
        #reshape
        input_seq = current_sequence[np.newaxis, :, :]
        
        #predict
        prediction = model.predict(input_seq, verbose=0)[0]  # Shape: (4,)
        
        #apply temp
        if temperature != 1.0:
            prediction = prediction / temperature
        
        #round t
        next_step = np.round(prediction).astype(np.int32)
        
        #makes sure no negative
        next_step = np.maximum(next_step, 0)
        
        # append to the sequence
        generated = np.vstack([generated, next_step])
        
        #window
        current_sequence = np.vstack([current_sequence[1:], next_step])
    
    return generated

#gnerate chorales
print("\nGenerating chorales...")

#seed
seed_chorale = test_chorales[0]
seed_sequence = seed_chorale[:sequence_length] 

#generate multipel 
for i in range(3):
    print(f"Generating chorale {i+1}...")
    generated = generate_chorale(model, seed_sequence, length=200, temperature=1.0)
    
    #save genrated as csv
    output_df = pd.DataFrame(generated, columns=["note0", "note1", "note2", "note3"])
    output_path = f"generated_chorale_{i+1}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

print("\nDone! Generated chorales saved as CSV files.")

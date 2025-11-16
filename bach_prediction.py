import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Load all chorales from a directory
def load_all_chorales(data_dir="jsb_chorales/train"):
    """Load all chorales from a directory and return as list of numpy arrays."""
    chorales = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(path)
        df.columns = ["note0", "note1", "note2", "note3"]
        # Convert to numpy array: shape (time_steps, 4)
        chorale_array = df[["note0", "note1", "note2", "note3"]].values
        chorales.append(chorale_array)
    return chorales

# Create sequences from chorales (similar to shakespearegen.py)
def create_sequences(chorales, sequence_length, shuffle=False, seed=None, batch_size=32):
    """
    Create sequences for training.
    Returns: tf.data.Dataset with (X, y) where:
        X: (batch, sequence_length, 4) - input sequences
        y: (batch, 4) - next time step (4 notes)
    """
    # Combine all chorales into one long sequence
    all_sequences = []
    all_targets = []
    
    for chorale in chorales:
        # Create sliding windows
        for i in range(len(chorale) - sequence_length):
            seq = chorale[i:i + sequence_length]  # (sequence_length, 4)
            target = chorale[i + sequence_length]  # (4,)
            all_sequences.append(seq)
            all_targets.append(target)
    
    # Convert to numpy arrays
    X = np.array(all_sequences, dtype=np.float32)  # (n_samples, sequence_length, 4)
    y = np.array(all_targets, dtype=np.float32)  # (n_samples, 4)
    
    # Shuffle if requested
    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

# Load data
print("Loading chorales...")
train_chorales = load_all_chorales("jsb_chorales/train")
valid_chorales = load_all_chorales("jsb_chorales/valid")
test_chorales = load_all_chorales("jsb_chorales/test")

print(f"Loaded {len(train_chorales)} training chorales")
print(f"Loaded {len(valid_chorales)} validation chorales")
print(f"Loaded {len(test_chorales)} test chorales")

# Create datasets
sequence_length = 32  # Length of input sequences
batch_size = 32
tf.random.set_seed(42)
np.random.seed(42)

print("Creating training sequences...")
train_set = create_sequences(train_chorales, sequence_length=sequence_length, 
                             shuffle=True, seed=42, batch_size=batch_size)
valid_set = create_sequences(valid_chorales, sequence_length=sequence_length, 
                            shuffle=False, batch_size=batch_size)

# Build model
print("Building model...")
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(128, return_sequences=False, input_shape=(sequence_length, 4)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(4)  # Output 4 notes
])

model.compile(
    optimizer="adam",
    loss="mse",  # Mean squared error for regression
    metrics=["mae"]  # Mean absolute error
)

model.summary()

# Train model
print("Training model...")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "bach_model_best.h5",
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

# Load best model
model = tf.keras.models.load_model("bach_model_best.h5")

# Generation function
def generate_chorale(model, seed_sequence, length=100, temperature=1.0):
    """
    Generate a chorale by iteratively predicting next time steps.
    
    Args:
        model: Trained model
        seed_sequence: Initial sequence of shape (sequence_length, 4)
        length: Number of time steps to generate
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated chorale as numpy array of shape (length, 4)
    """
    generated = seed_sequence.copy()  # Start with seed
    current_sequence = seed_sequence.copy()
    
    for _ in range(length):
        # Reshape for model input: (1, sequence_length, 4)
        input_seq = current_sequence[np.newaxis, :, :]
        
        # Predict next time step
        prediction = model.predict(input_seq, verbose=0)[0]  # Shape: (4,)
        
        # Apply temperature (optional, for more variety)
        if temperature != 1.0:
            prediction = prediction / temperature
        
        # Round to nearest integer (note indices)
        next_step = np.round(prediction).astype(np.int32)
        
        # Ensure notes are non-negative (0 means no note)
        next_step = np.maximum(next_step, 0)
        
        # Append to generated sequence
        generated = np.vstack([generated, next_step])
        
        # Update current sequence (sliding window)
        current_sequence = np.vstack([current_sequence[1:], next_step])
    
    return generated

# Generate chorales
print("\nGenerating chorales...")

# Use a seed from a test chorale as seed
seed_chorale = test_chorales[0]
seed_sequence = seed_chorale[:sequence_length]  # First sequence_length time steps

# Generate multiple chorales
for i in range(3):
    print(f"Generating chorale {i+1}...")
    generated = generate_chorale(model, seed_sequence, length=200, temperature=1.0)
    
    # Save as CSV
    output_df = pd.DataFrame(generated, columns=["note0", "note1", "note2", "note3"])
    output_path = f"generated_chorale_{i+1}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

print("\nDone! Generated chorales saved as CSV files.")

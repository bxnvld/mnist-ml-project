"""
MNIST Model Training Script
Trains a CNN model for digit recognition and saves it with version info
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os
from datetime import datetime

def create_model():
    """Create a simple CNN model for MNIST"""
    model = keras.Sequential([
        # Input layer - 28x28 grayscale images
        keras.layers.Input(shape=(28, 28, 1)),
        
        # Convolutional layers
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        #keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #keras.layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        keras.layers.Flatten(),
        #keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_save_model(version='v1.0.0', epochs=10):
    """Train model and save with version info"""
    
    print(f"🚀 Training MNIST model version {version}...")
    
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📊 Test data shape: {X_test.shape}")
    
    # Create and train model
    model = create_model()
    
    print("\n🎯 Model Architecture:")
    model.summary()
    # batch_size = 128
    print(f"\n🏋️ Training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test accuracy: {test_accuracy:.4f}")
    print(f"✅ Test loss: {test_loss:.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'models/mnist_model_{version}.h5'
    model.save(model_filename)
    print(f"\n💾 Model saved: {model_filename}")
    
    # Save metadata
    metadata = {
        'version': version,
        'timestamp': timestamp,
        'accuracy': float(test_accuracy),
        'loss': float(test_loss),
        'epochs': epochs,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'architecture': 'CNN',
        'input_shape': [28, 28, 1],
        'output_classes': 10
    }
    
    metadata_filename = f'models/mnist_model_{version}_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Metadata saved: {metadata_filename}")
    
    print("\n🎉 Training complete!")
    return model, metadata

if __name__ == '__main__':
    import sys
    
    # Get version from command line or use default
    version = sys.argv[1] if len(sys.argv) > 1 else 'v1.0.0'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    print("="*60)
    print("🔢 MNIST Digit Recognition - Model Training")
    print("="*60)
    
    model, metadata = train_and_save_model(version=version, epochs=epochs)
    
    print("\n" + "="*60)
    print("📋 Model Summary:")
    print(f"  Version: {metadata['version']}")
    print(f"  Accuracy: {metadata['accuracy']:.4f}")
    print(f"  Loss: {metadata['loss']:.4f}")
    print("="*60)
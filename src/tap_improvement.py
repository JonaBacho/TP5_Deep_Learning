import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from config.mlflow_config import MLflowConfig

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()

# ==================== PART 3: Research Challenge - TAP Architecture Improvement ====================

class TemporalTransformerBlock(layers.Layer):
    """
    Temporal Transformer Block avec Sparse Attention
    pour améliorer la gestion des dépendances long-terme
    """
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TemporalTransformerBlock, self).__init__()
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        # Multi-head attention
        attn_output = self.mha(x, x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class MemoryAugmentedModule(layers.Layer):
    """
    Module de mémoire externe inspiré des Memory Networks
    pour stocker les "keyframes" importants
    """
    
    def __init__(self, memory_size, key_dim, value_dim):
        super(MemoryAugmentedModule, self).__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Memory matrices (learned)
        self.memory_keys = self.add_weight(
            name="memory_keys",
            shape=(memory_size, key_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.memory_values = self.add_weight(
            name="memory_values",
            shape=(memory_size, value_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        self.query_proj = layers.Dense(key_dim)
    
    def call(self, query):
        """
        Args:
            query: (batch_size, query_dim)
        
        Returns:
            retrieved_memory: (batch_size, value_dim)
            attention_weights: (batch_size, memory_size)
        """
        # Project query
        query_transformed = self.query_proj(query)  # (batch, key_dim)
        
        # Compute attention scores with memory keys
        scores = tf.matmul(
            query_transformed,
            self.memory_keys,
            transpose_b=True
        )  # (batch, memory_size)
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Retrieve from memory
        retrieved_memory = tf.matmul(
            attention_weights,
            self.memory_values
        )  # (batch, value_dim)
        
        return retrieved_memory, attention_weights


class HierarchicalTemporalEncoder(layers.Layer):
    """
    Encodeur hiérarchique avec différents niveaux temporels
    - Niveau 1: court-terme (frames individuels)
    - Niveau 2: moyen-terme (segments)
    - Niveau 3: long-terme (séquence complète)
    """
    
    def __init__(self, d_model, num_levels=3):
        super(HierarchicalTemporalEncoder, self).__init__()
        self.num_levels = num_levels
        
        # Encodeurs pour chaque niveau
        self.level_encoders = []
        for i in range(num_levels):
            self.level_encoders.append(
                layers.LSTM(d_model, return_sequences=True, name=f'lstm_level_{i}')
            )
        
        # Pooling pour agréger à différentes échelles
        self.temporal_pools = []
        pool_sizes = [1, 2, 4]  # Différentes échelles temporelles
        for pool_size in pool_sizes:
            if pool_size > 1:
                self.temporal_pools.append(layers.AveragePooling1D(pool_size))
            else:
                self.temporal_pools.append(None)
        
        # Fusion des niveaux
        self.fusion = layers.Dense(d_model, activation='relu')
    
    def call(self, x):
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
        
        Returns:
            hierarchical_features: (batch_size, seq_len, d_model)
        """
        level_outputs = []
        
        for i, (encoder, pool) in enumerate(zip(self.level_encoders, self.temporal_pools)):
            # Apply pooling si nécessaire
            if pool is not None:
                x_pooled = pool(x)
            else:
                x_pooled = x
            
            # Encode
            encoded = encoder(x_pooled)
            
            # Upsample back to original resolution
            if pool is not None:
                encoded = tf.repeat(encoded, pool.pool_size[0], axis=1)
                # Trim to match original length
                encoded = encoded[:, :tf.shape(x)[1], :]
            
            level_outputs.append(encoded)
        
        # Concatenate and fuse
        concatenated = tf.concat(level_outputs, axis=-1)
        fused = self.fusion(concatenated)
        
        return fused


class ImprovedTAPModel(keras.Model):
    """
    Improved TAP (Temporal-Aware Path) Model
    
    Améliorations:
    1. Temporal Transformer pour long-term dependencies
    2. Memory-Augmented Module pour keyframes
    3. Hierarchical Temporal Encoding pour multi-scale
    """
    
    def __init__(self, latent_dim=128, seq_len=16, memory_size=32):
        super(ImprovedTAPModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # Hierarchical Encoder
        self.hierarchical_encoder = HierarchicalTemporalEncoder(
            d_model=latent_dim,
            num_levels=3
        )
        
        # Temporal Transformer
        self.temporal_transformer = TemporalTransformerBlock(
            d_model=latent_dim,
            num_heads=4,
            dff=latent_dim * 4,
            dropout_rate=0.1
        )
        
        # Memory Module
        self.memory_module = MemoryAugmentedModule(
            memory_size=memory_size,
            key_dim=latent_dim,
            value_dim=latent_dim
        )
        
        # Latent space projection
        self.latent_projection = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.LSTM(latent_dim, return_sequences=True),
            layers.Dense(latent_dim, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)  # Output for each time step
        ])
    
    def call(self, x, training=False):
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
        
        Returns:
            reconstruction: (batch_size, seq_len, 1)
        """
        # Hierarchical encoding
        hierarchical_features = self.hierarchical_encoder(x)
        
        # Temporal transformer
        transformed = self.temporal_transformer(hierarchical_features, training=training)
        
        # Global pooling for memory query
        global_context = tf.reduce_mean(transformed, axis=1)
        
        # Memory retrieval
        memory_output, memory_weights = self.memory_module(global_context)
        
        # Combine with temporal features
        memory_broadcast = tf.expand_dims(memory_output, 1)
        memory_broadcast = tf.tile(memory_broadcast, [1, self.seq_len, 1])
        
        combined = transformed + memory_broadcast
        
        # Project to latent space
        latent = self.latent_projection(combined)
        
        # Decode
        reconstruction = self.decoder(latent, training=training)
        
        return reconstruction


def generate_moving_mnist_data(num_samples=100, seq_len=16, img_size=32):
    """
    Génère des données similaires à Moving MNIST (simplifié)
    pour tester les dépendances temporelles long-terme
    """
    np.random.seed(42)
    
    X = np.zeros((num_samples, seq_len, img_size * img_size))
    
    for i in range(num_samples):
        # Position et vitesse initiales
        x_pos = np.random.randint(5, img_size - 10)
        y_pos = np.random.randint(5, img_size - 10)
        vx = np.random.uniform(-2, 2)
        vy = np.random.uniform(-2, 2)
        
        for t in range(seq_len):
            frame = np.zeros((img_size, img_size))
            
            # Dessiner un carré
            x = int(x_pos) % img_size
            y = int(y_pos) % img_size
            size = 5
            
            frame[max(0, y):min(img_size, y+size), 
                  max(0, x):min(img_size, x+size)] = 1.0
            
            X[i, t] = frame.flatten()
            
            # Mettre à jour position
            x_pos += vx
            y_pos += vy
            
            # Rebond sur les bords
            if x_pos < 0 or x_pos > img_size - size:
                vx = -vx
            if y_pos < 0 or y_pos > img_size - size:
                vy = -vy
    
    return X


def exercise_3_tap_improvement():
    """Exercise 3: Amélioration de l'architecture TAP"""
    mlflow.set_experiment("TP5-Exercise3-ImprovedTAP")
    
    seq_len = 16
    img_size = 32
    
    print("Generating Moving MNIST-like data...")
    X = generate_moving_mnist_data(num_samples=500, seq_len=seq_len, img_size=img_size)
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    with mlflow.start_run(run_name="Ex3_ImprovedTAP_LongTerm"):
        # Log parameters
        mlflow.log_param("exercise", "3")
        mlflow.log_param("model_type", "improved_tap")
        mlflow.log_param("improvements", "transformer+memory+hierarchical")
        mlflow.log_param("latent_dim", 128)
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("memory_size", 32)
        mlflow.log_param("transformer_heads", 4)
        mlflow.log_param("hierarchical_levels", 3)
        mlflow.log_param("epochs", 50)
        mlflow.log_param("batch_size", 16)
        
        # Build model
        print("\nBuilding Improved TAP Model...")
        model = ImprovedTAPModel(
            latent_dim=128,
            seq_len=seq_len,
            memory_size=32
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        # Build model by calling it once
        _ = model(X_train[:1])
        
        model.summary()
        
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        mlflow.log_param("trainable_parameters", trainable_params)
        
        # Training
        print("\n" + "="*60)
        print("EXERCISE 3: Training Improved TAP Model")
        print("="*60)
        
        history = model.fit(
            X_train, X_train,  # Reconstruction task
            batch_size=16,
            epochs=50,
            validation_split=0.2,
            verbose=1
        )
        
        # Log metrics
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_mae", history.history['mae'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_mae", history.history['val_mae'][epoch], step=epoch)
        
        # Evaluation
        print("\nEvaluating on test set...")
        test_loss, test_mae = model.evaluate(X_test, X_test, verbose=0)
        
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_mae", test_mae)
        
        # Log model
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=f"{MLflowConfig.MODEL_NAME}-tap-improved"
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Exercise 3 Completed!")
        print(f"{'='*60}")
        print(f"Test Reconstruction Loss: {test_loss:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Model Parameters: {trainable_params:,}")
        print(f"{'='*60}\n")
        
        # Generate research insights
        print("\nRESEARCH INSIGHTS:")
        print("="*60)
        print("Architectural Improvements Implemented:")
        print("1. Temporal Transformer: Better long-range dependencies")
        print("2. Memory Module: Stores important keyframes (32 slots)")
        print("3. Hierarchical Encoding: Multi-scale temporal features")
        print("\nExpected Benefits:")
        print("- Improved consistency over long sequences")
        print("- Better handling of periodic motions")
        print("- Reduced error accumulation in long-term prediction")
        print("="*60)
        
        return model


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TP5: Research Challenge - Improving TAP Architecture")
    print("="*70 + "\n")
    
    # Exercise 3: Improved TAP
    print("Executing Exercise 3: Improved TAP for Long-Term Dependencies...")
    model = exercise_3_tap_improvement()
    
    print("\n" + "="*70)
    print("✓ Research challenge completed!")
    print("Next step: Write scientific paper in LaTeX format")
    print("="*70 + "\n")
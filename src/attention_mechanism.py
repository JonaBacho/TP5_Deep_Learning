import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config.mlflow_config import MLflowConfig

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()

# ==================== PART 1: Exercise 1 - Basic Attention Layer ====================

class SimpleAttention(layers.Layer):
    """
    Custom Attention Layer implémentant Scaled Dot-Product Attention
    
    Formule:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, hidden_dim)
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(SimpleAttention, self).build(input_shape)
    
    def call(self, x):
        """
        Args:
            x: Tensor (batch_size, seq_len, hidden_dim)
        
        Returns:
            context_vector: Tensor (batch_size, hidden_dim)
            alignment_weights: Tensor (batch_size, seq_len)
        """
        # Calculate attention scores
        # e_t = tanh(W * h_t + b)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        e = tf.squeeze(e, axis=-1)
        
        # Apply softmax to get alignment weights
        # alpha_t = softmax(e_t)
        alignment_weights = tf.nn.softmax(e, axis=-1)
        
        # Compute context vector (weighted sum)
        # context = sum(alpha_t * h_t)
        context_vector = tf.reduce_sum(
            x * tf.expand_dims(alignment_weights, -1),
            axis=1
        )
        
        return context_vector, alignment_weights


class ScaledDotProductAttention(layers.Layer):
    """
    Scaled Dot-Product Attention (version Transformer)
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
    
    def call(self, queries, keys, values, mask=None):
        """
        Args:
            queries: (batch_size, seq_len_q, d_k)
            keys: (batch_size, seq_len_k, d_k)
            values: (batch_size, seq_len_v, d_v)
            mask: Optional mask
        
        Returns:
            output: (batch_size, seq_len_q, d_v)
            attention_weights: (batch_size, seq_len_q, seq_len_k)
        """
        d_k = tf.cast(tf.shape(keys)[-1], tf.float32)
        
        # Calculate scores: QK^T / sqrt(d_k)
        scores = tf.matmul(queries, keys, transpose_b=True)
        scores = scores / tf.math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores += (mask * -1e9)
        
        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention weights to values
        output = tf.matmul(attention_weights, values)
        
        return output, attention_weights


def build_gru_attention_model(seq_length, feature_dim, hidden_units=64, num_classes=10):
    """
    Construit un modèle GRU + Attention pour classification de séquences
    """
    inputs = keras.Input(shape=(seq_length, feature_dim))
    
    # GRU Layer (return full sequences for attention)
    gru_output = layers.GRU(
        hidden_units,
        return_sequences=True,
        name='gru_layer'
    )(inputs)
    
    # Apply custom attention
    context_vector, attention_weights = SimpleAttention(name='attention_layer')(gru_output)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(context_vector)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='classification_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Modèle séparé pour visualiser l'attention
    attention_model = keras.Model(inputs=inputs, outputs=attention_weights)

    return model, attention_model


def generate_synthetic_sequence_data(num_samples=1000, seq_length=50, feature_dim=10, num_classes=3):
    """
    Génère des séquences synthétiques pour tester l'attention
    """
    np.random.seed(42)
    
    X = np.zeros((num_samples, seq_length, feature_dim))
    y = np.zeros(num_samples, dtype=np.int32)
    
    for i in range(num_samples):
        class_label = np.random.randint(0, num_classes)
        y[i] = class_label
        
        # Créer une séquence avec un pattern spécifique pour chaque classe
        for t in range(seq_length):
            if class_label == 0:
                # Classe 0: signal dans la première moitié
                if t < seq_length // 2:
                    X[i, t, :] = np.random.normal(1.0, 0.2, feature_dim)
                else:
                    X[i, t, :] = np.random.normal(0.0, 0.1, feature_dim)
            elif class_label == 1:
                # Classe 1: signal dans la seconde moitié
                if t >= seq_length // 2:
                    X[i, t, :] = np.random.normal(1.0, 0.2, feature_dim)
                else:
                    X[i, t, :] = np.random.normal(0.0, 0.1, feature_dim)
            else:
                # Classe 2: signal au milieu
                if seq_length // 3 < t < 2 * seq_length // 3:
                    X[i, t, :] = np.random.normal(1.0, 0.2, feature_dim)
                else:
                    X[i, t, :] = np.random.normal(0.0, 0.1, feature_dim)
    
    return X, y


def visualize_attention_weights(attention_weights, num_samples=3, save_path="attention_results"):
    """Visualise les poids d'attention"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        sns.heatmap(
            attention_weights[i:i+1],
            cmap='viridis',
            cbar=True,
            ax=axes[i]
        )
        axes[i].set_title(f'Attention Weights - Sample {i+1}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Sample')
    
    plt.tight_layout()
    save_file = os.path.join(save_path, 'attention_weights.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_file


def exercise_1_basic_attention():
    """Exercise 1: Implémentation et test de l'attention basique"""
    mlflow.set_experiment("TP5-Exercise1-BasicAttention")
    
    print("Generating synthetic sequence data...")
    X, y = generate_synthetic_sequence_data(num_samples=1000, seq_length=50, feature_dim=10)
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=3)
    
    with mlflow.start_run(run_name="Ex1_GRU_Attention"):
        mlflow.log_param("exercise", "1")
        mlflow.log_param("model_type", "gru_attention")
        mlflow.log_param("task", "sequence_classification")
        mlflow.log_param("seq_length", 50)
        mlflow.log_param("feature_dim", 10)
        mlflow.log_param("hidden_units", 64)
        mlflow.log_param("num_classes", 3)
        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch_size", 32)
        
        # Build model
        print("\nBuilding GRU + Attention model...")
        model, attention_model = build_gru_attention_model(
            seq_length=50,
            feature_dim=10,
            hidden_units=64,
            num_classes=3
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        # Count parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        mlflow.log_param("trainable_parameters", trainable_params)
        
        # Training
        print("\n" + "="*60)
        print("EXERCISE 1: Training GRU with Attention")
        print("="*60)
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=20,
            validation_split=0.2,
            verbose=1
        )
        
        # Log metrics
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Evaluation
        print("\nEvaluating on test set...")
        predictions = model.predict(X_test)
        attention_weights = attention_model.predict(X_test)
        
        test_loss = keras.losses.categorical_crossentropy(y_test_cat, predictions).numpy().mean()
        test_accuracy = (np.argmax(predictions, axis=1) == y_test).mean()
        
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Analyze attention span
        # Attention span: how many time steps have significant weights (>0.05)
        significant_threshold = 0.05
        attention_span = (attention_weights > significant_threshold).sum(axis=1).mean()
        mlflow.log_metric("average_attention_span", attention_span)
        
        # Visualize attention weights
        print("\nVisualizing attention weights...")
        viz_path = visualize_attention_weights(attention_weights, num_samples=5)
        mlflow.log_artifact(viz_path)
        
        # Log model
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=MLflowConfig.MODEL_NAME
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Exercise 1 Completed!")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Average Attention Span: {attention_span:.2f} time steps")
        print(f"{'='*60}\n")
        
        return model, attention_weights


# ==================== Theory Analysis ====================

def analyze_scaled_attention():
    """Analyse théorique du Scaled Dot-Product Attention"""
    mlflow.set_experiment("TP5-Theory-ScaledAttention")
    
    with mlflow.start_run(run_name="Scaled_Attention_Analysis"):
        mlflow.log_param("analysis_type", "theoretical")
        
        print("\n" + "="*70)
        print("THEORETICAL ANALYSIS: Scaled Dot-Product Attention")
        print("="*70)
        
        # Demonstrate why scaling is necessary
        d_k_values = [16, 64, 256, 512]
        
        print("\nWhy scaling factor 1/sqrt(d_k) is necessary:")
        print("-" * 70)
        
        for d_k in d_k_values:
            # Simulate attention scores without scaling
            q = np.random.randn(1, 10, d_k)
            k = np.random.randn(1, 10, d_k)
            
            scores_no_scale = np.matmul(q, k.transpose(0, 2, 1))
            scores_scaled = scores_no_scale / np.sqrt(d_k)
            
            var_no_scale = np.var(scores_no_scale)
            var_scaled = np.var(scores_scaled)
            
            print(f"d_k = {d_k:4d} | Variance without scaling: {var_no_scale:8.2f} | "
                  f"Variance with scaling: {var_scaled:6.2f}")
            
            mlflow.log_metric(f"variance_no_scale_dk{d_k}", var_no_scale)
            mlflow.log_metric(f"variance_scaled_dk{d_k}", var_scaled)
        
        print("\nConclusion:")
        print("- Without scaling: variance grows with d_k")
        print("- Large variances → extreme values → softmax saturates")
        print("- Scaling by 1/sqrt(d_k) → stable gradients")
        print("="*70 + "\n")
        
        # Self-Attention vs Cross-Attention
        print("\nSelf-Attention vs Cross-Attention:")
        print("-" * 70)
        print("Self-Attention:")
        print("  Q, K, V all come from the SAME sequence")
        print("  Used for: understanding relationships within a sequence")
        print("  Example: Transformer encoder, BERT")
        print()
        print("Cross-Attention:")
        print("  Q comes from one sequence, K and V from ANOTHER sequence")
        print("  Used for: aligning two different sequences")
        print("  Example: Encoder-Decoder attention, translation")
        print("="*70 + "\n")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TP5: Sequence Modeling and Attention Mechanisms")
    print("="*70 + "\n")
    
    # Theoretical analysis
    print("Running Theoretical Analysis...")
    analyze_scaled_attention()
    
    # Exercise 1: Basic Attention
    print("\nExecuting Exercise 1: Basic Attention Implementation...")
    model, attention_weights = exercise_1_basic_attention()
    
    print("\n" + "="*70)
    print("✓ Part 1 exercises completed successfully!")
    print("Check MLflow UI for detailed results")
    print("="*70 + "\n")
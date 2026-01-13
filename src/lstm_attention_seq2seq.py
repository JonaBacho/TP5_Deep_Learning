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

# ==================== PART 2: Exercise 2 - LSTM-Attention for Time Series ====================

class BahdanauAttention(layers.Layer):
    """
    Bahdanau (Additive) Attention pour Seq2Seq
    """
    
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
    def call(self, query, values):
        """
        Args:
            query: Decoder hidden state (batch_size, hidden_dim)
            values: Encoder outputs (batch_size, seq_len, hidden_dim)
        
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        # Expand query to match values shape
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Score = V * tanh(W1(query) + W2(values))
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, tf.squeeze(attention_weights, -1)


class Encoder(keras.Model):
    """Encoder Bi-directional LSTM"""
    
    def __init__(self, lstm_units, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.lstm_units = lstm_units
        self.lstm = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"lstm_units": self.lstm_units})
        return config

    
    def call(self, x):
        output, forward_h, forward_c, backward_h, backward_c = self.lstm(x)
        
        # Combiner les états forward et backward
        state_h = layers.Concatenate()([forward_h, backward_h])
        state_c = layers.Concatenate()([forward_c, backward_c])
        
        return output, state_h, state_c


class Decoder(keras.Model):
    """Decoder LSTM avec Cross-Attention"""
    
    def __init__(self, lstm_units, attention_units, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.lstm_units = lstm_units         
        self.attention_units = attention_units
        self.lstm = layers.LSTM(lstm_units * 2, return_sequences=True, return_state=True)
        self.attention = BahdanauAttention(attention_units)
        self.fc = layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lstm_units": self.lstm_units,
            "attention_units": self.attention_units,
        })
        return config
    
    def call(self, x, hidden, encoder_output):
        # x shape: (batch_size, 1, input_dim)
        # hidden shape: (batch_size, lstm_units * 2)
        # encoder_output shape: (batch_size, seq_len, lstm_units * 2)
        
        # Attention
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        
        # Concatenate input and context
        x = tf.concat([tf.squeeze(x, 1), context_vector], axis=-1)
        x = tf.expand_dims(x, 1)
        
        # LSTM
        output, state_h, state_c = self.lstm(x, initial_state=[hidden, hidden])
        
        # Prediction
        output = tf.squeeze(output, 1)
        prediction = self.fc(output)
        
        return prediction, state_h, attention_weights


def generate_time_series_data(num_samples=1000, input_len=50, output_len=10):
    """
    Génère des séries temporelles synthétiques complexes
    (combinaison de sinus, tendances long-terme)
    """
    np.random.seed(42)
    
    X = np.zeros((num_samples, input_len, 1))
    y = np.zeros((num_samples, output_len, 1))
    
    for i in range(num_samples):
        # Paramètres aléatoires
        freq1 = np.random.uniform(0.05, 0.2)
        freq2 = np.random.uniform(0.1, 0.3)
        phase1 = np.random.uniform(0, 2 * np.pi)
        phase2 = np.random.uniform(0, 2 * np.pi)
        trend = np.random.uniform(-0.01, 0.01)
        
        # Générer la série complète
        t = np.arange(input_len + output_len)
        series = (np.sin(2 * np.pi * freq1 * t + phase1) + 
                 0.5 * np.sin(2 * np.pi * freq2 * t + phase2) +
                 trend * t +
                 np.random.normal(0, 0.1, len(t)))
        
        X[i, :, 0] = series[:input_len]
        y[i, :, 0] = series[input_len:input_len + output_len]
    
    return X, y


def build_seq2seq_attention_model(input_len, output_len, lstm_units=64, attention_units=32):
    """
    Construit un modèle Seq2Seq avec Attention
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_len, 1))
    encoder = Encoder(lstm_units)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    
    # Decoder (teacher forcing during training)
    decoder_inputs = keras.Input(shape=(output_len, 1))
    decoder = Decoder(lstm_units, attention_units)
    
    # Decoder loop
    decoder_outputs = []
    attention_weights_list = []
    hidden = state_h
    
    for t in range(output_len):
        decoder_input = decoder_inputs[:, t:t+1, :]
        prediction, hidden, attention_weights = decoder(decoder_input, hidden, encoder_outputs)
        decoder_outputs.append(prediction)
        attention_weights_list.append(attention_weights)
    
    # Stack outputs
    combined = layers.Concatenate(axis=1)(decoder_outputs)
    decoder_outputs = layers.Reshape((output_len, 1))(combined)
    
    model = keras.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs
    )
    
    return model, encoder, decoder


def calculate_attention_span(attention_weights, threshold=0.02):
    """
    Calcule l'attention span: combien de time steps ont des poids significatifs
    """
    if threshold is None:
        threshold = 1.0 / attention_weights.shape[-1]

    significant_weights = attention_weights > threshold
    attention_span = significant_weights.sum(axis=-1).mean()
    
    # Calcule aussi la distance moyenne pondérée
    seq_len = attention_weights.shape[-1]
    positions = np.arange(seq_len)
    weighted_positions = (attention_weights * positions).sum(axis=-1)
    
    return attention_span, weighted_positions.mean()


def visualize_attention_heatmap(attention_weights, save_path="seq2seq_results"):
    """Visualise les poids d'attention en heatmap"""
    os.makedirs(save_path, exist_ok=True)
    
    # Prendre un échantillon
    sample_idx = 0
    attention = attention_weights[sample_idx]  # (output_len, input_len)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(attention, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Encoder Time Step (Input)')
    plt.ylabel('Decoder Time Step (Output)')
    plt.title('Attention Weights Heatmap (Seq2Seq)')
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'attention_heatmap.png')
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_file


def exercise_2_lstm_attention_seq2seq():
    """Exercise 2: LSTM-Attention pour prédiction de séries temporelles"""
    mlflow.set_experiment("TP5-Exercise2-LSTM-Attention-Seq2Seq")
    
    input_len = 50
    output_len = 10
    
    print("Generating time series data...")
    X, y = generate_time_series_data(num_samples=1000, input_len=input_len, output_len=output_len)
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    with mlflow.start_run(run_name="Ex2_BiLSTM_CrossAttention"):
        mlflow.log_param("exercise", "2")
        mlflow.log_param("model_type", "seq2seq_attention")
        mlflow.log_param("encoder", "bidirectional_lstm")
        mlflow.log_param("decoder", "lstm_cross_attention")
        mlflow.log_param("attention_mechanism", "bahdanau")
        mlflow.log_param("input_length", input_len)
        mlflow.log_param("output_length", output_len)
        mlflow.log_param("lstm_units", 64)
        mlflow.log_param("attention_units", 32)
        mlflow.log_param("epochs", 30)
        mlflow.log_param("batch_size", 32)
        
        # Build model
        print("\nBuilding Seq2Seq with Attention...")
        model, encoder, decoder = build_seq2seq_attention_model(
            input_len=input_len,
            output_len=output_len,
            lstm_units=64,
            attention_units=32
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        model.summary()
        
        # Count parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        mlflow.log_param("trainable_parameters", trainable_params)
        
        # Training (avec teacher forcing)
        print("\n" + "="*60)
        print("EXERCISE 2: Training Seq2Seq with Cross-Attention")
        print("="*60)
        
        history = model.fit(
            [X_train, y_train],  # Decoder input = target (teacher forcing)
            y_train,
            batch_size=32,
            epochs=30,
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
        test_loss, test_mae = model.evaluate([X_test, y_test], y_test, verbose=0)
        
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_mae", test_mae)
        
        # Analyze attention (need to extract attention weights)
        # For simplicity, we'll use a custom prediction loop
        print("\nAnalyzing attention patterns...")
        
        # Get encoder outputs
        encoder_outputs, state_h, state_c = encoder(X_test[:10])
        
        # Collect attention weights during decoding
        all_attention_weights = []
        hidden = state_h
        
        for t in range(output_len):
            decoder_input = y_test[:10, t:t+1, :]
            _, hidden, attention_weights = decoder(decoder_input, hidden, encoder_outputs)
            all_attention_weights.append(attention_weights.numpy())
        
        # Stack: (output_len, batch_size, input_len)
        attention_weights_array = np.stack(all_attention_weights, axis=0)
        attention_weights_array = np.transpose(attention_weights_array, (1, 0, 2))
        
        # Calculate attention span
        attention_span, avg_position = calculate_attention_span(attention_weights_array)
        
        mlflow.log_metric("attention_span", attention_span)
        mlflow.log_metric("avg_attention_position", avg_position)
        
        print(f"Average Attention Span: {attention_span:.2f} time steps")
        print(f"Average Attention Position: {avg_position:.2f}")
        
        # Visualize attention
        viz_path = visualize_attention_heatmap(attention_weights_array)
        mlflow.log_artifact(viz_path)
        
        # Log model
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=f"{MLflowConfig.MODEL_NAME}-seq2seq"
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Exercise 2 Completed!")
        print(f"{'='*60}")
        print(f"Test Loss (MSE): {test_loss:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Attention Span: {attention_span:.2f} steps")
        print(f"{'='*60}\n")
        
        return model, attention_weights_array


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TP5: LSTM-Attention for Time Series Forecasting")
    print("="*70 + "\n")
    
    # Exercise 2: Seq2Seq with Attention
    print("Executing Exercise 2: Seq2Seq with Cross-Attention...")
    model, attention_weights = exercise_2_lstm_attention_seq2seq()
    
    print("\n" + "="*70)
    print("✓ Part 2 exercises completed successfully!")
    print("Check MLflow UI for detailed results")
    print("="*70 + "\n")
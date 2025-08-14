import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('cnn/fer_model.keras')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization for size/performance
tflite_model = converter.convert()

# Save the TFLite model
with open('fer_model.tflite', 'wb') as f:
    f.write(tflite_model)
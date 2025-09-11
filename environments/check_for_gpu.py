import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print("GPU devices:", gpus)

# Warn if no GPU is found
if not gpus:
    print("***Warning***: No GPU found. Ensure you have a compatible GPU and the necessary drivers installed.")

# Test DDM simulation
@tf.function
def test_ddm():
    return tf.random.normal([1000, 100])

print("DDM test successful:", test_ddm().shape)
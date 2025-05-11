# Custom L1 Distance layer module

#Import Dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance Layer
class L1Dist(Layer):
    # Inheritance happens here
    def __init__(self, **kwargs): # kwargs -  keywords and arguments identifiable
        super().__init__()
        
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.abs(input_embedding - validation_embedding)
    
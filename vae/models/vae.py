import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class VariationalAutoEncoder(tf.Module):
    
    def __init__(self, data_shape, num_latent_dims):
        self._num_latent_dims = num_latent_dims
        self._encoder = self._make_encoder(data_shape, num_latent_dims)
        self._encoder.summary()
        self._decoder = self._make_decoder(num_latent_dims)
        self._decoder.summary()
        
        self._prior = tfd.MultivariateNormalDiag(
            loc=tf.zeros(num_latent_dims), 
            scale_diag=tf.ones(num_latent_dims))
        
    def decode(self, latent_vars):
        return self._decoder(latent_vars)
        
    def encode(self, data):
        params = self._encoder(data)
        mean = params[..., :self._num_latent_dims]
        logvar = params[..., self._num_latent_dims:]
        return mean, logvar
    
    def generate_data(self, batch_size):
        latent_vars = self.sample_latent_prior(batch_size)
        return self.generate_data_from_latent(latent_vars)
    
    def generate_data_from_latent(self, latent_vars):
        logits = self.decode(latent_vars)
        probs = tf.math.sigmoid(logits)
        unif_vals = tf.random.uniform(logits.shape)
        fake_data = unif_vals <= probs
        return fake_data, probs
    
    def sample_latent_prior(self, batch_size):
        return tf.random.normal([batch_size, self._num_latent_dims])
        
    def sample_latent_posterior(self, data):
        mean, logvar = self.encode(data)
        noise = tf.random.normal(mean.shape)
        return mean + tf.math.sqrt(tf.math.exp(logvar)) * noise
    
    def _make_decoder(self, num_latent_dims):
        model = keras.Sequential([
              layers.Input(shape=[num_latent_dims]),
              layers.Dense(units=7*7*32, activation=tf.nn.relu),
              layers.Reshape(target_shape=(7, 7, 32)),
              layers.Conv2DTranspose(
                  filters=64,
                  kernel_size=3,
                  strides=(2, 2),
                  padding='SAME',
                  activation='relu'),
              layers.Conv2DTranspose(
                  filters=32,
                  kernel_size=3,
                  strides=(2, 2),
                  padding='SAME',
                  activation='relu'),
              # No activation
              layers.Conv2DTranspose(
                  filters=1, kernel_size=3, strides=(1, 1), padding='SAME'),
        ])
        return model
    
    def _make_encoder(self, data_shape, num_latent_dims):
        model = keras.Sequential([
            layers.Input(shape=data_shape, dtype='float32'),
            layers.Conv2D(
                  filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            layers.Conv2D(
                  filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            layers.Flatten(),
            # No activation
            layers.Dense(2 * num_latent_dims)
        ])
        return model
    
    def _sample_latent_prior(self, batch_size):
        return self._prior.sample(batch_size)
        
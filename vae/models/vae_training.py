import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def compute_loss(model, data, latent_vars):
    mean, logvar = model.encode(data)

    kl = 0.5 * tf.reduce_sum(
        1.0 + logvar - tf.math.square(mean) - tf.math.exp(logvar),
        axis=-1)
    
    data_logits = model.decode(latent_vars)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=data_logits, labels=data)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    return -tf.reduce_mean(kl + logpx_z)


@tf.function
def train_step(model, optimizer, data):
    latent_vars = model.sample_latent_posterior(data)
    with tf.GradientTape() as tape:
        loss = compute_loss(model, data, latent_vars)
        tf.debugging.check_numerics(loss, 'loss')
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    return loss


def train_vae(model, dataset, num_epoch, optimizer):
    loss_metric = keras.metrics.Mean('loss', dtype=tf.float32)
    
    epoch_logs = []
    for epoch_idx in tqdm(range(num_epoch)):
        for batch_data in dataset:
            loss = train_step(model, optimizer, batch_data)
            
            loss_metric(loss)
        
        epoch_log_row = {'loss_mean': float(loss_metric.result().numpy())}
        print(epoch_log_row)
        epoch_logs.append(epoch_log_row)
    
    return epoch_logs

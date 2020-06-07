import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, data, latent_vars, beta):
    mean, logvar = model.encode(data)
    
    neg_kl = 0.5 * tf.reduce_sum(
        1.0 + logvar - tf.math.square(mean) - tf.math.exp(logvar),
        axis=-1)

    data_logits = model.decode(latent_vars)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=data_logits, labels=data)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return -tf.reduce_mean(beta * neg_kl + logpx_z)


#def compute_loss(model, data, latent_vars):
#    mean, logvar = model.encode(data)
#    
#    # kl = 0.5 * tf.reduce_sum(
#    #     1.0 + logvar - tf.math.square(mean) - tf.math.exp(logvar),
#    #     axis=-1)
#
#    data_logits = model.decode(latent_vars)
#    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
#            logits=data_logits, labels=data)
#    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#
#    logpz = log_normal_pdf(latent_vars, 0.0, 0.0)
#    logqz_x = log_normal_pdf(latent_vars, mean, logvar)
#    
#    return -tf.reduce_mean(logpx_z + logpz - logqz_x)
#    #return -tf.reduce_mean(kl - logpx_z)


@tf.function
def train_step(model, optimizer, data, beta):
    with tf.GradientTape() as tape:
        latent_vars = model.sample_latent_posterior(data)
        loss = compute_loss(model, data, latent_vars)
        tf.debugging.check_numerics(loss, 'loss')
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
    return loss


def train_vae(model, dataset, num_epoch, optimizer, beta=1.0):
    loss_metric = keras.metrics.Mean('loss', dtype=tf.float32)
    
    epoch_logs = []
    for epoch_idx in tqdm(range(num_epoch)):
        for batch_data in dataset:
            loss = train_step(model, optimizer, batch_data, beta)
            
            loss_metric(loss)
        
        epoch_log_row = {'loss_mean': float(loss_metric.result().numpy())}
        print(epoch_log_row)
        epoch_logs.append(epoch_log_row)
    
    return epoch_logs

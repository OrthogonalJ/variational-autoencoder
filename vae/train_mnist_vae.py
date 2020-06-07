import random
import os
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from vae.utils import coalesce_none, create_dir_if_missing, current_timestamp
from vae.models.vae_training import train_vae
from vae.models.vae import VariationalAutoEncoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#tf.debugging.enable_check_numerics()


#### CONSTANTS ####

NUM_EPOCH = 100
LATENT_DIM = 50
TRAIN_BUF = 60000
BATCH_SIZE = 64
TEST_BUF = 10000
ONLY_INCLUDE_DIGITS = [1,3,8]
LEARNING_RATE = 1e-2
DATA_SHAPE = [28, 28, 1]

BASE_EXPORT_DIR = './data/debug'

#### HELPERS ####

def preprocess_mnist_images_and_labels(images, labels, max_obs=None):
    if ONLY_INCLUDE_DIGITS is not None and len(ONLY_INCLUDE_DIGITS) > 0:
        images = images[np.isin(labels, ONLY_INCLUDE_DIGITS)]
    
    np.random.shuffle(images)
    np.random.shuffle(labels)
    
    num_obs = int(min(images.shape[0], coalesce_none(max_obs, float('inf'))))
    images = images[:num_obs]
    labels = labels[:num_obs]
    
    images = images.reshape([images.shape[0]] + DATA_SHAPE).astype('float32')
    
    # Normalizing the images to the range of [0., 1.]
    images /= 255.0
    
    # Binarization
    images[images >= 0.5] = 1.0
    images[images < 0.5] = 0.0
    images[images >= 0.5] = 1.0
    images[images < 0.5] = 0.0
    
    image_dataset = tf.data.Dataset.from_tensor_slices(images) \
        .shuffle(num_obs) \
        .batch(BATCH_SIZE)
    
    return image_dataset, images, labels


def main():
    mnist_data_raw = tf.keras.datasets.mnist.load_data()

    mnist_train_dataset, _, _ = preprocess_mnist_images_and_labels(
            mnist_data_raw[0][0], mnist_data_raw[0][1], max_obs=TRAIN_BUF)
    mnist_test_dataset, _, _ = preprocess_mnist_images_and_labels(
            mnist_data_raw[1][0], mnist_data_raw[1][1],max_obs=TEST_BUF)

    mnist_train_dataset = mnist_train_dataset.prefetch(1)

    del mnist_data_raw

    mnist_vae = VariationalAutoEncoder(
        data_shape=DATA_SHAPE, 
        num_latent_dims=LATENT_DIM)
    mnist_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    mnist_logs = train_vae(
        model=mnist_vae, 
        dataset=mnist_train_dataset, 
        num_epoch=NUM_EPOCH,
        optimizer=mnist_optimizer)

    exp_name = 'vae_{}'.format(current_timestamp())
    export_dir = os.path.join(BASE_EXPORT_DIR, exp_name)
    create_dir_if_missing(export_dir)
    
    model_export_dir = os.path.join(export_dir, 'model')
    mnist_vae.save_model(model_export_dir)
    #create_dir_if_missing(model_export_dir)
    #tf.saved_model.save(mnist_vae, model_export_dir)

    with open(os.path.join(export_dir, 'epoch-logs.json'), 'w') as fd:
        json.dump(json.dumps(mnist_logs), fd)

    print('DONE!')

    #plt.plot([row['loss_mean'] for row in mnist_logs])

    #latent_vars = mnist_vae.sample_latent_prior(10)
    #fake_data, probs = mnist_vae.generate_data_from_latent(latent_vars)

    # for i in range(3):
    #     plt.figure()
    #     plt.imshow(np.squeeze(probs[i]), cmap='gray')
    #     plt.show()

    # for i in range(3):
    #     plt.figure()
    #     plt.imshow(np.squeeze(fake_data[i]), cmap='gray')
    #     plt.show()

#### MAIN ####

if __name__ == '__main__':
    main()


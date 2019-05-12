from keras.models import Model
from keras.layers import (Conv2D, Activation, Dense, Lambda, Input,
    MaxPooling2D, Dropout, Flatten, Reshape, UpSampling2D, Concatenate)
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K

image_shape = (28, 28, 1)
original_dim = image_shape[0] * image_shape[1]
input_shape = (original_dim,)
num_classes = 36
batch_size = 128
latent_dim = 6

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_model():
    """Builds/compiles the model and returns (encoder, decoder, vae)."""

    # encoder
    inputs = Input(shape=input_shape)
    x = Reshape(image_shape)(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    label_inputs = Input(shape=(num_classes,), name='label')
    x = Concatenate()([latent_inputs, label_inputs])
    x = Dense(128, activation='relu')(x)
    x = Dense(14 * 14 * 32, activation='relu')(x)
    x = Reshape((14, 14, 32))(x)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    outputs = Reshape(input_shape)(x)

    decoder = Model([latent_inputs, label_inputs], outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # variational autoencoder
    outputs = decoder([encoder(inputs)[2], label_inputs])
    vae = Model([inputs, label_inputs], outputs, name='vae_mlp')
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    # loss function
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return encoder, decoder, vae

def from_weights(weights_file):
    """Build a model and load its pretrained weights from `weights_file`."""
    encoder, decoder, vae = build_model()
    vae.load_weights(weights_file)
    return encoder, decoder, vae

if __name__ == '__main__':
    encoder, decoder, vae = from_weights('vae_cnn_mnist.h5')
    # decoder.save('decoder.h5')
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(decoder, 'docs')


from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, Reshape, Permute, RepeatVector, Concatenate, Conv3D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
from keras.initializers import RandomNormal

from functools import partial

import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt

from music21 import midi
from music21 import note, stream, duration, tempo



class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class MuseGAN():
    def __init__(self
        , input_dim
        , critic_learning_rate
        , generator_learning_rate
        , optimiser
        , grad_weight
        , z_dim
        , batch_size
        , n_tracks
        , n_bars
        , n_steps_per_bar
        , n_pitches
        ):

        self.name = 'gan'

        self.input_dim = input_dim

        self.critic_learning_rate = critic_learning_rate

        self.generator_learning_rate = generator_learning_rate
        
        self.optimiser = optimiser

        self.z_dim = z_dim
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches

        self.weight_init = RandomNormal(mean=0., stddev=0.02) # 'he_normal' #RandomNormal(mean=0., stddev=0.02)
        self.grad_weight = grad_weight
        self.batch_size = batch_size


        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        self._build_critic()
        self._build_generator()

        self._build_adversarial()

    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, interpolated_samples)[0]

        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)

    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
        return layer

    def conv(self, x, f, k, s, a, p):
        x = Conv3D(
                    filters = f
                    , kernel_size = k
                    , padding = p
                    , strides = s
                    , kernel_initializer = self.weight_init
                    )(x)
        if a =='relu':
            x = Activation(a)(x)
        elif a== 'lrelu':
            x = LeakyReLU()(x)

        
        return x

    def _build_critic(self):

        critic_input = Input(shape=self.input_dim, name='critic_input')

        x = critic_input

        x = self.conv(x, f=128, k = (2,1,1), s = (1,1,1), a = 'lrelu', p = 'valid')
        x = self.conv(x, f=128, k = (self.n_bars - 1,1,1), s = (1,1,1), a = 'lrelu', p = 'valid')
        
        x = self.conv(x, f=128, k = (1,1,12), s = (1,1,12), a = 'lrelu', p = 'same')
        x = self.conv(x, f=128, k = (1,1,7), s = (1,1,7), a = 'lrelu', p = 'same')
        x = self.conv(x, f=128, k = (1,2,1), s = (1,2,1), a = 'lrelu', p = 'same')
        x = self.conv(x, f=128, k = (1,2,1), s = (1,2,1), a = 'lrelu', p = 'same')
        x = self.conv(x, f=256, k = (1,4,1), s = (1,2,1), a = 'lrelu', p = 'same')
        x = self.conv(x, f=512, k = (1,3,1), s = (1,2,1), a = 'lrelu', p = 'same')

        x = Flatten()(x)

        x = Dense(1024, kernel_initializer = self.weight_init)(x)
        x = LeakyReLU()(x)
        
        critic_output = Dense(1, activation=None, kernel_initializer = self.weight_init)(x)
        

        self.critic = Model(critic_input, critic_output)

    def conv_t(self, x, f, k, s, a, p, bn):
        x = Conv2DTranspose(
                    filters = f
                    , kernel_size = k
                    , padding = p
                    , strides = s
                    , kernel_initializer = self.weight_init
                    )(x)

        if bn:
            x = BatchNormalization(momentum = 0.9)(x)

        if a == 'relu':
            x = Activation(a)(x)
        elif a == 'lrelu':
            x = LeakyReLU()(x)
        
        return x

    def TemporalNetwork(self):

        input_layer = Input(shape=(self.z_dim,), name='temporal_input')

        x = Reshape([1,1,self.z_dim])(input_layer)
        x = self.conv_t(x, f=1024, k=(2,1), s=(1,1), a= 'relu', p = 'valid', bn = True)
        x = self.conv_t(x, f=self.z_dim, k=(self.n_bars - 1,1), s=(1,1), a= 'relu', p = 'valid', bn = True)

        output_layer = Reshape([self.n_bars, self.z_dim])(x)

        return Model(input_layer, output_layer)

    def BarGenerator(self):

        input_layer = Input(shape=(self.z_dim * 4,), name='bar_generator_input')

        x = Dense(1024)(input_layer)
        x = BatchNormalization(momentum = 0.9)(x)
        x = Activation('relu')(x)

        x = Reshape([2,1,512])(x)
        x = self.conv_t(x, f=512, k=(2,1), s=(2,1), a= 'relu',  p = 'same', bn = True)
        x = self.conv_t(x, f=256, k=(2,1), s=(2,1), a= 'relu', p = 'same', bn = True)
        x = self.conv_t(x, f=256, k=(2,1), s=(2,1), a= 'relu', p = 'same', bn = True)
        x = self.conv_t(x, f=256, k=(1,7), s=(1,7), a= 'relu', p = 'same',bn = True)
        x = self.conv_t(x, f=1, k=(1,12), s=(1,12), a= 'tanh', p = 'same', bn = False)

        output_layer = Reshape([1, self.n_steps_per_bar , self.n_pitches ,1])(x)

        return Model(input_layer, output_layer)

    def _build_generator(self):

        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input')
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')

        # CHORDS -> TEMPORAL NETWORK
        self.chords_tempNetwork = self.TemporalNetwork()
        self.chords_tempNetwork.name = 'temporal_network'
        chords_over_time = self.chords_tempNetwork(chords_input) # [n_bars, z_dim]
        
        # MELODY -> TEMPORAL NETWORK
        melody_over_time = [None] * self.n_tracks # list of n_tracks [n_bars, z_dim] tensors
        self.melody_tempNetwork = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.melody_tempNetwork[track] = self.TemporalNetwork()
            melody_track = Lambda(lambda x: x[:,track,:])(melody_input)
            melody_over_time[track] = self.melody_tempNetwork[track](melody_track)


        # CREATE BAR GENERATOR FOR EACH TRACK
        self.barGen = [None] * self.n_tracks
        for track in range(self.n_tracks):
            self.barGen[track] = self.BarGenerator()

        # CREATE OUTPUT FOR EVERY TRACK AND BAR
        bars_output = [None] * self.n_bars
        for bar in range(self.n_bars):
            track_output = [None] * self.n_tracks

            c = Lambda(lambda x: x[:,bar,:], name = 'chords_input_bar_' + str(bar))(chords_over_time)  # [z_dim]
            s = style_input # [z_dim]

            for track in range(self.n_tracks):

                m = Lambda(lambda x: x[:,bar,:])(melody_over_time[track]) # [z_dim]
                g = Lambda(lambda x: x[:,track,:])(groove_input) # [z_dim]
                
                z_input = Concatenate(axis = 1, name = 'total_input_bar_{}_track_{}'.format(bar, track))([c,s,m,g])
                
                track_output[track] = self.barGen[track](z_input)

            bars_output[bar] = Concatenate(axis = -1)(track_output)

        generator_output = Concatenate(axis = 1, name = 'concat_bars')(bars_output)

        self.generator = Model([chords_input, style_input, melody_input, groove_input], generator_output)




    def get_opti(self, lr):
        if self.optimiser == 'adam':
            opti = Adam(lr=lr, beta_1=0.5, beta_2 = 0.9)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(lr=lr)
        else:
            opti = Adam(lr=lr)

        return opti


    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def _build_adversarial(self):
                
        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.set_trainable(self.generator, False)

        # Image input (real sample)
        real_img = Input(shape=self.input_dim)

        # Fake image
        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input')
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')


        fake_img = self.generator([chords_input, style_input, melody_input, groove_input])

        # critic determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'interpolated_samples' argument
        self.partial_gp_loss = partial(self.gradient_penalty_loss,
                          interpolated_samples=interpolated_img)
        self.partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, chords_input, style_input, melody_input, groove_input],
                            outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(
            loss=[self.wasserstein,self.wasserstein, self.partial_gp_loss]
            ,optimizer=self.get_opti(self.critic_learning_rate)
            ,loss_weights=[1, 1, self.grad_weight]
            )
        
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)

        # Sampled noise for input to generator
        chords_input = Input(shape=(self.z_dim,), name='chords_input')
        style_input = Input(shape=(self.z_dim,), name='style_input')
        melody_input = Input(shape=(self.n_tracks, self.z_dim), name='melody_input')
        groove_input = Input(shape=(self.n_tracks, self.z_dim), name='groove_input')


        # Generate images based of noise
        img = self.generator([chords_input, style_input, melody_input, groove_input])
        # Discriminator determines validity
        model_output = self.critic(img)
        # Defines generator model
        self.model = Model([chords_input, style_input, melody_input, groove_input], model_output)

        self.model.compile(optimizer=self.get_opti(self.generator_learning_rate)
        , loss=self.wasserstein
        )

        self.set_trainable(self.critic, True)

    def train_critic(self, x_train, batch_size, using_generator):

        valid = np.ones((batch_size,1), dtype=np.float32)
        fake = -np.ones((batch_size,1), dtype=np.float32)
        dummy = np.zeros((batch_size, 1), dtype=np.float32) # Dummy gt for gradient penalty

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]
        
        chords_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        style_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        melody_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))

        d_loss = self.critic_model.train_on_batch([true_imgs, chords_noise, style_noise,melody_noise,groove_noise], [valid, fake, dummy])
        return d_loss

    def train_generator(self, batch_size):
        valid = np.ones((batch_size,1), dtype=np.float32)
        
        chords_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        style_noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        melody_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (batch_size, self.n_tracks, self.z_dim))

        return self.model.train_on_batch([chords_noise, style_noise,melody_noise,groove_noise], valid)


    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches = 10
    , n_critic = 5
    , using_generator = False):

        for epoch in range(self.epoch, self.epoch + epochs):

            if epoch % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic

            for _ in range(critic_loops):
                d_loss = self.train_critic(x_train, batch_size, using_generator)

            g_loss = self.train_generator(batch_size)

            
            print ("%d (%d, %d) [D loss: (%.1f)(R %.1f, F %.1f, G %.1f)] [G loss: %.1f]" % (epoch, critic_loops, 1, d_loss[0], d_loss[1],d_loss[2],d_loss[3],g_loss))
            

            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

            # If at save interval => save generated image samples
            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                
                self.generator.save_weights(os.path.join(run_folder, 'weights/weights-g.h5'))
                
                self.critic.save_weights(os.path.join(run_folder, 'weights/weights-c.h5'))

                # with open(os.path.join(run_folder,"generator.json", "w") as json_file:
                #     json_file.write(self.generator.to_json())
                
                # with open(os.path.join(run_folder,"critic.json", "w") as json_file:
                #     json_file.write(self.critic.to_json())

                self.save_model(run_folder)

            if epoch % 500 == 0:
                self.generator.save_weights(os.path.join(run_folder, 'weights/weights-g-%d.h5' % (epoch)))
                self.critic.save_weights(os.path.join(run_folder, 'weights/weights-c-%d.h5' % (epoch)))
                

            self.epoch+=1


    def sample_images(self, run_folder):
        r = 5

        chords_noise = np.random.normal(0, 1, (r, self.z_dim))
        style_noise = np.random.normal(0, 1, (r, self.z_dim))
        melody_noise = np.random.normal(0, 1, (r, self.n_tracks, self.z_dim))
        groove_noise = np.random.normal(0, 1, (r, self.n_tracks, self.z_dim))

        gen_scores = self.generator.predict([chords_noise, style_noise, melody_noise, groove_noise])

        np.save(os.path.join(run_folder, "images/sample_%d.npy" % self.epoch), gen_scores)

        self.notes_to_midi(run_folder, gen_scores, 0)


    def binarise_output(self, output):
        # output is a set of scores: [batch size , steps , pitches , tracks]

        max_pitches = np.argmax(output, axis = 3)

        return max_pitches

    def notes_to_midi(self, run_folder, output, filename = None):

        for score_num in range(len(output)):

            max_pitches = self.binarise_output(output)

            midi_note_score = max_pitches[score_num].reshape([self.n_bars * self.n_steps_per_bar, self.n_tracks])
            parts = stream.Score()
            parts.append(tempo.MetronomeMark(number= 66))

            for i in range(self.n_tracks):
                last_x = int(midi_note_score[:,i][0])
                s= stream.Part()
                dur = 0
                

                for idx, x in enumerate(midi_note_score[:, i]):
                    x = int(x)
                
                    if (x != last_x or idx % 4 == 0) and idx > 0:
                        n = note.Note(last_x)
                        n.duration = duration.Duration(dur)
                        s.append(n)
                        dur = 0

                    last_x = x
                    dur = dur + 0.25
                
                n = note.Note(last_x)
                n.duration = duration.Duration(dur)
                s.append(n)
                
                parts.append(s)

            if filename is None:
                parts.write('midi', fp=os.path.join(run_folder, "samples/sample_{}_{}.midi".format(self.epoch, score_num)))
            else:
                parts.write('midi', fp=os.path.join(run_folder, "samples/{}.midi".format(filename)))


    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.critic, to_file=os.path.join(run_folder ,'viz/critic.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)



            
    def save(self, folder):

            with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
                pickle.dump([
                    self.input_dim
                    , self.critic_learning_rate
                    , self.generator_learning_rate
                    , self.optimiser
                    , self.grad_weight
                    , self.z_dim
                    , self.batch_size
                    , self.n_tracks
                    , self.n_bars
                    , self.n_steps_per_bar
                    , self.n_pitches
                    ], f)

            self.plot_model(folder)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.critic.save(os.path.join(run_folder, 'critic.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))
        # pickle.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

    def load_weights(self, run_folder, epoch=None):

        if epoch is None:

            self.generator.load_weights(os.path.join(run_folder, 'weights', 'weights-g.h5'))
            self.critic.load_weights(os.path.join(run_folder, 'weights', 'weights-c.h5'))
        else:
            self.generator.load_weights(os.path.join(run_folder, 'weights', 'weights-g-{}.h5'.format(epoch)))
            self.critic.load_weights(os.path.join(run_folder, 'weights', 'weights-c-{}.h5'.format(epoch)))


    def draw_bar(self, data, score_num, bar, part):
        plt.imshow(data[score_num,bar,:,:,part].transpose([1,0]), origin='lower', cmap = 'Greys', vmin=-1, vmax=1)

    def draw_score(self, data, score_num):


        fig, axes = plt.subplots(ncols=self.n_bars, nrows=self.n_tracks,figsize=(12,8), sharey = True, sharex = True)
        fig.subplots_adjust(0,0,0.2,1.5,0,0)

        for bar in range(self.n_bars):
            for track in range(self.n_tracks):

                if self.n_bars > 1:
                    axes[track, bar].imshow(data[score_num,bar,:,:,track].transpose([1,0]), origin='lower', cmap = 'Greys', vmin=-1, vmax=1)
                else:
                    axes[track].imshow(data[score_num,bar,:,:,track].transpose([1,0]), origin='lower', cmap = 'Greys', vmin=-1, vmax=1)
    
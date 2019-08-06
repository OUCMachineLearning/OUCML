import os.path
import tarfile, sys, math
from six.moves import urllib
import tensorflow as tf
from ops import batch_normal, conv2d, fully_connect, lrelu, de_conv, variable_summaries, Residual_G, Residual_D, avgpool2d
from utils import save_images
import numpy as np
import scipy

import time

#adversarial mutural learning
class SSGAN(object):

    # build model
    def __init__(self, flags, data):

        self.batch_size = flags.batch_size
        self.max_iters = flags.max_iters
        self.model_path = flags.model_path
        self.data_ob = data
        self.sample_size = flags.sample_size
        self.sample_path = flags.sample_path
        self.log_dir = flags.log_dir
        self.learning_rate = flags.learning_rate
        self.log_vars = []
        self.ssup = flags.ssup
        self.channel = data.channel
        self.output_size = data.image_size
        self.num_rotation = flags.num_rotation
        self.loss_type = flags.loss_type
        self.resnet = flags.resnet
        self.weight_rotation_loss_d = flags.weight_rotation_loss_d
        self.weight_rotation_loss_g = flags.weight_rotation_loss_g
        self.is_adam = flags.is_adam
        self.sn = flags.sn
        self.n_dis= flags.n_dis
        self.iter_power = flags.iter_power
        self.beta1 = flags.beta1
        self.beta2 = flags.beta2

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        self.test_z = tf.placeholder(tf.float32, [100, self.sample_size])
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.channel])
        self.r_labels = tf.placeholder(tf.float32, [self.batch_size])
        self.global_step = tf.Variable(0, trainable=False)
        self.softmax = None
        self.pool3 = None

    def build_model_GAN(self):

        #The loss of dis network
        self.fake_images = self.generate(self.z, batch_size=self.batch_size, resnet=self.resnet, is_train=True, reuse=False)
        self.test_fake_images = self.generate(self.test_z, batch_size=100, resnet=self.resnet, is_train=False, reuse=True)

        if self.ssup:
            self.images_all, self.images_all_label = self.Rotation_ALL(self.images)
            print "self.images_all, self.images_all_label", self.images_all.shape, self.images_all_label.shape
            self.fake_images_all, _ = self.Rotation_ALL(self.fake_images)
            print "self.fake_images_all", self.fake_images_all.shape
            _, self.D_real_pro_logits, self.D_real_rot_logits, self.D_real_rot_prob = self.discriminate(self.images_all, resnet=self.resnet, reuse=False)
            self.D_real_pro_logits = self.D_real_pro_logits[:self.batch_size]
            _, self.G_fake_pro_logits, self.G_fake_rot_logits, self.G_fake_rot_prob = self.discriminate(self.fake_images_all, resnet=self.resnet, reuse=True)
            self.G_fake_pro_logits = self.G_fake_pro_logits[:self.batch_size]
            print "self.D_real_pro_logits", self.D_real_pro_logits
            print "self.G_fake_pro_logits", self.G_fake_pro_logits

        else:
            _, self.D_real_pro_logits = self.discriminate(self.images, resnet=self.resnet, reuse=False)
            _, self.G_fake_pro_logits = self.discriminate(self.fake_images, resnet=self.resnet, reuse=True)

        if self.loss_type == 0:
            self.D_loss = tf.reduce_mean(self.G_fake_pro_logits) - tf.reduce_mean(self.D_real_pro_logits)
            self.G_loss = - tf.reduce_mean(self.G_fake_pro_logits)

            if self.sn == False:
                # gradient penalty from WGAN-GP
                self.differences = self.fake_images - self.images
                self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
                interpolates = self.images + (self.alpha * self.differences)
                _, discri_logits, _ = self.discriminate(interpolates, resnet=self.resnet, reuse=True)
                gradients = tf.gradients(discri_logits, [interpolates])[0]
                ##2 norm
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
                self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                tf.summary.scalar("gp_loss", self.gradient_penalty)
                self.D_loss += 10 * self.gradient_penalty
            else:
                self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.D_real_pro_logits - 0.0))

        elif self.loss_type == 1:
            self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.G_fake_pro_logits), logits=self.G_fake_pro_logits
            ))
            self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.D_real_pro_logits), logits=self.D_real_pro_logits
            ))

            self.D_loss = self.D_fake_loss + self.D_real_loss
            self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.G_fake_pro_logits), logits=self.G_fake_pro_logits
            ))

        elif self.loss_type == 2:
            self.D_loss = self.loss_dis(self.D_real_pro_logits, self.G_fake_pro_logits)
            self.G_loss = self.loss_gen(self.G_fake_pro_logits)

        else:
            self.D_loss = self.loss_hinge_dis(self.D_real_pro_logits, self.G_fake_pro_logits)
            self.G_loss = self.loss_hinge_gen(self.G_fake_pro_logits)

        if self.ssup:
            self.d_real_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    labels=tf.one_hot(self.images_all_label, self.num_rotation), logits=self.D_real_rot_logits))
            self.g_fake_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    labels=tf.one_hot(self.images_all_label, self.num_rotation), logits=self.G_fake_rot_logits))
          #   self.d_real_class_loss = - tf.reduce_mean(
          # tf.reduce_sum(tf.one_hot(self.images_all_label, self.num_rotation) * tf.log(self.D_real_rot_prob + 1e-10), 1))
          #   self.g_fake_class_loss = - tf.reduce_mean(
          # tf.reduce_sum(tf.one_hot(self.images_all_label, self.num_rotation) * tf.log(self.G_fake_rot_prob + 1e-10), 1))
            self.real_accuracy = self.Accuracy(self.D_real_rot_prob, self.images_all_label)
            self.fake_accuracy = self.Accuracy(self.G_fake_rot_prob, self.images_all_label)

            self.D_loss = self.D_loss + self.weight_rotation_loss_d * self.d_real_class_loss
            self.G_loss = self.G_loss + self.weight_rotation_loss_g * self.g_fake_class_loss

        else:
            self.D_loss = self.D_loss
            self.G_loss = self.G_loss

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        self.log_vars.append(("G_loss", self.G_loss))
        self.log_vars.append(("D_loss", self.D_loss))

        print("d_vars", len(self.d_vars))
        print("g_vars", len(self.g_vars))

        #create summaries to visualize weights
        for var in tf.trainable_variables():
            #print var.name.split('/')
            variable_summaries(var, var.name.split('/')[1])

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def Accuracy(self, pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(pred, 1), tf.int32), y),
                                      tf.float32))

    def loss_dis(self, d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))

        return l1 + l2

    def Rotation_by_R_label(self, image, r):
        image = tf.cond(tf.equal(tf.squeeze(r), tf.constant(0, dtype=tf.float32)), lambda: image, lambda: image)  #rotation=0
        image = tf.cond(tf.equal(tf.squeeze(r), tf.constant(1, dtype=tf.float32)), lambda: tf.image.flip_up_down(tf.image.transpose_image(image)), lambda: image)
        image = tf.cond(tf.equal(tf.squeeze(r), tf.constant(2, dtype=tf.float32)), lambda: tf.image.flip_left_right(tf.image.flip_up_down(image)), lambda: image)
        image = tf.cond(tf.equal(tf.squeeze(r), tf.constant(3, dtype=tf.float32)), lambda: tf.image.transpose_image(tf.image.flip_up_down(image)), lambda: image)

        return image

    def Rotation_ALL(self, images):

        images1 = tf.map_fn(lambda img: img, images,
                  dtype=tf.float32) # rotation = 0
        images2 = tf.map_fn(lambda img: tf.image.flip_up_down(tf.image.transpose_image(img)), images,
                  dtype=tf.float32) #rotation=90
        images3 = tf.map_fn(lambda img: tf.image.flip_left_right(tf.image.flip_up_down(img)), images,
                  dtype=tf.float32) #rotation=180
        images4 = tf.map_fn(lambda img: tf.image.transpose_image(tf.image.flip_up_down(img)), images,
                  dtype=tf.float32) #rotation=270
        images_all = tf.concat([images1, images2, images3, images4], axis=0)

        images_all_label = tf.constant(np.repeat(np.arange(self.num_rotation, dtype=np.int32), self.batch_size))

        return images_all, images_all_label

    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    #Hinge Loss + sigmoid
    def loss_hinge_dis(self, d_real_logits, d_fake_logits):
        loss = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
        loss += tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))

        return loss

    def loss_hinge_gen(self, d_fake_logits):
        loss = - tf.reduce_mean(d_fake_logits)
        return loss

    #kl loss
    def kl_loss_compute(self, logits1, logits2):
        #loss = tf.distributions.kl_divergence(logits2, logits1)
        pred1 = tf.nn.sigmoid(logits1)
        pred1 = tf.concat([pred1, 1 - pred1], axis=1)
        pred2 = tf.nn.sigmoid(logits2)
        pred2 = tf.concat([pred2, 1 - pred2], axis=1)

        loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

        return loss

    def test2(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config, graph=tf.get_default_graph()) as sess:

            sess.run(init)
            # optimization D
            self.saver.restore(sess, self.aml_model_path + str(94000))
            max_samples = 50000 / self.batch_size
            for i in range(max_samples):

                z_var = np.random.normal(0, 1, size=(self.batch_size, self.sample_size))
                samples = sess.run(self.fake_images, feed_dict={self.z: z_var})

                for j in range(self.batch_size):
                    save_images(samples[j:j+1], [1, 1],
                                '{}/{:02d}_{:02d}test.png'.format(self.sample_path, i, j))
    # do train
    def train(self):

        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 5000,
                                                   0.96, staircase=True)
        if self.is_adam:
            d_trainer = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1, beta2=self.beta2)
        else:
            d_trainer = tf.train.RMSPropOptimizer(learning_rate)

        d_gradients = d_trainer.compute_gradients(loss=self.D_loss, var_list=self.d_vars)
        #create summaries to visualize gradients
        # for grad, var in d_gradients:
        #     tf.summary.histogram(var.name + "/gradient", grad)
        opti_D = d_trainer.apply_gradients(d_gradients)

        if self.is_adam:
            g_trainer = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1, beta2=self.beta2)
        else:
            g_trainer = tf.train.RMSPropOptimizer(learning_rate)

        g_gradients = g_trainer.compute_gradients(self.G_loss, var_list=self.g_vars)
        opti_G = g_trainer.apply_gradients(g_gradients)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config, graph=tf.get_default_graph()) as sess:

            sess.run(init)
            step = 0
            lr_decay = 1
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                step = 0
                print(" No pretraining model")

            batch_num = 0
            print("Starting the training")

            while step <= self.max_iters:

                read_data_start_time = time.time()
                start_time = time.time()

                for j in range(self.n_dis):

                    train_data_list = self.data_ob.getNextBatch(batch_num)
                    read_data_end_time = time.time() - read_data_start_time

                    z_var = np.random.normal(0, 1, size=(self.batch_size, self.sample_size))
                    r_label = np.array([np.float(np.random.randint(0, 4)) for a in range(0, self.batch_size)])
                    sess.run(opti_D, feed_dict={self.images: train_data_list, self.z: z_var, self.r_labels: r_label, self.global_step:step})
                    batch_num += 1

                sess.run(opti_G, feed_dict={self.images: train_data_list, self.z: z_var, self.r_labels:r_label, self.global_step:step})
                all_time = time.time() - start_time

                if step % 50 == 0:

                    if self.ssup:
                        d_loss, g_loss, r_labels, real_accuracy, fake_accuracy, real_class_loss, fake_class_loss \
                            = sess.run([self.D_loss, self.G_loss, self.r_labels, self.real_accuracy,
                                        self.fake_accuracy, self.d_real_class_loss, self.g_fake_class_loss],
                        feed_dict={self.images: train_data_list, self.z: z_var, self.r_labels: r_label})
                        print("step %d D_loss=%.4f, G_loss = %.4f, real_accuracy=%.4f , fake_accuracy=%.4f, real_class_loss=%.4f,"
                              "fake_class_loss=%.4f, all_time=%.3f, read_data_time = %.3f," % (
                                step, d_loss, g_loss, real_accuracy, fake_accuracy, real_class_loss, fake_class_loss,
                                        all_time, read_data_end_time))
                    else:
                        d_loss, g_loss, r_labels = sess.run(
                            [self.D_loss, self.G_loss, self.r_labels],
                        feed_dict={self.images: train_data_list, self.z: z_var, self.r_labels: r_label})

                        print("step %d D_loss=%.4f, G_loss = %.4f, all_time=%.3f, read_data_time = %.3f," % (
                                        step, d_loss, g_loss, all_time, read_data_end_time))

                if step % 5000 == 0:

                    if self.ssup:
                        samples = sess.run([self.fake_images,
                                    self.images_all, self.fake_images_all],
                                           feed_dict={self.images:train_data_list, self.z: z_var, self.r_labels:r_label})
                        result_concat = np.concatenate([train_data_list[0:self.batch_size],
                                        samples[1][0:self.batch_size],
                                        samples[0][0:self.batch_size], samples[2][0+self.batch_size:self.batch_size*2],
                                        samples[2][self.batch_size*2:self.batch_size*3]], axis=0)
                        save_images(result_concat,
                                    [result_concat.shape[0]/self.batch_size, self.batch_size],
                                    '{}/{:02d}_output.jpg'.format(self.sample_path, step))
                    else:
                        samples = sess.run(self.fake_images, feed_dict={self.images: train_data_list,
                                            self.z: z_var, self.r_labels:r_label})
                        result_concat = np.concatenate([train_data_list[0:self.batch_size],samples[0:self.batch_size]], axis=0)
                        save_images(result_concat,
                                    [result_concat.shape[0] / self.batch_size, self.batch_size],
                                    '{}/{:02d}_output.jpg'.format(self.sample_path, step))

                if np.mod(step, 1000) == 0 and step != 0:

                    print("saving model")
                    self.saver.save(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(step)))
                    print("Computing Inception scores")

                    bs = 100
                    test_numbers = 50000
                    preds = []
                    n_batches = int(math.ceil(float(test_numbers) / float(bs)))

                    for i in range(n_batches):

                        if i % 10 == 0:
                            sys.stdout.write(".")
                            sys.stdout.flush()

                        z_var = np.random.normal(0, 1, size=(bs, self.sample_size))
                        #print z_var[0][0]
                        samples = sess.run(self.test_fake_images, feed_dict={self.test_z: z_var})
                        # save_images(samples[0:10], [1, 10],
                        #             '{}/{:02d}_testoutput.jpg'.format(self.sample_path, step))
                        #print self.softmax.shape
                        pred = sess.run(self.softmax, {'ExpandDims:0': (samples + 1.0) * 127.5})
                        preds.append(pred)

                    preds = np.concatenate(preds, 0)
                    print("preds", preds.shape)
                    scores = []
                    for i in range(10):

                        part = preds[(i * preds.shape[0] // 10):((i + 1) * preds.shape[0] // 10), :]
                        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
                        kl = np.mean(np.sum(kl, 1))
                        scores.append(np.exp(kl))

                    print("\n%d", step, "Inception Scores", str(np.mean(scores)) + "+-" + str(np.std(scores)))
                    print("Computing FID")

                    test_numbers = 10000
                    nbatches = test_numbers // bs
                    n_used_imgs = nbatches * bs
                    pred_arr = np.empty((n_used_imgs, 2048))
                    pred_arr_r = np.empty((n_used_imgs, 2048))

                    for i in range(nbatches):

                        start = i * bs
                        end = start + bs

                        #get generated samples
                        z_var = np.random.normal(0, 1, size=(bs, self.sample_size))
                        samples = sess.run(self.test_fake_images, feed_dict={self.test_z: z_var})
                        # get real samples
                        train_data_list = self.data_ob.getNextBatch2(i, bs)

                        pred = sess.run(self.pool3, {'ExpandDims:0': (samples + 1.0) * 127.5})
                        pred_arr[start:end] = pred.reshape(bs, -1)

                        # Get the activations for real sampels
                        pred2 = sess.run(self.pool3, {'ExpandDims:0': (train_data_list + 1.0) * 127.5})
                        pred_arr_r[start:end] = pred2.reshape(bs, -1)

                    g_mu = np.mean(pred_arr, axis=0)
                    g_sigma = np.cov(pred_arr, rowvar=False)

                    g_mu_r = np.mean(pred_arr_r, axis=0)
                    g_sigma_r = np.cov(pred_arr_r, rowvar=False)

                    m = np.square(g_mu - g_mu_r).sum()
                    # s = sp.linalg.sqrtm(np.dot(sigma1, sigma2)) # EDIT: commented out
                    s, _ = scipy.linalg.sqrtm(np.dot(g_sigma, g_sigma_r), disp=False)  # EDIT: added
                    dist = m + np.trace(g_sigma + g_sigma_r - 2 * s)

                    print("FID: ", np.real(dist))

                step += 1

            save_path = self.saver.save(sess, os.path.join(self.model_path, 'model_{:06d}.ckpt'.format(step)))
            print("Model saved in file: %s" % save_path)

    def discriminate(self, x_var, resnet=False, reuse=False):

        print x_var.shape
        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()
            if resnet == False:

                conv1 = lrelu(conv2d(x_var, spectural_normed=self.sn, iter=self.iter_power,
                       output_dim=64, kernel=3, stride=1, name='dis_conv1_1'))
                # conv1 = lrelu(conv2d(conv1, spectural_normed=self.sn, iter=self.iter_power,
                #        output_dim=64, name='dis_conv1_2'))
                # conv2 = lrelu(conv2d(conv1, spectural_normed=self.sn, iter=self.iter_power,
                #                      output_dim=128, k_w=3, k_h=3, d_h=1, d_w=1, name='dis_conv2_1'))
                conv2 = lrelu(conv2d(conv1, spectural_normed=self.sn, iter=self.iter_power,
                                     output_dim=128, name='dis_conv2_2'))
                # conv3 = lrelu(conv2d(conv2, spectural_normed=self.sn, iter=self.iter_power,
                #                      output_dim=256, k_h=3, k_w=3, d_w=1, d_h=1, name='dis_conv3_1'))
                conv3 = lrelu(conv2d(conv2, spectural_normed=self.sn, iter=self.iter_power,
                                     output_dim=256, name='dis_conv3_2'))
                conv4 = lrelu(conv2d(conv3, spectural_normed=self.sn, iter=self.iter_power,
                                     output_dim=512, kernel=1, name='dis_conv4'))
                conv4 = tf.reshape(conv4, [self.batch_size*self.num_rotation, -1])
                #for D
                gan_logits = fully_connect(conv4, spectural_normed=self.sn, iter=self.iter_power,
                                           output_size=1, scope='dis_fully1')
                if self.ssup:
                    rot_logits = fully_connect(conv4, spectural_normed=self.sn, iter=self.iter_power,
                                               output_size=4, scope='dis_fully2')
                    rot_prob = tf.nn.softmax(rot_logits)

            else:

                re1 = Residual_D(x_var, spectural_normed=self.sn, output_dims=128, residual_name='re1', down_sampling=True, is_start=True)
                re2 = Residual_D(re1, spectural_normed=self.sn, output_dims=128, residual_name='re2', down_sampling=True)
                re3 = Residual_D(re2, spectural_normed=self.sn, output_dims=128, residual_name='re3')
                re4 = Residual_D(re3, spectural_normed=self.sn, output_dims=128, residual_name='re4')
                re4 = tf.nn.relu(re4)
                #gsp
                gsp = tf.reduce_sum(re4, axis=[1, 2])
                gan_logits = fully_connect(gsp, spectural_normed=self.sn, iter=self.iter_power, output_size=1, scope='dis_fully1')
                if self.ssup:
                    rot_logits = fully_connect(gsp, spectural_normed=self.sn, iter=self.iter_power, output_size=4, scope='dis_fully2')
                    rot_prob = tf.nn.softmax(rot_logits)

            #tf.summary.histogram("logits", gan_logits)
            if self.ssup:
                return tf.nn.sigmoid(gan_logits), gan_logits, rot_logits, rot_prob
            else:
                return tf.nn.sigmoid(gan_logits), gan_logits

    def generate(self, z_var, batch_size=64, resnet=False, is_train=True, reuse=False):

        with tf.variable_scope('generator') as scope:

            s = 4
            if reuse:
                scope.reuse_variables()
            if self.output_size == 32:
                s = 4
            elif self.output_size == 48:
                s = 6

            d1 = fully_connect(z_var, output_size=s*s*256, scope='gen_fully1')
            d1 = tf.reshape(d1, [-1, s, s, 256])

            if resnet == False:

                d1 = tf.nn.relu(d1)
                d2 = tf.nn.relu(batch_normal(de_conv(d1, output_shape=[batch_size, s*2, s*2, 256], name='gen_deconv2')
                                             , scope='bn1', is_training=is_train))
                d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[batch_size, s*4, s*4, 128], name='gen_deconv3')
                                             , scope='bn2', is_training=is_train))
                d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[batch_size, s*8, s*8, 64], name='gen_deconv4')
                                             , scope='bn3', is_training=is_train))
                d5 = conv2d(d4, output_dim=self.channel, stride=1, kernel=3, name='gen_conv')

            else:

                d2 = Residual_G(d1, output_dims=256, up_sampling=True, residual_name='in1')
                d3 = Residual_G(d2, output_dims=256, up_sampling=True, residual_name='in2')
                d4 = Residual_G(d3, output_dims=256, up_sampling=True, residual_name='in3')
                d4 = tf.nn.relu(batch_normal(d4, scope='in4'))
                d5 = conv2d(d4, output_dim=self.channel, kernel=3, stride=1, name='gen_conv')

            return tf.tanh(d5)

    def _init_inception(self):

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
        with tf.gfile.FastGFile(os.path.join(
                MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        # Works with an arbitrary minibatch size.

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.Session(config=config, graph=tf.get_default_graph()) as sess:
            pool3 = sess.graph.get_tensor_by_name('pool_3:0')
            ops = pool3.graph.get_operations()
            for op_idx, op in enumerate(ops):
                for o in op.outputs:
                    shape = o.get_shape()
                    shape = [s.value for s in shape]
                    new_shape = []
                    for j, s in enumerate(shape):
                        if s == 1 and j == 0:
                            new_shape.append(None)
                        else:
                            new_shape.append(s)
                    o._shape = tf.TensorShape(new_shape)

            w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
            logits = tf.matmul(tf.squeeze(pool3), w)

            self.softmax = tf.nn.softmax(logits)
            self.pool3 = pool3

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

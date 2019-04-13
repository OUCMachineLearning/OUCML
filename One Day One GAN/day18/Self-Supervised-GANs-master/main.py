import tensorflow as tf
from utils import mkdir_p
from utils import Cifar, STL
from Model import SSGAN
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

flags = tf.app.flags
flags.DEFINE_integer("OPER_FLAG", 0, "flag of opertion")
flags.DEFINE_boolean("sn", True, "whether using spectural normalization")
flags.DEFINE_integer("n_dis", 1, "the number of D training for every g")
flags.DEFINE_integer("iter_power", 1, "the iteration of power")
flags.DEFINE_float("beta1", 0.0, "the beat1 for adam method")
flags.DEFINE_float("beta2", 0.9, "the beta2 for adam method")
flags.DEFINE_float("weight_rotation_loss_d", 1.0, "weight for rotation loss of D")
flags.DEFINE_float("weight_rotation_loss_g", 0.5, "weight for rotation loss for G")
flags.DEFINE_integer("num_rotation", 4, "0, 90, 180, 270")
flags.DEFINE_integer("loss_type", 3, "wgan:0; va: 1; -log(d(x)):2; 3: hinge loss")
flags.DEFINE_boolean("resnet", True, "whether using resnet architecture")
flags.DEFINE_boolean("is_adam", True, "using adam")
flags.DEFINE_boolean("ssup", False, "whether using self-supervised learning")
flags.DEFINE_integer("max_iters", 20000, "maxi iterations of networks")
flags.DEFINE_integer("batch_size", 128, "number of a batch")
flags.DEFINE_integer("sample_size", 128, "size of sample")
flags.DEFINE_float("learning_rate", 0.0002, "lr for g and d")
flags.DEFINE_integer("image_size", 32, "resolution of image; 32 for cifar")
flags.DEFINE_integer("dataset", 0, "0:cifar10; 1: stl")
flags.DEFINE_string("log_dir", "./output_w/log/", "path of log")
flags.DEFINE_string("model_path", "./output_w/model/", "path of model")
flags.DEFINE_string("sample_path", "./output_w/sample/", "path of sample")

FLAGS = flags.FLAGS
if __name__ == "__main__":

    mkdir_p([FLAGS.log_dir, FLAGS.model_path, FLAGS.sample_path])

    if FLAGS.dataset == 0:
        m_ob = Cifar(batch_size=FLAGS.batch_size)
    elif FLAGS.dataset == 1:
        m_ob = STL(batch_size=FLAGS.batch_size)

    ssgan = SSGAN(flags=FLAGS, data=m_ob)

    if FLAGS.OPER_FLAG == 0:
        ssgan._init_inception()
        ssgan.build_model_GAN()
        ssgan.train()

    if FLAGS.OPER_FLAG == 1:
        ssgan.build_model_GAN()
        ssgan.test2()

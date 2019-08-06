import os
import errno
import numpy as np
import scipy
import scipy.misc
import pickle
from scipy.ndimage.interpolation import zoom

def mkdir_p(path_list):
    for i in range(len(path_list)):
        try:
            os.makedirs(path_list[i])
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path_list[i]):
                pass
            else:
                raise

class STL(object):

    def __init__(self, batch_size):

        self.dataname = "stl10"
        #48
        self.image_size = 48
        self.dims = self.image_size*self.image_size
        self.channel = 3
        self.batchsize = batch_size
        self.shape = [self.image_size, self.image_size , self.channel]
        self.train_data_list = self.load_stl10()
        self.length = len(self.train_data_list)
        self.ro_num = self.length / self.batchsize

        print "Train_data_list", self.length

    def load_stl10(self):

        all_images = []
        f = open('./datasets/stl10_binary/unlabeled_X.bin', 'rb')
        for i in range(100000):

            #print i
            # read whole file in uint8 chunks
            image = np.fromfile(f, dtype=np.uint8, count=96*96*3)
            image = np.reshape(image, (3, 96, 96))
            image = np.transpose(image, (2, 1, 0))
            image = zoom(image, zoom=[0.5, 0.5, 1], mode='nearest')
            all_images.append(image)

        f.close()
        # with open('./datasets/stl10_binary/test_X.bin', 'rb') as f:
        #
        #     everything = np.fromfile(f, dtype=np.uint8)
        #     te_images = np.reshape(everything, (-1, 3, 96, 96))
        #     te_images = np.transpose(te_images, (0, 3, 2, 1))
        #     te_images = zoom(te_images, zoom=[1, 0.5, 0.5, 1], mode='nearest')

        #images = np.concatenate([tra_images, te_images], axis=0)
        all_images = np.array(all_images)

        return all_images / 127.5 - 1

    def getNextBatch(self, batch_num=0):

        if batch_num % self.ro_num == 0:
            perm = np.arange(self.length)
            np.random.shuffle(perm)

        return self.train_data_list[(batch_num % self.ro_num) * self.batchsize: (batch_num % self.ro_num + 1) * self.batchsize]

    def getNextBatch2(self, batch_num=0, batch_size=100):
        return self.train_data_list[(batch_num % self.ro_num) * batch_size: (batch_num % self.ro_num + 1) * batch_size]


class Cifar(object):

    def __init__(self, batch_size):

        self.dataname = "cifar10"
        self.image_size = 32
        self.dims = self.image_size*self.image_size
        self.channel = 3
        self.batchsize = batch_size
        self.shape = [self.image_size, self.image_size , self.channel]
        print "before loading cifar"
        self.train_data_list = self.load_cifar()
        print "after loading cifar"
        self.length = len(self.train_data_list)
        self.ro_num = self.length / self.batchsize

    # load cifar dataset
    def load_cifar(self):

        def load_CIFAR_batch(filename):
            """ load single batch of cifar """
            with open(filename, 'rb') as f:
                datadict = pickle.load(f)
                X = datadict['data']
                Y = datadict['labels']
                X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
                Y = np.array(Y)
            return X, Y

        data_dir = os.path.join("./datasets", self.dataname, 'cifar-10-batches-py')

        xs = []
        ys = []

        for b in range(1, 6):

            f = os.path.join(data_dir, 'data_batch_%d' % b)
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)

        trX = np.concatenate(xs)
        trY = np.concatenate(ys)

        del X, Y

        teX, teY = load_CIFAR_batch(os.path.join(data_dir, 'test_batch'))

        # convert label to one-hot
        y_vec = np.zeros((len(trY), 10), dtype=np.float32)
        for i, label in enumerate(trY):
            y_vec[i, int(trY[i])] = 1.0

        print "trX", trX.shape
        #concat between train data and test data.
        trX = np.concatenate([trX, teX])

        return trX / 127.5 - 1

    def getNextBatch(self, batch_num=0):

        if batch_num % self.ro_num == 0:
            perm = np.arange(self.length)
            np.random.shuffle(perm)

        return self.train_data_list[(batch_num % self.ro_num) * self.batchsize: (batch_num % self.ro_num + 1) * self.batchsize]

    def getNextBatch2(self, batch_num=0, batch_size=100):
        return self.train_data_list[(batch_num % self.ro_num) * batch_size: (batch_num % self.ro_num + 1) * batch_size]

class CelebA(object):

    def __init__(self, image_size):

        self.dataname = "CelebA"
        self.image_size = image_size
        self.dims = image_size*image_size
        self.channel = 3
        self.shape = [image_size, image_size, self.channel]
        self.train_data_list, self.train_lab_list = self.load_celebA()

    def load_celebA(self):

        # get the list of image path
        images_list, images_label = read_image_list_file('/home/jichao/dataset/', is_test=False)
        #images_array = self.getShapeForData(images_list)
        return images_list, images_label

    def getShapeForData(self, filenames):
        array = [get_image(batch_file, 108, is_crop=True, resize_w=self.image_size,
                           is_grayscale=False) for batch_file in filenames]

        sample_images = np.array(array)
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64):

        ro_num = len(self.train_data_list) / batch_size
        if batch_num % ro_num == 0:

            length = len(self.train_data_list)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.train_data_list = np.array(self.train_data_list)
            self.train_data_list = self.train_data_list[perm]
            self.train_lab_list = np.array(self.train_lab_list)
            self.train_lab_list = self.train_lab_list[perm]

        return self.train_data_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.train_lab_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

def get_image(image_path , image_size , is_crop=True, resize_w = 64 , is_grayscale = False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def transform(image, npx = 64 , is_crop=False, resize_w = 64):

    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w)

    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])

    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float32)
    else:
        return scipy.misc.imread(path).astype(np.float32)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image):
    return ((image + 1)* 127.5).astype('int32')

def read_image_list(category):

    filenames = []
    print("list file")
    list = os.listdir(category)
    list.sort()
    for file in list:
        if 'jpg' or 'png' in file:
            filenames.append(category + "/" + file)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames

def read_all_image_list(category):

    filenames = []
    print("list file")
    list = os.walk(category)
    for file in list:
        for img in file[2]:
            if 'jpg' in img:
                filenames.append(file[0] + "/" + img)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames


def read_image_list_file(category, is_test):

    path = ''
    skip_num = 0
    if is_test == False:
        skip_num = 5001
        path = "/home/jichao/dataset/celebA/"

    list_image = []
    list_label = []
    lines = open(category+"list_attr_celeba.txt")

    li_num = 0

    for line in lines:

        if li_num < skip_num:
            li_num += 1

            continue

        flag = line.split('1 ', 41)[15]
        file_name = line.split(' ', 1)[0]

        #add the image
        list_image.append(path + file_name)

        # print flag
        if flag == ' ':
            #one-hot
            list_label.append(1)
        else:
            list_label.append(0)

        li_num += 1

    return list_image, list_label

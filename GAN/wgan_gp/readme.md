解决了display的问题

去白边的merge

def merge(images, size):

	h, w= images.shape[1], images.shape[2]
	img = np.zeros((h * size[0], w * size[1]))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j*h:j*h+h, i*w:i*w+w] = image
	return img
	
合并多张图在一张
```python
from scipy.misc import *	
r, c = 10, 10
noise = np.random.normal(0, 1, (r * c, self.latent_dim))
gen_imgs = self.generator.predict(noise)
```

# Rescale images 0 - 1
```python
gen_imgs = 0.5 * gen_imgs + 1
gen_imgs=gen_imgs.reshape(-1,28,28)
gen_imgs = merge(gen_imgs[:49], [7,7])
imsave("images/mnist_%d.png" % epoch,gen_imgs)
```
运行python wgan.py

结果
29995 [D loss: -1.087117] [G loss: 4.016634]
29996 [D loss: -0.511691] [G loss: 3.625752]
29997 [D loss: -0.533835] [G loss: 4.005987]
29998 [D loss: -0.423012] [G loss: 3.547036]
29999 [D loss: 0.091400] [G loss: 4.133564]
存在的问题,很容易梯度爆炸,目前W-ACGAN复现失败

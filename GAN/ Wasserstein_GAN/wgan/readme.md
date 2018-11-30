## 常见的Diesplay错误

在`import matplotlib.pyplot as plt `前面加
`import matplotlib  
matplotlib.use('Agg')`  
即可
### 运行
	python wgan_mnist.py
### 结果在`/images`目录下面,

分别有`mnist`,`fashion_mnist`, <strike>CIFAR10</strike> ,三个数据集的结果  
算了算了,mnist跑出来的结果都这么烂,我就不跑cifar了  
算了还是做了吧,日后好diss沃瑟斯坦  
还是wgan_gp的结果好,
###沃瑟斯坦loss  
其实超级容易理解
沃瑟斯坦距离就是,groud_truth=-1,fake=1
loss就是预测值和groud_truth相乘

	wloss=wasserstein_loss
	valid = -np.ones((batch_size, 1))
	fake = np.ones((batch_size, 1))

	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)
顾名思义,原来SGAN的loss='binary_crossentropy'

	valid = np.ones((batch_size, 1))
	fake = np.zeros((batch_size, 1))
	d_loss_real = self.discriminator.train_on_batch(imgs, valid)
	d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
	d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

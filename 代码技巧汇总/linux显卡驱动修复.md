#  nvidia-smi报错（重装Nvidia驱动）

遇到一个莫名其妙的问题：

> NVIDIA-SMI has failed because it couldn’t communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

解决方案：重装NVIDIA驱动（非cuda）

首先在[官网](https://www.nvidia.com/Download/index.aspx?lang=cn)下载你自己显卡对应的驱动`NVIDIA-Linux-x86_64-xxx.xx.run`，拷贝到Linux某个目录后先改权限

```
chomod 777 NVIDIA-Linux-x86_64-xxx.xx.run
1
```

卸载原驱动

```
sudo apt-get remove --purge nvidia*  # 提示有残留可以接 
sudo apt autoremove
1
```

临时关闭显示服务

```
sudo service lightdm stop
1
```

运行安装程序

```
sudo ./NVIDIA-Linux-x86_64-375.66.run 

```

安装后再重启显示

```
sudo service lightdm start
```
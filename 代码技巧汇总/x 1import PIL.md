```python
import PIL.Image as Image
import os

IMAGES_PATH = './test/'  # 图片集地址
IMAGES_FORMAT = ['.jpg','.JPG']  # 图片格式
IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 10  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 10  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = './big_test'  # 图片转换后的地址
try:
    os.mkdir(IMAGE_SAVE_PATH)
except:
    pass
# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir (IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext (name)[1] == item]
image_names.sort()

print(len(image_names))
# # 简单的对于参数的设定和实际图片集的大小进行数量判断
# if len (image_names) != IMAGE_ROW * IMAGE_COLUMN:
#     raise ValueError ("合成图片的参数和要求的数量不能匹配！")


# 定义图像拼接函数

def image_compose (i):
    to_image = Image.new ('RGB',(IMAGE_COLUMN * IMAGE_SIZE,IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range (1,IMAGE_ROW + 1):
        for x in range (1,IMAGE_COLUMN + 1):
            from_image = Image.open (IMAGES_PATH + image_names[i*100+IMAGE_COLUMN * (y - 1) + x - 1]).resize (
                (IMAGE_SIZE,IMAGE_SIZE),Image.ANTIALIAS)
            to_image.paste (from_image,((x - 1) * IMAGE_SIZE,(y - 1) * IMAGE_SIZE))
    return to_image.save (IMAGE_SAVE_PATH+'/'+str(i)+'_100x.jpg')  # 保存新图


for i in range(len(image_names)//100):
    image_compose (i)  # 调用函数
```


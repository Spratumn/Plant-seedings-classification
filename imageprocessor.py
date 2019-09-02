import cv2 as cv
import numpy as np
from random import randint
class ImageProcessor:
    def __init__(self):
        pass
    def load_image(self,image_path, image_width=224, is_augment=False, rgb=False, is_normal=False):
        """
        读取图片
        :param image_path: 图片路径
        :param image_width: 读取后的图片尺寸，若源图像尺寸不符，则会在函数中进行resize
        :param is_augment: 是否进行数据增强（随机的对图片进行90°翻转、镜像）
        :param rgb:源图片颜色编码是否为RGB，若是则会在函数中改为BGR
        :param is_normal: 是否进行归一化，归一化后像素值范围由0-255整数变成0-1浮点数
        :return: 处理后的新图片
        """
        image = cv.imread(image_path)
        image = cv.resize(image, (image_width, image_width))
        if is_augment:
            image_choice = randint(1, 4)
            if image_choice == 1:
                image = rotate_image(image)
            elif image_choice == 2:
                image = rotate_image(rotate_image(image))
            elif image_choice == 3:
                image = rotate_image(rotate_image(rotate_image(image)))
            elif image_choice == 4:
                image = flip_image(image)
        if rgb:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if is_normal:
            image = image / 255
        return image
    def rotate_image(self,image):
        """
        将图片顺时针旋转90°
        :param image:
        :return: 旋转后的新图片
        """
        image_out = cv.transpose(image)
        image_out = cv.flip(image_out, 1)
        return image_out

    def flip_image(self,image):
        """
        将图片左右镜像翻转
        :param image:
        :return: 反转后的新图片
        """
        image_out = cv.flip(image, 1)
        return image_out


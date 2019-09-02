import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from imageprocessor import ImageProcessor
from keras.applications import ResNet50
from path import *
class Dataset:
    def __init__(self):
        """
        初始化

        """
        # 设置数据集路径
        self.train_path=TRAIN_DATA_PATH
        self.check_path=CHECK_DATA_PATH
        # 分类的标签类别
        self.classes=self.__get_classes()
        # 图片文件格式
        self.type_list=['.png', '.jpeg', '.jpg']
        # 图片预处理
        self.img_porcessor=ImageProcessor()
        # 训练集与测试集
        self.train_set={'names':[],'labels':[]}
        self.test_set = {'names': [], 'labels': []}
        # 使用预训练的ResNet50模型对数据集进行预处理
        resnet_path = os.path.join(TRAINED_MODEL_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
        self.model = ResNet50(weights=resnet_path, include_top=False, input_shape=(224, 224, 3))
    def __get_classes(self):
        # 从文件目录生成分类类别
        class_list=[]
        for dirs in os.listdir(self.train_path):
            class_list.append(dirs)
        return class_list
    def __get_file_name(self,father_path):
        """
        从文件夹中读取文件名（包含路径）
        :param father_path:主文件夹路径
        :param type_list: 选择的文件类型
        :return: 文件名列表
        """
        file_name_list = []
        for root, dirs, files in os.walk(father_path):
            for file in files:
                if os.path.splitext(file)[1] in self.type_list:
                    file_name_list.append(os.path.join(root,file))
        return file_name_list

    def __get_pathes_labels(self,split_rate=0.9, hot=True,random_order=True):
        """
        获取训练集与测试集
        :param split_rate:
        :param hot: 是否进行独热编码
        :param random_order：是否将数据集顺序打散
        """
        names = []
        labels = []
        for i in range(len(self.classes)):
            temp_pathes = self.__get_file_name(os.path.join(self.train_path , self.classes[i]))
            names += temp_pathes
            labels += [i] * len(temp_pathes)
        names = np.array(names)
        labels = np.array(labels)
        if hot:
            labels = OneHotEncoder(sparse=False, categories='auto')\
                    .fit(np.arange(len(self.classes)).reshape(-1,1))\
                    .transform(labels.reshape(-1,1))
        cut_count = int(len(names) * split_rate)
        if random_order:
            # 打散
            randomly_index = np.random.permutation(len(labels))
            names = names[randomly_index]
            labels = labels[randomly_index]
        self.train_set['names'] = names[0:cut_count]
        self.train_set['labels'] = labels[0:cut_count]
        self.test_set['names'] = names[cut_count:]
        self.test_set['labels'] = labels[cut_count:]

    def get_test_features(self):
        """
        获取测试的数据集，结果用于生成待提交的信息
        :return: 测试文件路径，预处理过的测试文件特征
        """
        check_names = self.__get_file_name(self.check_path)
        check_names = np.array(check_names)
        check_images = [self.img_porcessor.load_image(image_name) for image_name in check_names]
        check_images = np.array(check_images).reshape(-1, 224, 224, 3)
        check_features = self.model.predict(check_images)
        check_features = np.reshape(check_features, (-1, 7 * 7 * 2048))
        return check_names,check_features

    def get_feature_set(self):
        """
        获取预处理过的训练集
        :return: 训练集特征，训练集标签，验证集特征，验证集标签
        """
        self.__get_pathes_labels()
        train_images = [self.img_porcessor.load_image(image_name) for image_name in self.train_set['names']]
        train_images = np.array(train_images).reshape(-1, 224, 224, 3)
        train_feature = self.model.predict(train_images)

        test_images = [self.img_porcessor.load_image(image_name) for image_name in self.test_set['names']]
        test_images = np.array(test_images).reshape(-1, 224, 224, 3)
        test_feature = self.model.predict(test_images)

        train_feature = np.reshape(train_feature, (-1, 7 * 7 * 2048))
        test_feature = np.reshape(test_feature, (-1, 7 * 7 * 2048))

        return train_feature,self.train_set['labels'],test_feature,self.test_set['labels']
    def get_batch_feature(self,is_train=True,batch_size=32):
        """
        生成一个训练批次的数据，用于分批次训练
        :param is_train: 是否生成训练集，否则生成验证集
        :param batch_size: 批次大小
        :return: 一个训练批次的数据（特征，标签）
        """
        self.__get_pathes_labels(random_order=False)
        if is_train:
            set_count = len(self.train_set['labels'])
            names = self.train_set['names']
            labels = self.train_set['labels']
        else:
            set_count = len(self.test_set['labels'])
            names = self.test_set['names']
            labels = self.test_set['labels']

        indx = np.random.choice(set_count, size=batch_size)
        batch_names = names[indx]
        batch_images = [self.img_porcessor.load_image(image_name) for image_name in batch_names]
        batch_images = np.array(batch_images).reshape(-1, 224, 224, 3)
        batch_feature = self.model.predict(batch_images)
        batch_feature = np.reshape(batch_feature, (-1, 7 * 7 * 2048))
        batch_label = labels[indx]
        return batch_feature, batch_label



if __name__=='__main__':
    ds=Dataset()
    ds.model.summary()
    print(ds.train_set['names'][0:5])
    print(ds.train_set['labels'][0:5])


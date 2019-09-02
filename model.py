from dataset import *
from keras import models
from keras import layers
from keras import optimizers

class MyModel:
    def __init__(self):
        self.batch_size=0
        self.md=models.Sequential()
        self.md.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 2048))
        self.md.add(layers.Dropout(0.5))
        self.md.add(layers.Dense(12, activation='softmax'))
        self.md.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
    def train(self,dataset,batch_size=0,epoch=10):
        """
        训练
        :param dataset: 一个Dataset实例
        :param batch_size: 控制训练批次（为0时不分批训练）
        :param epoch: 训练次数
        :return:
        """
        self.batch_size=batch_size
        if self.batch_size==0:
            train_features, train_labels, test_features, test_labels = dataset.get_feature_set()
            history = self.md.fit(train_features, train_labels,
                                epochs=epoch,
                                batch_size=self.batch_size,
                                validation_data=(test_features, test_labels))

        else:
            for i in range(epoch):
                batch_train_features,batch_train_labels=dataset.get_batch_feature(is_train=True,batch_size=self.batch_size)
                batch_check_features, batch_check_labels = dataset.get_batch_feature(is_train=False, batch_size=self.batch_size)
                train_history=self.md.train_on_batch(batch_train_features,batch_train_labels)
                check_history = self.md.test_on_batch(batch_check_features, batch_check_labels)

                train_acc = train_history[1]
                train_loss = train_history[0]
                check_acc = check_history[1]
                check_loss = check_history[0]
                print('------------------------------------------------------------------')
                print('epoch:',i)
                print('train_acc:%.2f,train_loss:%.2f<---->check_acc:%.2f,check_loss:%.2f'
                      %(train_acc,train_loss,check_acc,check_loss))

    def get_test_names_labels(self,dataset):
        """
        获取测试集的预测结果
        :param dataset: 一个Dataset实例
        :return: 测试集文件名，测试集预测标签
        """
        classes=np.array(dataset.classes)
        test_names,test_features = dataset.get_test_features()
        test_image_names = [file_name.split('\\')[-1] for file_name in test_names]
        test_labels = self.md.predict(test_features)
        prediction_index = np.argmax(test_labels, axis=1)
        test_labels = classes[prediction_index]
        return test_image_names,test_labels















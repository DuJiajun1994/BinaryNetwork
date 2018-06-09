import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

image_size = 32
num_labels = 10

def display_data():
    print ('loading Matlab data...')
    train = sio.loadmat('../data/svhn/train_32x32.mat')
    data=train['X']
    label=train['y']
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.title(label[i][0])
        plt.imshow(data[...,i])
        plt.axis('off')
    plt.show()

def load_data(one_hot = False):
    
    train = sio.loadmat('../data/svhn/train_32x32.mat')
    test = sio.loadmat('../data/svhn/test_32x32.mat')
    
    train_data=train['X']
    train_label=train['y']
    test_data=test['X']
    test_label=test['y']
    
    for i in range(train_label.shape[0]):
         if train_label[i][0] == 10:
             train_label[i][0] = 0
                        
    for i in range(test_label.shape[0]):
         if test_label[i][0] == 10:
             test_label[i][0] = 0

    if one_hot:
        train_label = (np.arange(num_labels) == train_label[:,]).astype(np.float32)
        test_label = (np.arange(num_labels) == test_label[:,]).astype(np.float32)

    return train_data,train_label, test_data,test_label

if __name__ == '__main__':
    train_data,train_label, test_data,test_label=load_data()
    print(train_data.shape,train_label.shape,test_data.shape,test_label.shape)
    
    train = []
    for i in range(train_data.shape[3]):
        sample = np.transpose(train_data[:,:,:,i].reshape(1024,3)).reshape(1,3072)
        train.append(sample)
    data = np.hstack(train)
   
    print(data.shape)
   
    data.tofile('../data/svhn/test_data.bin')
    
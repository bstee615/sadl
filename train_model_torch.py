import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

CLIP_MIN = -0.5
CLIP_MAX = 0.5


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_torch(args):
    """
    Add image classifier tutorial from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    # TODO: Change hyperparameters to those in original project

    import torch
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    from torch import nn
    import torch.nn.functional as F
    import torch.optim as optim

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')  # Assuming that we are on a CUDA machine, this should print a CUDA device:

    if args.d == "mnist":
        # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
        # https://github.com/pytorch/examples/blob/master/mnist/main.py
        transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize((0.1307,), (0.3081,))
             # This causes abysmal performance (recognizes all digits as 2). TODO why?
             transforms.Normalize((0.5,), (0.5,))
             ])

        classes = list(range(10))
        dataset = torchvision.datasets.MNIST
        conv_flat = 16 * 4 * 4

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(conv_flat, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, conv_flat)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    elif args.d == "cifar":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        dataset = torchvision.datasets.CIFAR10
        # raise NotImplementedError()
    else:
        raise NotImplementedError()

    trainset = dataset(root='./data', train=True,
                       download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = dataset(root='./data', train=False,
                      download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    if args.debug:
        # get some random training images
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    PATH = './model/mnist_net.pth'

    net = Net()
    if os.path.exists(PATH) and not args.force:
        print(f'Loading already trained model found at {PATH}')
        net.load_state_dict(torch.load(PATH))
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        torch.save(net.state_dict(), PATH)
        print(f'Saved to {PATH}')

    if args.test:
        if args.debug:
            dataiter = iter(testloader)
            images, labels = next(dataiter)

            # print images
            imshow(torchvision.utils.make_grid(images))
            output = net.forward(images)
            _, pred = torch.max(output, 1)
            print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
            print('Prediction:  ', ' '.join('%5s' % classes[pred[j]] for j in range(4)))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net.forward(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

        correct_by_class = defaultdict(int)
        count_by_class = defaultdict(int)
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net.forward(images)
                _, pred = torch.max(outputs.data, 1)
                for i in range(images.shape[0]):
                    my_class = labels[i].item()
                    my_pred = pred[i].item()
                    count_by_class[my_class] += 1
                    correct_by_class[my_class] += 1 if (my_pred == my_class) else 0
        for i in sorted(correct_by_class.keys()):
            print(f'Accuracy of the network on class {i}: {100 * correct_by_class[i] / count_by_class[i]}')


def train_keras(args):
    import tensorflow as tf
    from keras.datasets import mnist, cifar10
    from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
    from keras.models import Sequential
    from keras.regularizers import l2
    from keras.utils import np_utils
    from tensorflow_core.python.client import device_lib

    print('GPU available:', tf.test.is_gpu_available(), 'CUDA available:', tf.test.is_built_with_cuda())
    print('devices:', device_lib.list_local_devices())

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        layers = [
            Conv2D(64, (3, 3), padding="valid", input_shape=(28, 28, 1)),
            Activation("relu"),
            Conv2D(64, (3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(128),
            Activation("relu"),
            Dropout(0.5),
            Dense(10),
        ]

    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        layers = [
            Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)),
            Activation("relu"),
            Conv2D(32, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding="same"),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation("relu"),
            Dropout(0.5),
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation("relu"),
            Dropout(0.5),
            Dense(10),
        ]

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.add(Activation("softmax"))

    print(model.summary())
    model.compile(
        loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=128,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    model.save("./model/model_{}.h5".format(args.d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--force", action='store_true')
    cmd_args = parser.parse_args()
    assert cmd_args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    train_torch(cmd_args)

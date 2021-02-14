import unittest

import numpy as np
import torch
from parameterized import parameterized_class
from tensorflow.python.keras.models import load_model

from sa import get_layer_outputs
from train_model import MNISTNet

layer_names = ['activation_3']
dataset_size = 23
keras_dataset = [np.random.rand(dataset_size, 28, 28, 1)]  # Mock Keras MNIST dataset
torch_dataset = [(torch.rand((dataset_size, 1, 28, 28)), torch.rand((dataset_size,)))]  # Mock torch MNIST dataset


@parameterized_class(('model', 'dataset'), [
    (MNISTNet().cuda(), torch_dataset),
    (load_model("model/model_mnist.h5"), keras_dataset),
])
class TestLayerOutputs(unittest.TestCase):
    def test_sanity(self):
        get_layer_outputs(self.model, layer_names, self.dataset, dataset_size)

    def test_result_lengths(self):
        layer_outputs, pred = get_layer_outputs(self.model, layer_names, self.dataset, dataset_size)
        self.assertSequenceEqual(pred.shape, (dataset_size,))
        self.assertSequenceEqual(layer_outputs[0].shape, (dataset_size, 128))

    def test_layer_output_counts(self):
        layer_outputs, pred = get_layer_outputs(self.model, layer_names, self.dataset, dataset_size)
        self.assertEqual(len(layer_outputs), len(layer_names))


if __name__ == '__main__':
    unittest.main()

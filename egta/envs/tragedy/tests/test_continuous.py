import torch

from ..env import Continuous

def test_sampling():
    test_low_high = [(0, 1), (-2, 1), (-23, 61), (200, 200.1)]
    test_shapes = [(1,), (2,), (2,10), (100, 100)]

    for shape in test_shapes:
        for low, high in test_low_high:
            temp_space = Continuous(low, high)
            sample = temp_space.sample(shape)

            # test that the sample shape is correct
            assert tuple(sample.shape) == shape

            # test that the sample values are within the correct range
            assert torch.all(sample <= high) # according to the pytorch docs, this should be exclusive, however, I don't care about that here
            assert torch.all(low <= sample)

# def test_device
# def test_dtype

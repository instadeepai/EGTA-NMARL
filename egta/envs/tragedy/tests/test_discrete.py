import torch

from ..env import Discrete

def test_num_options():
    test_num_option_too_small()
    test_num_options_valid()

def test_num_option_too_small():
    test_num_options = [-100, -1, 0, 1]

    for num_options in test_num_options:
        try:
            temp_space = Discrete(num_options)
            assert False, "The `Discrete` class should raise an error here as these values are too small"
        except AssertionError:
            # this should trigger
            assert True

def test_num_options_valid():
    test_num_options = [2, 3, 4, 8, 1024]

    for num_options in test_num_options:
        temp_space = Discrete(num_options)
        # if there is no error up to this point, this test passes
        assert True

def test_sampling():
    test_num_options = [2, 3, 4, 8, 1024]
    test_shapes = [(1,), (2,), (2,10), (100, 100)]

    for shape in test_shapes:
        for num_options in test_num_options:
            temp_space = Discrete(num_options)
            sample = temp_space.sample(shape)

            # test that the sample shape is correct
            assert tuple(sample.shape) == shape

            # test that the sample values are within the correct range
            assert torch.all(sample < num_options)
            assert torch.all(0 <= sample)

# def test_device
# def test_dtype

import torch

from ..env import TensorConditioner

def test_device():
    cpu()

    if torch.cuda.is_available():
        gpu()

    torch_device()
    automatic_device()
    device_exception()

def cpu():
    tc = TensorConditioner(device="cpu")
    assert str(tc._device) == "cpu"
    assert isinstance(tc._device, torch.device)

def gpu(): # this will not run on my laptop as I have no GPU
    tc = TensorConditioner(device="cuda")
    assert str(tc._device) == "cuda"
    assert isinstance(tc._device, torch.device)

def torch_device():
    torch_device_cpu()

    if torch.cuda.is_available():
        torch_device_gpu()

def torch_device_cpu():
    tc = TensorConditioner(device=torch.device("cpu"))
    assert str(tc._device) == "cpu"
    assert isinstance(tc._device, torch.device)

def torch_device_gpu():
    tc = TensorConditioner(device=torch.device("cuda"))
    assert str(tc._device) == "cuda"
    assert isinstance(tc._device, torch.device)

def automatic_device():
    tc = TensorConditioner()

    if torch.cuda.is_available():
        assert str(tc._device) == "cuda"
    else:
        assert str(tc._device) == "cpu"

    assert isinstance(tc._device, torch.device)

def device_exception():
    invalid_device_string()
    invalid_device_arg_type()

def invalid_device_string():
    try:
        tc = TensorConditioner(device="blah")
        assert False, "'TensorConditioner' should raise an error here"
    except RuntimeError:
        # this should trigger
        assert True

def invalid_device_arg_type():
    try:
        tc = TensorConditioner(device=1)
        assert False, "'TensorConditioner' should raise an error here"
    except TypeError:
        # this should trigger
        assert True

def test_dtype():
    types = [torch.long, torch.float32, torch.float16]

    for t in types:
        tc = TensorConditioner(dtype=t)
        assert tc._dtype == t

def test_generic_tensor():
    types = [torch.long, torch.float32]

    # test half precision if gpu is available (half precision not available for cpu)
    if torch.cuda.is_available():
        types.append(torch.float16)

    data = [[1, 2, 3], 1, [[2, 59, 67], [9, 400, 82]]]

    for t in types:
        tc = TensorConditioner(dtype=t)

        for d in data:
            temp = tc.get_tensor(d)

            # check that the tensor is of the correct type
            assert temp.dtype == t

            # check that the tensor is on the correct device
            assert temp.device == tc._device == torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # check that the gradient settings are correct
            assert temp.requires_grad == tc._requires_grad

            # check that that data inside the tensor is correct
            # assert torch.equal(temp, torch.tensor(d))
            assert torch.all(torch.eq(temp, torch.tensor(d)))

def test_convert_tensor():
    original = torch.tensor([1, 2, 3])

    types = [torch.long, torch.float32]#, torch.float64, torch.uint8, torch.int8, torch.bool]

    # test half precision if gpu is available (half precision not available for cpu)
    if torch.cuda.is_available():
        types.append(torch.float16)

    for t in types:
        tc = TensorConditioner(dtype=t)
        temp = tc.convert_tensor(original)

        # check that the tensor is of the correct type
        assert temp.dtype == t

        # check that the tensor is on the correct device
        assert temp.device == tc._device == torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # check that the gradient settings are correct
        assert temp.requires_grad == tc._requires_grad

        # check that that data inside the tensor is correct
        # assert torch.equal(temp, torch.tensor(d))
        assert torch.all(torch.eq(temp, original))

def test_lossy_convert_tensor():
    original = torch.tensor([1.3, 5e-3, 3.9])

    t = torch.long
    tc = TensorConditioner(dtype=t)
    temp = tc.convert_tensor(original)

    # check that the tensor is of the correct type
    assert temp.dtype == t

    # check that the tensor is on the correct device
    assert temp.device == tc._device == torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check that the gradient settings are correct
    assert temp.requires_grad == tc._requires_grad

    # check that that data inside the tensor has been cast to integers (not rounded!)
    # assert torch.equal(temp, torch.tensor(d))
    assert not torch.all(torch.eq(temp, original))

def test_ones():
    shapes = [1, (1,), 10, (10,), (10, 10), (69, 420, 42)]

    tc = TensorConditioner(dtype=torch.float32)

    for shape in shapes:
        temp = tc.ones(shape)

        # check that the tensor is of the correct type
        assert temp.dtype == torch.float32

        # check that the tensor is on the correct device
        assert temp.device == tc._device == torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # check that the gradient settings are correct
        assert temp.requires_grad == tc._requires_grad

        # check that that data inside the tensor is correct
        # assert torch.equal(temp, torch.tensor(d))
        assert torch.all(torch.eq(temp, torch.tensor(1)))

        if isinstance(shape, tuple):
            assert tuple(temp.shape) == shape
        elif isinstance(shape, int):
            assert tuple(temp.shape) == (shape,)
        else:
            raise TypeError("Invalid shape type in 'test_ones' test")


# def test_zeros():
#     pass

# def test_range():
#     pass

# def test_grad

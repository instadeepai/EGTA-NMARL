import torch

from ..env import Tragedy

def test_defaults():
    env = Tragedy(regen_properties={"type":"constant", "rate": 0.075})
    obs = env.reset()
    print("start", obs)
    new_obs, reward, done, _ = env.step(torch.tensor([0,]*4))
    print("obs", new_obs)
    print("reward", reward)
    print("done", done)
    new_obs, reward, done, _ = env.step(torch.tensor([1,]*4))
    print("obs", new_obs)
    print("reward", reward)
    print("done", done)
    # new_obs, reward, done, _ = env.step(torch.tensor([2,]*6))
    # print("obs", new_obs)
    # print("reward", reward)
    # print("done", done)

    for _ in range(2):
        new_obs, reward, done, _ = env.step(torch.tensor([0,] + [1,]*3))
        print("obs", new_obs)
        print("reward", reward)
        print("done", done)

    # for _ in range(2):
    #     new_obs, reward, done, _ = env.step(torch.tensor([2,] + [1,]*5))
    #     print("obs", new_obs)
    #     print("reward", reward)
    #     print("done", done)

    for _ in range(2):
        new_obs, reward, done, _ = env.step(torch.tensor([1, 0] + [1,]*2))
        print("obs", new_obs)
        print("reward", reward)
        print("done", done)

    for _ in range(2):
        new_obs, reward, done, _ = env.step(torch.tensor([[0,]*3 + [1,]]))
        print("obs", new_obs)
        print("reward", reward)
        print("done", done)

    # for _ in range(2):
    #     new_obs, reward, done, _ = env.step(torch.tensor([1, 2,] + [1,]*4))
    #     print("obs", new_obs)
    #     print("reward", reward)
    #     print("done", done)

    assert False

# def test_defaults_batch_env():
#     env = Tragedy()
#     obs = env.reset()
#     print("start", obs)
#     new_obs, reward, done, _ = env.step(torch.tensor([[0.3, 0.3, 0.2], [0.3, 0.3, 0.4]]))
#     print("obs", new_obs)
#     print("reward", reward)
#     print("done", done)
#     new_obs, reward, done, _ = env.step(torch.tensor([[0.3, 0.3, 0.3], [0.1, 0.1, 0.2]]))
#     print("obs", new_obs)
#     print("reward", reward)
#     print("done", done)
#     new_obs, reward, done, _ = env.step(torch.tensor([[0.33, 0.33, 0.35], [0.34, 0.33, 0.32]]))
#     print("obs", new_obs)
#     print("reward", reward)
#     print("done", done)
#     assert False

# def test_discrete_binary_batch_env():
#     env = Tragedy(num_agents=3, batch_size=2, action_space="discrete")
#     obs = env.reset()
#     print("start", obs)
#     new_obs, reward, done, _ = env.step(torch.tensor([[0, 1, 0], [0, 0, 0]]))
#     print("obs", new_obs)
#     print("reward", reward)
#     print("done", done)
#     new_obs, reward, done, _ = env.step(torch.tensor([[1, 0, 1], [1, 1, 1]]))
#     print("obs", new_obs)
#     print("reward", reward)
#     print("done", done)
#     new_obs, reward, done, _ = env.step(torch.tensor([[1, 1, 0], [0, 0, 0]]))
#     print("obs", new_obs)
#     print("reward", reward)
#     print("done", done)
#     assert False

# def test_discrete_trinary_batch_env():
#     env = Tragedy(num_agents=3, batch_size=2, action_space="discrete")
#     obs = env.reset()
#     print("start", obs)

#     for _ in range(6):
#         new_obs, reward, done, _ = env.step(torch.tensor([[0, 0, 0], [0, 0, 0]]))
#         print("obs", new_obs)
#         print("reward", reward)
#         print("done", done)

#     for _ in range(20):
#         new_obs, reward, done, _ = env.step(torch.tensor([[2, 2, 2], [2, 2, 2]]))
#         print("obs", new_obs)
#         print("reward", reward)
#         print("done", done)

#     assert False
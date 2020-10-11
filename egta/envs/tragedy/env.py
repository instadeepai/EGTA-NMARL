import torch#, gym
from torch.multiprocessing import Pool
from collections import OrderedDict

class TensorConditioner(): # NB, this does not allow for dynamic type, device, etc changes! This is possible but the code will need to be changed!
    def __init__(self, device=None, dtype=None, requires_grad=False):
        self._initialised = False

        self.set_device(device)
        self.set_dtype(dtype)
        self.set_grad(requires_grad)

        self._initialised = True

        self._update_kwargs()

    def _update_kwargs(self):
        if self._initialised:
            self.set_tensor_kwargs()
            self.set_convert_kwargs()

    def set_dtype(self, dtype):
        if dtype:
            self._dtype = dtype
        else:
            self._dtype = torch.float16 if "cuda" in str(self._device) else torch.float32

        self._update_kwargs()

    def set_device(self, device):
        if device:
            if isinstance(device, torch.device):
                self._device = device
            elif isinstance(device, str):
                self._device = torch.device(device)
            else:
                raise TypeError("The 'TensorConditioner' class needs either a 'torch.device' type or a 'string' type for the 'device' argument.")

        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._update_kwargs()

    def set_grad(self, requires_grad):
        self._requires_grad = requires_grad

        self._update_kwargs()

    def set_tensor_kwargs(self):
        self._tensor_kwargs = {
            "dtype": self._dtype,
            "device": self._device,
            "requires_grad": self._requires_grad
        }

    def set_convert_kwargs(self):
        self._convert_kwargs = {
            "dtype": self._dtype,
            "device": self._device
        }

    def get_tensor(self, data):
        return torch.tensor(data, **self._tensor_kwargs)

    def convert_tensor(self, tensor):
        new_tensor = tensor.to(**self._convert_kwargs)
        new_tensor.requires_grad = self._requires_grad
        return new_tensor

    def ones(self, shape):
        return torch.ones(shape, **self._tensor_kwargs)

    def zeros(self, shape):
        return torch.zeros(shape, **self._tensor_kwargs)

    def arange(self, *args, **kwargs):
        return torch.arange(*args, **kwargs, **self._tensor_kwargs)

    def empty(self, shape):
        return torch.empty(shape, **self._tensor_kwargs)

class Discrete(TensorConditioner):
    def __init__(self, num_options, device=None):
        assert num_options > 1, "'num_options' must be bigger than 1"

        super(Discrete, self).__init__(device=device)

        self._dist = torch.distributions.categorical.Categorical(
            probs=self.ones(num_options)
        )

    def sample(self, shape=(1,)):
        return self._dist.sample(shape)

class Continuous(TensorConditioner):
    def __init__(self, low=0.0, high=1.0, device=None):
        super(Continuous, self).__init__(device=device)

        self._dist = torch.distributions.uniform.Uniform(
            self.get_tensor(low), self.get_tensor(high)
        )

    def sample(self, shape=(1,)):
        return self._dist.sample(shape)

# class Box(low, high, shape, dtype):
#     pass

class Tragedy(TensorConditioner):
# class Tragedy(gym.Env):

    WAIT_ACTION = 0
    TAKE_ACTION = 1
    # TAG_ACTION = 2
    DEFAULT_CONSTANT_REGEN_RATE = 0.1
    DEFAULT_INITIAL_RESOURCE = 0.5
    # DEFAULT_TAG_TURNS = 3

    def __init__(
        self, start_resource=DEFAULT_INITIAL_RESOURCE, batch_size=1,
        regen_properties={"type":"constant", "rate": DEFAULT_CONSTANT_REGEN_RATE},
        num_agents=4, device=None, action_space="discrete", #tag_turns=DEFAULT_TAG_TURNS
    ):
        super(Tragedy, self).__init__(device=device)

        self.set_start_resource(start_resource)
        self.set_num_agents(num_agents)
        # self.set_tag_turns(tag_turns)
        self.set_batch_size(batch_size)

        # self.observation_space = gym.spaces.Box(
        #     low=0.0, high=1.0,
        #     shape=(1,),
        #     dtype=torch.float16
        # )

        # set up regeneration settings for the shared resource
        self._regen_rate = regen_properties["rate"]

        if regen_properties["type"] == "constant":
            self._regen = self._constant_regen

            if not (0 < self._regen_rate < 1):
                raise ValueError("In the 'constant' regen mode, the 'rate' must follow: 0 < rate < 1.")

        elif regen_properties["type"] == "linear":
            self._regen = self._linear_regen

            if self._regen_rate <= 1:
                raise ValueError("In the 'linear' regen mode, the 'rate' must follow: 1 < rate.")

        else:
            raise ValueError("The value corresponding to the 'type' key in the 'regen_properties' must be either 'constant' or 'linear'.")

        # set up action spaces
        if action_space == "continuous":
            raise NotImplementedError("The continuous action space version is out of date.")
            self.action_space = Continuous(0.0, 1.0)
            self._discrete_action_space = False
            # self.action_space = gym.spaces.
        elif action_space == "discrete":
        # elif isinstance(action_space, int):
            self.action_space = Discrete(2)
            self._discrete_action_space = True
            # self.action_space = gym.spaces.Discrete(action_space)
        else:
            raise TypeError("Invalid type for 'action_space'.")

        # set up how to map between integer action labels to amount of shared resource taken
        if self._discrete_action_space:
            if self._num_agents:
                self._calculate_action_set()
                self._map_actions = self._static_discrete_map_actions
            else:
                self._map_actions = self._dynamic_discrete_map_actions
        else:
            self._map_actions = self._continuous_map_actions

        # set up the initial state
        self._dones = self.ones((self._batch_size,)).to(dtype=torch.bool)
        self.reset(done_only=False)

    def set_start_resource(self, start_resource):
        self._start_resource = start_resource

    def set_num_agents(self, num_agents):
        self._num_agents = num_agents

    # def set_tag_turns(self, tag_turns):
    #     self._tag_turns = tag_turns

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def reset(self, done_only=True):
        if done_only:
            self._resource[self._dones] = self._start_resource
            self._total_taken[self._dones] = 0.0
            # self._tag_counters[self._dones] = 0.0
        else:
            self._resource = self.get_tensor([self._start_resource,]*self._batch_size).reshape((self._batch_size, 1))
            self._total_taken = self.zeros((self._batch_size, self._num_agents))
            # self._tag_counters = self.zeros((self._batch_size, self._num_agents))

        return self._gen_agent_observations()

    def _constant_regen(self):
        return self._resource + (self._regen_rate * self._not_dones)

    def _linear_regen(self):
        return self._resource * (self._regen_rate - ((self._regen_rate - 1) * self._dones))

    def _regen(self):
        raise RuntimeError("The regen function has not been set in the 'Tragedy' class constructor.")

    # def _get_tags(self, taggers):
    #     # generate a random agent to tag for each tagger
    #     # agents should not be allowed to tag themselves
    #     # (agents should prefer to tag agents that are not already tagged?)

    #     ### Non vectorised solution for now... gross but not enough time
    #     # multi-process over rows
    #     taggies = self.zeros(taggers.shape).to(torch.bool)

    #     for row, batch in enumerate(taggers):
    #         for agent, action in enumerate(batch):
    #             if action: # True if tag action
    #                 # create a set of agent indices to sample from
    #                 other_agents = self.arange(self._num_agents).to(torch.long)

    #                 # remove this agent's index
    #                 other_agents = other_agents[other_agents != agent]

    #                 # remove agents that are already tagged?

    #                 # # sample
    #                 # tagged_agent = other_agents[torch.randint(0, self._num_agents - 1, (1,))].to(torch.long)
    #                 # taggies[row, tagged_agent] = True

    #                 # tag all other agents
    #                 taggies[row, other_agents] = True

    #     return taggies

    #     # slightly better solution:
    #     # (can use the torch.multiprocessing.Pool module to help with speed
    #     # because this isn't fully vectorised either)
    #     # create N multinomial distributions (one for each agent)
    #     # that each agent can sample from such that it's own index is not in the sample space
    #     # then sample one batch from each of those distributions (sample(shape=(taggers.size(0),)))
    #     # then stack those horizontally (we don't need to convert to a boolean mask, indices are fine)
    #     # distribution = self.ones(self._num_agents)
    #     # distribution[agent] = 0
    #     # return torch.multinomial(distribution, num_samples=taggers.size(0), replacement=True).reshape((-1, 1))
    #     # samples = multiprocess
    #     # NOT THIS ONE torch.stack([samples], dim=-1) * taggers (mask out samples that were generated for agents that didn't chose the tag action)
        # torch.stack([samples], dim=-1)[taggers] (mask out samples that were generated for agents that didn't chose the tag action)

    # def _get_tagged(self):
    #     return self._tag_counters > 0.0

    def step(self, agent_outputs):
        with torch.no_grad(): # probably not necessary
            # # get "tagged" agents mask (these agents may not tag other agents)
            # tagged_mask = self._get_tagged()
            # not_tagged_mask = ~tagged_mask

            # # identify the agents that performed the "tag" action (that are not themselves, tagged)
            # taggers = (agent_outputs == Tragedy.TAG_ACTION) * not_tagged_mask

            # # randomly select an opposing agent to "tag"
            # taggies = self._get_tags(taggers)

            # # set taggies as tagged
            # self._tag_counters[taggies] = self._tag_turns

            # # update "tagged" agents mask (these agents may not take from the shared resource)
            # tagged_mask = self._get_tagged()

            # get values that agents want to take from the shared-resource
            actions = self._process_agent_outputs(agent_outputs)
            # actions = self._process_agent_outputs(agent_outputs, tagged_mask)

            # let the un-"tagged" agents take from the shared resource
            self._resource = self._resource - actions.sum(-1)

            # check if the episode is done (when the shared resource depletes)
            dones = self._resource <= 0
            not_dones = ~dones

            # calculate the rewards
            rewards = not_dones * actions
            self._total_taken += rewards
            # rewards -= 1.25 * taggers * self._action_set[Tragedy.TAKE_ACTION]

            self._dones = dones
            self._not_dones = not_dones
            self._resource = self._regen()

            # Set a cap on the amount of shared resource that can be 'stock piled' (at 1.0)
            self._resource = torch.min(self._resource, self.ones(self._resource.shape))

            # # reduce the "tag counters"
            # self._tag_counters -= 1

            # # add a lower bound on the "tag counters" of 0.0
            # self._tag_counters = torch.max(self._tag_counters, self.zeros(self._tag_counters.shape))

            # # results[dones == False] = self._start_state
            # # self._state = torch.stack([results, not_dones * self._start_state], dim=-1)
            observations = self._gen_agent_observations()

        return observations, rewards, dones, {}

    def _gen_agent_observations(self):
        batches = []

        for i, taken in enumerate(self._total_taken):
        # for i, counters in enumerate(self._tag_counters / self._tag_turns):
            batches.append(torch.cartesian_prod(self._resource[i], taken).reshape((1, self._num_agents, 2)))
            # batches.append(torch.cartesian_prod(self._resource[i], counters).reshape((1, self._num_agents, 2)))

        return torch.cat(batches, dim=0)

        # return torch.cartesian_prod(self._resource, self._tag_counters.reshape((-1,))).reshape((-1, self._num_agents, 2))

    def _ensure_action_shape(self, agent_outputs):
        if len(agent_outputs.shape) < 2:
            agent_outputs = agent_outputs.unsqueeze(0)

        return agent_outputs

    def _calculate_action_set(self):
        # trinary action space

        action_none = 0
        action_min = Tragedy.DEFAULT_CONSTANT_REGEN_RATE / self._num_agents

        # action_min = self._regen_rate / (self._num_agents + 1.1)
        # action_max = 2 * action_min

        action_map = OrderedDict({
            Tragedy.WAIT_ACTION: action_none,
            Tragedy.TAKE_ACTION: action_min,
            # Tragedy.TAG_ACTION: action_none,
        })

        action_set = self.empty((len(action_map),))

        for index, value in action_map.items():
            action_set[index] = value

        # action_set = self.get_tensor([action_none, action_min, action_none])
        # action_set = self.get_tensor([action_none, action_min, action_max])
        # action_set = self.get_tensor([action_min, action_max])

        self._action_set = action_set

    def _dynamic_discrete_map_actions(self, agent_outputs):
    # def _dynamic_discrete_map_actions(self, agent_outputs, tagged_agents):
        self._num_agents = action_indices.shape[1]
        self._calculate_action_set()
        return self._static_discrete_map_actions(agent_outputs)
        # return self._static_discrete_map_actions(agent_outputs, tagged_agents)

    def _static_discrete_map_actions(self, agent_outputs):
    # def _static_discrete_map_actions(self, agent_outputs, tagged_agents):
        # not_tagged_agents = ~tagged_agents
        # zero out the take actions that the "tagged" agents would have made
        return self._action_set[agent_outputs]# * not_tagged_agents

    def _continuous_map_actions(self, agent_outputs):
        # regen_prediction = self._regen() - self._state
        # max_action = 2 * regen_prediction / num_agents
        # scaled_actions = agent_outputs * max_action
        # return scaled_actions
        return agent_outputs

    def _process_agent_outputs(self, agent_outputs):
    # def _process_agent_outputs(self, agent_outputs, tagged_agents):
        agent_outputs = self._ensure_action_shape(agent_outputs)
        return self._map_actions(agent_outputs)
        # return self._map_actions(agent_outputs, tagged_agents)

    def _map_actions(self, agent_outputs):
        raise RuntimeError("The action mapping function has not been set in the 'Tragedy' class constructor.")

import numpy as np
import torch

import utils


class ReplayBuffer:
    """ Replay buffer for storing transitions. """

    def __init__(self, observation_space, action_space, capacity, start_o, device):
        self.observation_space = observation_space
        self.action_space = action_space

        self.observations = utils.space_zeros(observation_space, capacity, device=device)
        self.actions = utils.space_zeros(action_space, capacity, device=device)
        self.next_rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_terms = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_truncs = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_observations = utils.space_zeros(observation_space, capacity, device=device)

        self.observations[0] = torch.as_tensor(np.array(start_o)).to(device=device, non_blocking=True)
        self.len = 0

        self.episode_reward = 0.0
        self.episode_rewards = []

    def to(self, device):
        if device == self.observations.device:
            return self

        self.observations = self.observations.to(device, non_blocking=True)
        self.actions = self.actions.to(device, non_blocking=True)
        self.next_rewards = self.next_rewards.to(device, non_blocking=True)
        self.next_terms = self.next_terms.to(device, non_blocking=True)
        self.next_truncs = self.next_truncs.to(device, non_blocking=True)
        self.next_observations = self.next_observations.to(device, non_blocking=True)

        return self

    @property
    def device(self):
        return self.observations.device

    @property
    def capacity(self):
        return self.observations.shape[0]

    @property
    def cont_o(self):
        """ Get the last observation, can be used to select the next action. """
        return self.observations[self.len].unsqueeze(0)

    def __len__(self):
        return self.len

    def get(self, idx, *keys):
        """ Get the data at the given indices. """

        if len(keys) == 0:
            keys = ('o', 'a', 'next_r', 'next_term', 'next_trunc', 'next_o')

        tensors = {'o': self.observations, 'a': self.actions, 'next_r': self.next_rewards,
                   'next_term': self.next_terms, 'next_trunc': self.next_truncs, 'next_o': self.next_observations}

        tensors = tuple(tensors[key] for key in keys)

        device = self.device

        if isinstance(idx, np.ndarray):
            idx = torch.as_tensor(idx).to(device=device, non_blocking=True)

        if torch.is_tensor(idx):
            if device is not None and idx.device != device:
                idx = idx.to(device=device, non_blocking=True)

            assert torch.all(idx >= 0)
            assert torch.all(idx < self.len)

            flat_idx = idx.flatten()
            get_item = lambda t: t[flat_idx].reshape(idx.shape + t.shape[1:])

        else:
            assert torch.all(idx < self.len)

            get_item = lambda t: t[idx]

        if isinstance(tensors, (tuple, list)):
            return tuple(get_item(t) for t in tensors)
        elif isinstance(tensors, dict):
            return {key: get_item(t) for key, t in tensors.items()}
        elif torch.is_tensor(tensors):
            return get_item(tensors)
        else:
            raise ValueError()

    def step(self, a, next_o, next_r, next_term, next_trunc, cont_o):
        """ Add a transition to the buffer. """

        if self.len >= self.capacity:
            raise ValueError('Buffer is full')

        i = self.len
        next_o = torch.as_tensor(np.array(next_o), dtype=self.observations.dtype) \
            .to(device=self.observations.device, non_blocking=True)
        self.next_observations[i] = next_o

        self.actions[i] = a
        self.next_rewards[i] = next_r
        self.next_terms[i] = next_term
        self.next_truncs[i] = next_trunc

        self.episode_reward += next_r

        if next_term or next_trunc:
            next_o = torch.as_tensor(np.array(cont_o), dtype=self.observations.dtype) \
                .to(device=self.observations.device, non_blocking=True)

            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0.0

        self.len += 1
        if self.len < self.capacity:
            self.observations[self.len] = next_o

        next_r = self.next_rewards[i]
        next_term = self.next_terms[i]
        next_trunc = self.next_truncs[i]
        return next_r.unsqueeze(0), next_term.unsqueeze(0), next_trunc.unsqueeze(0), next_o.unsqueeze(0)

    def sample_idx(self, count, rng):
        """ Sample indices from the buffer. """
        idx = rng.choice(self.len, count, replace=False, shuffle=False)
        idx = torch.as_tensor(idx).to(device=self.device, non_blocking=True)
        return idx

    def get_stats(self):
        """ Get statistics about the buffer. """
        episode_rewards = np.array(self.episode_rewards)
        stats = {
            'buffer_size': self.len,
            'buffer_episodes': len(episode_rewards),
            'buffer_max_episode_reward': np.max(episode_rewards),
            'buffer_total_reward': np.sum(episode_rewards)
        }
        return stats

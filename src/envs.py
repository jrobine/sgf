from functools import partial

import gymnasium as gym


def atari(game, make, sticky, full_action_space, max_frames, noop_max, resolution, grayscale, frame_skip, frame_stack, episodic_life):
    """ Create an Atari environment. """
    if max_frames > 108000:
        raise NotImplementedError('NoFrameskip-v4 environments do not support max_frames > 108000')

    version = 'v0' if sticky else 'v4'
    env_id = f'ale_py:{game}NoFrameskip-{version}'
    wrappers = [
        partial(gym.wrappers.TimeLimit, max_episode_steps=max_frames),
        partial(gym.wrappers.AtariPreprocessing, noop_max=noop_max, frame_skip=frame_skip, screen_size=resolution,
                terminal_on_life_loss=False, grayscale_obs=grayscale, grayscale_newaxis=True),
        partial(gym.wrappers.FrameStackObservation, stack_size=frame_stack),
    ]

    if episodic_life:
        wrappers.append(EpisodicLifeWrapper)

    if game == 'Breakout':
        # fire on reset for Breakout
        if episodic_life:
            wrappers.append(FireResetWrapper)
        else:
            wrappers.append(FireLifeWrapper)

    kwargs = dict(full_action_space=full_action_space)

    if make:
        env = gym.make(env_id, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    else:
        return env_id, wrappers, kwargs


class EpisodicLifeWrapper(gym.Wrapper):
    # different from AtariPreprocessing: real reset only when game over

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.game_over = True

    def _ale_lives(self):
        return self.env.unwrapped.ale.lives()

    def reset(self, **kwargs):
        if self.game_over:
            o, info = self.env.reset(**kwargs)
        else:
            # noop after lost life
            o, _, _, _, info = self.env.step(0)
        self.lives = self._ale_lives()
        return o, info

    def step(self, action):
        next_o, next_r, next_term, next_trunc, info = self.env.step(action)
        self.game_over = next_term or next_trunc or self.game_over
        lives = self._ale_lives()
        if lives < self.lives and lives > 0:
            next_term = True
        self.lives = lives
        return next_o, next_r, next_term, next_trunc, info


class FireResetWrapper(gym.Wrapper):
    # adopted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py

    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self.fire = len(action_meanings) >= 3 and action_meanings[1] == 'FIRE'
        self.reset_reward = None

    def reset(self, **kwargs):
        if not self.fire:
            return self.env.reset(**kwargs)

        self.env.reset(**kwargs)

        _, next_r, next_term, next_trunc, _ = self.env.step(1)
        self.reset_reward = next_r
        if next_term or next_trunc:
            self.env.reset(**kwargs)
            self.reset_reward = 0

        o, next_r, next_term, next_trunc, info = self.env.step(2)
        self.reset_reward += next_r
        if next_term or next_trunc:
            o, info = self.env.reset(**kwargs)
            self.reset_reward = None

        return o, info

    def step(self, action):
        next_o, next_r, next_term, next_trunc, info = self.env.step(action)
        if self.reset_reward is not None:
            next_r += self.reset_reward
            self.reset_reward = None
        return next_o, next_r, next_term, next_trunc, info


class FireLifeWrapper(gym.Wrapper):
    # adopted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py

    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self.fire = len(action_meanings) >= 3 and action_meanings[1] == 'FIRE'
        self.lives = 0
        self.reset_reward = None

    def _ale_lives(self):
        return self.env.unwrapped.ale.lives()

    def reset(self, **kwargs):
        if not self.fire:
            return self.env.reset(**kwargs)

        self.env.reset(**kwargs)

        _, next_reward, next_term, next_trunc, _ = self.env.step(1)
        self.reset_reward = next_reward
        if next_term or next_trunc:
            self.env.reset(**kwargs)
            self.reset_reward = 0

        o, next_reward, next_term, next_trunc, info = self.env.step(2)
        self.reset_reward += next_reward
        if next_term or next_trunc:
            o, info = self.env.reset(**kwargs)
            self.reset_reward = None

        self.lives = self._ale_lives()
        return o, info

    def step(self, action):
        next_o, next_r, next_term, next_trunc, info = self.env.step(action)
        if self.reset_reward is not None:
            next_r += self.reset_reward
            self.reset_reward = None

        if not self.fire:
            return next_o, next_r, next_term, next_trunc, info

        lives = self._ale_lives()
        if not (next_term or next_trunc) and lives < self.lives and lives > 0:
            _, next_r_, next_term, next_trunc, _ = self.env.step(1)
            self.reset_reward = next_r_
            if not (next_term or next_trunc):
                _, next_r_, next_term, next_trunc, _ = self.env.step(2)
                self.reset_reward += next_r_
        self.lives = lives
        return next_o, next_r, next_term, next_trunc, info

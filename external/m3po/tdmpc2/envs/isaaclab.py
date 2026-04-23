import os
import warnings
from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings('ignore')


ISAACLAB_TASK_PREFIXES = ('Isaac-', 'Template-')


def _split_task_name(task: str) -> str:
	return task.split(':', 1)[-1]


def is_isaaclab_task(task: str) -> bool:
	"""Return whether the task refers to an Isaac Lab gym registration."""
	task_name = _split_task_name(task)
	if not task_name.startswith(ISAACLAB_TASK_PREFIXES):
		return False
	try:
		gym.spec(task_name)
	except gym.error.Error:
		return False
	return True


def _parse_bool_env(name: str) -> bool | None:
	value = os.environ.get(name)
	if value is None:
		return None
	return value.lower() in ('1', 'true', 'yes', 'on')


def _make_env_cfg(task_name: str, num_envs: int) -> object:
	from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

	device = os.environ.get('M3PO_ISAACLAB_DEVICE', 'cuda:0')
	use_fabric = _parse_bool_env('M3PO_ISAACLAB_USE_FABRIC')
	return parse_env_cfg(task_name, device=device, num_envs=num_envs, use_fabric=use_fabric)


def probe_task_info(task: str) -> dict:
	"""Collect action-space and horizon metadata for an Isaac Lab task."""
	task_name = _split_task_name(task)
	if not is_isaaclab_task(task_name):
		raise ValueError(f'Unknown task: {task}')

	env = gym.make(task_name, cfg=_make_env_cfg(task_name, num_envs=1))
	try:
		if not isinstance(env.single_action_space, gym.spaces.Box):
			raise TypeError(f'Isaac Lab task {task_name} does not expose a continuous Box action space.')
		return {
			'embodiment': task_name,
			'instruction': task_name.replace('-', ' '),
			'action_dim': int(np.prod(env.single_action_space.shape)),
			'max_episode_steps': int(env.unwrapped.max_episode_length),
			'text_embedding': [],
		}
	finally:
		env.close()


def _flatten_space(space: gym.Space) -> gym.spaces.Box:
	if isinstance(space, gym.spaces.Box):
		return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(int(np.prod(space.shape)),), dtype=np.float32)
	if isinstance(space, gym.spaces.Dict):
		size = sum(int(np.prod(_flatten_space(subspace).shape)) for subspace in space.spaces.values())
		return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)
	raise TypeError(f'Unsupported Isaac Lab observation space type: {type(space)}')


def _flatten_obs(obs) -> torch.Tensor:
	if isinstance(obs, torch.Tensor):
		return obs.reshape(obs.shape[0], -1)
	if isinstance(obs, np.ndarray):
		return torch.from_numpy(obs).reshape(obs.shape[0], -1)
	if isinstance(obs, Mapping):
		chunks = [_flatten_obs(obs[key]) for key in sorted(obs.keys())]
		return torch.cat(chunks, dim=-1)
	raise TypeError(f'Unsupported Isaac Lab observation type: {type(obs)}')


class IsaacLabWrapper(gym.Wrapper):
	"""Adapter from Isaac Lab vectorized environments to the M3PO env API."""

	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.num_envs = env.num_envs
		self.max_episode_steps = int(env.unwrapped.max_episode_length)
		self._success_term = None
		if hasattr(env.unwrapped, 'termination_manager'):
			active_terms = getattr(env.unwrapped.termination_manager, 'active_terms', [])
			if 'success' in active_terms:
				self._success_term = 'success'

		policy_space = env.single_observation_space['policy']
		self.observation_space = _flatten_space(policy_space)
		self.action_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			shape=env.single_action_space.shape,
			dtype=np.float32,
		)

	def _extract_obs(self, obs_dict) -> torch.Tensor:
		policy_obs = obs_dict['policy'] if isinstance(obs_dict, Mapping) else obs_dict
		return _flatten_obs(policy_obs).to(dtype=torch.float32, device='cpu')

	def _extract_success(self) -> torch.Tensor:
		if self._success_term is None:
			return torch.zeros(self.num_envs, dtype=torch.float32)
		success = self.env.unwrapped.termination_manager.get_term(self._success_term)
		return success.detach().to(dtype=torch.float32, device='cpu')

	def reset(self, **kwargs):
		obs_dict, _ = self.env.reset(**kwargs)
		obs = self._extract_obs(obs_dict)
		info = {'success': torch.zeros(self.num_envs, dtype=torch.float32)}
		return obs, info

	def step(self, action):
		if not isinstance(action, torch.Tensor):
			action = torch.tensor(action, dtype=torch.float32)
		action = action.to(device=self.env.unwrapped.device, dtype=torch.float32)

		obs_dict, reward, terminated, truncated, _ = self.env.step(action)
		obs = self._extract_obs(obs_dict)
		done = (terminated | truncated).detach().to(device='cpu', dtype=torch.bool)
		success = self._extract_success()

		info = {'success': success}
		if done.any():
			info['final_observation'] = obs[done].clone()
			info['final_info'] = {
				'success': success[done].clone(),
				'score': success[done].clone(),
			}

		return (
			obs,
			reward.detach().to(device='cpu', dtype=torch.float32),
			torch.zeros_like(done),
			done,
			info,
		)

	def rand_act(self):
		return torch.rand((self.num_envs, *self.action_space.shape), dtype=torch.float32) * 2 - 1

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)

	def close(self):
		return self.env.close()


def make_env(cfg):
	"""Make an Isaac Lab task for M3PO single-task training."""
	task_name = _split_task_name(cfg.task)
	if not is_isaaclab_task(task_name):
		raise ValueError('Unknown task:', cfg.task)
	if cfg.obs != 'state':
		raise NotImplementedError('Isaac Lab backend currently supports state observations only.')

	env_cfg = _make_env_cfg(task_name, num_envs=cfg.num_envs)
	env = gym.make(task_name, cfg=env_cfg, render_mode='rgb_array' if cfg.save_video else None)
	print(f'[Rank {cfg.rank}] Created Isaac Lab env for task {task_name}')
	return IsaacLabWrapper(env, cfg)

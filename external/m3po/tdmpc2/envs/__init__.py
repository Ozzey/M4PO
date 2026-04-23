import importlib
import warnings

warnings.filterwarnings('ignore')

import gymnasium as gym


_OPTIONAL_ENV_MODULES = (
	'envs.dmcontrol',
	'envs.maniskill',
	'envs.metaworld',
	'envs.mujoco',
	'envs.box2d',
	'envs.robodesk',
	'envs.ogbench',
	'envs.pygame',
	'envs.atari',
)


def make_env(cfg):
	"""
	Make an environment for M3PO experiments.
	"""
	gym.logger.set_level(40)
	import_failures = {}

	try:
		isaaclab_module = importlib.import_module('envs.isaaclab')
	except ModuleNotFoundError as exc:
		isaaclab_module = None
		import_failures['envs.isaaclab'] = exc

	if isaaclab_module is not None and isaaclab_module.is_isaaclab_task(cfg.task):
		env = isaaclab_module.make_env(cfg)
	elif not cfg.child_env:
		from envs.wrappers.vectorized_multitask import make_vectorized_multitask_env

		env = make_vectorized_multitask_env(cfg, make_env)
	else:
		env = None
		for module_name in _OPTIONAL_ENV_MODULES:
			try:
				module = importlib.import_module(module_name)
			except ModuleNotFoundError as exc:
				import_failures[module_name] = exc
				continue

			try:
				env = module.make_env(cfg)
				break
			except ValueError as exc:
				if 'Unknown task' in str(exc):
					continue
				raise

		if env is None:
			msg = f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.'
			if import_failures:
				details = ', '.join(
					f'{module} ({type(exc).__name__}: {exc})' for module, exc in import_failures.items()
				)
				msg += f' Optional backends that could not be imported: {details}.'
			raise ValueError(msg)

		assert cfg.num_envs == 1 or cfg.get('obs', 'state') == 'state', \
			'Vectorized environments only support state observations.'
		if cfg.save_video and cfg.get('num_demos', 0) > 0:
			from envs.wrappers.render import Render

			env = Render(env, cfg)
		print(f'[Rank {cfg.rank}] Created env for task {cfg.task}')

	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	return env

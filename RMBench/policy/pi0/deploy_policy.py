"""RMBench entry points expected by the simulator."""

from pi_model import MemBodiedPolicyAdapter


def encode_obs(observation):
    return (
        [
            observation["observation"]["head_camera"]["rgb"],
            observation["observation"]["right_camera"]["rgb"],
            observation["observation"]["left_camera"]["rgb"],
        ],
        observation["joint_action"]["vector"],
    )


def get_model(usr_args):
    return MemBodiedPolicyAdapter(
        backend=usr_args["backend"],
        config_name=usr_args["config_name"],
        checkpoint_dir=usr_args.get("checkpoint_dir"),
        asset_id=usr_args.get("asset_id"),
        action_chunk_size=usr_args.get("action_chunk_size", 50),
    )


def eval(task_env, model, observation):
    if model.observation_window is None:
        model.set_language(task_env.get_instruction())

    images, state = encode_obs(observation)
    model.update_observation_window(images, state)
    for action in model.get_action():
        task_env.take_action(action)
        observation = task_env.get_obs()
        images, state = encode_obs(observation)
        model.update_observation_window(images, state)


def reset_model(model):
    model.reset()


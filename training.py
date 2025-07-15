import os
import re
import yaml
import argparse
from environment.environment import Environment
from modifiers.metric_damage import RewardWrapper
from copy import deepcopy
from train.misc import make_sb3_env, linear_schedule, AutoSave, StartingSteps, CustomMetrics

from train.custom_ppo_policy import CustomPPOPolicy
from train.custom_extractors import CustomFlatExtractor, CustomCombinedExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./train/config.yaml", help='Type of control policy')
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as file:
        train_config_in = yaml.safe_load(file)

    train_config = deepcopy(train_config_in)
    num_envs = train_config["num_envs"]

    local_path = os.path.dirname(os.path.abspath(__file__))

    results_folder = os.path.join(local_path, "runs", train_config["name"])
    model_folder = os.path.join(results_folder, "model")
    tensor_board_folder = os.path.join(results_folder, "tb")
    monitor_folder = os.path.join(results_folder, "monitor")
    monitor_folder_eval = os.path.join(results_folder, "monitor_eval")

    evaluation_config = train_config["evaluation"]
    model_save_config = train_config["model_save"]
    training_stop_config = train_config["training_stop"]

    os.makedirs(model_folder, exist_ok=True)

    env = make_sb3_env(Environment, RewardWrapper, num_envs, seed=train_config["seed"], monitor_folder=monitor_folder)
    print("Activated {} environment(s)".format(num_envs))

    # Policy param
    module = globals()
    policy = module[train_config["policy"]]
    policy_kwargs = train_config["policy_kwargs"]
    policy_kwargs["features_extractor_class"] = module[policy_kwargs["features_extractor_class"]]
    print("Policy kwargs:", policy_kwargs)

    # Generic algo settings
    gamma = train_config["gamma"]
    model_checkpoint_path = train_config["model_checkpoint_path"]
    learning_rate = linear_schedule(train_config["learning_rate"][0], train_config["learning_rate"][1])

    if train_config["algo"] == "PPO":
        clip_range = linear_schedule(train_config["clip_range"][0], train_config["clip_range"][1])
        clip_range_vf = clip_range
        n_epochs = train_config["n_epochs"]
        n_steps = train_config["n_steps"]
        batch_size = train_config["batch_size"]
        min_steps = num_envs * n_steps
        assert training_stop_config["max_time_steps"] > min_steps, "The minimum number of training steps is {}".format(min_steps)
        gae_lambda = train_config["gae_lambda"]
        normalize_advantage = train_config["normalize_advantage"]
        ent_coef = train_config["ent_coef"]
        vf_coef = train_config["vf_coef"]
        max_grad_norm = train_config["max_grad_norm"]
        use_sde = train_config["use_sde"]
        sde_sample_freq = train_config["sde_sample_freq"]
        target_kl = train_config["target_kl"]
    else:
        raise ValueError(f"Algorithm {train_config['algo']} not supported")

    starting_steps = 0
    reset_num_timesteps = True
    callbacks = [CustomMetrics()]

    if model_checkpoint_path is None:
        if train_config["algo"] == "PPO":
            agent = PPO(policy, env, verbose=1,
                        gamma=gamma, batch_size=batch_size,
                        n_epochs=n_epochs, n_steps=n_steps,
                        learning_rate=learning_rate, clip_range=clip_range,
                        clip_range_vf=clip_range_vf, policy_kwargs=policy_kwargs,
                        gae_lambda=gae_lambda, normalize_advantage=normalize_advantage,
                        ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                        use_sde=use_sde, sde_sample_freq=sde_sample_freq, target_kl=target_kl,
                        tensorboard_log=tensor_board_folder, device="cpu")
    else:
        # Load the trained agent
        # Use regex to find the number after the latest underscore
        match = re.search(r'_(\d+)(?!.*_\d)', model_checkpoint_path)
        if not match:
            raise Exception(f"{model_checkpoint_path} should contain a number at the end of the filename indicating the number of training steps.")
        starting_steps = int(match.group(1))  # Convert the found number to an integer

        if train_config["algo"] == "PPO":
            agent = PPO.load(model_checkpoint_path, env=env,
                             batch_size=batch_size, n_epochs=n_epochs, n_steps=n_steps, gamma=gamma,
                             learning_rate=learning_rate, clip_range=clip_range, clip_range_vf=clip_range_vf,
                             gae_lambda=gae_lambda, normalize_advantage=normalize_advantage, ent_coef=ent_coef,
                             vf_coef=vf_coef, max_grad_norm=max_grad_norm, use_sde=use_sde,
                             sde_sample_freq=sde_sample_freq, target_kl=target_kl,
                             tensorboard_log=tensor_board_folder, device="cpu")
        reset_num_timesteps = False
        callbacks.append(StartingSteps(starting_steps=starting_steps))

    # Print policy network architecture
    print("Policy architecture:")
    print(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave = model_save_config["autosave"]["active"]
    autosave_freq = model_save_config["autosave"]["frequency"]
    save_best_model = model_save_config["save_best_model"]["active"]
    best_check_freq = model_save_config["save_best_model"]["frequency"]
    best_on_eval = model_save_config["save_best_model"]["reward_to_use"] == "eval"
    best_monitor_folder = None
    num_best_envs = None
    if autosave:
        if save_best_model:
            if best_on_eval:
                assert train_config["evaluation"]["active"], "Evaluation must be active to save the best model on evaluation"
                assert best_check_freq == train_config["evaluation"]["frequency"], "Best check frequency must be equal to evaluation frequency"
                best_monitor_folder = monitor_folder_eval
                num_best_envs = train_config["evaluation"]["num_eval_envs"]
            else:
                best_monitor_folder = monitor_folder
                num_best_envs = num_envs
        callbacks.append(AutoSave(check_freq=autosave_freq, num_envs=num_envs, save_path=model_folder, filename_prefix="model_", starting_steps=starting_steps,
                                  save_best_model=save_best_model, best_check_freq=best_check_freq, monitor_folder=best_monitor_folder, num_best_envs=num_best_envs))

    callback_on_best = None
    stop_train_callback = None
    if training_stop_config["reward_threshold"]["active"]:
        assert train_config["evaluation"]["active"], "Evaluation must be active to stop training on reward threshold"
        # Stop training when the model reaches the reward threshold
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=training_stop_config["reward_threshold"]["value"],
            verbose=1)
    if training_stop_config["no_improvement_evals"]["active"]:
        assert train_config["evaluation"]["active"], "Evaluation must be active to stop training on no improvement evals"
        # Stop training if there is no improvement after more than 3 evaluations
        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=training_stop_config["no_improvement_evals"]["max_no_improvement_evals"],
            min_evals=training_stop_config["no_improvement_evals"]["min_evals"],
            verbose=1)

    if evaluation_config["active"]:
        # Separate evaluation env
        num_eval_envs = evaluation_config["num_eval_envs"]
        assert num_eval_envs == 1 or env_type == EnvironmentType.STANDARD, "Evaluation must be done on a single environment for vectorized environments (custom monitor wrapper not supported yet)"
        eval_env = make_sb3_env(env_addresses[-num_eval_envs:], num_eval_envs, env_type, spaces_modifiers_config_file_path, reward_modifiers_config_file_path, monitor_folder=monitor_folder_eval, seed=train_config["seed"])

        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=evaluation_config["n_eval_episodes"],
            eval_freq=evaluation_config["frequency"],
            deterministic=evaluation_config["deterministic"],
            callback_on_new_best=callback_on_best,
            callback_after_eval=stop_train_callback,
            verbose=1,
            render=False
        )

        callbacks.append(eval_callback)

    # Train the agent
    time_steps = training_stop_config["max_time_steps"]
    agent.learn(total_timesteps=time_steps, reset_num_timesteps=reset_num_timesteps, callback=callbacks)

    # Save the agent
    new_model_checkpoint = "model_" + str(starting_steps + time_steps)
    model_path = os.path.join(model_folder, new_model_checkpoint)
    agent.save(model_path)

    # Free memory
    assert agent.env is not None
    agent.env.close()
    del agent.env
    del agent

    if evaluation_config["active"]:
        eval_env.close()
        del eval_env
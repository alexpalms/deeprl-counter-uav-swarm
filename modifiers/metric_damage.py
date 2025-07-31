import gymnasium as gym
from environment.supporting_classes import EffectorWeaponState, EffectorKinematicState

class CustomWrapper(gym.Wrapper):
    """
    A gym wrapper that normalizes rewards by dividing them by a scaling factor.
    """
    def __init__(self, env):
        super().__init__(env)
        self.max_reward_magnitude = None
        self.cumulative_damage = None
        self.effectors_kinematic_states_counters = {
            EffectorKinematicState.CHASING: 0,
            EffectorKinematicState.TRACKING: 0
        }
        self.effectors_weapon_states_counters = {
            EffectorWeaponState.READY: 0,
            EffectorWeaponState.SHOOTING: 0,
            EffectorWeaponState.RECHARGING: 0
        }
        self.num_effectors = len(self.env.effectors_list)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        for effector in self.env.effectors_list:
            self.effectors_kinematic_states_counters[effector.kinematic_state] += 1
            self.effectors_weapon_states_counters[effector.weapon_state] += 1
        normalized_reward = abs(float(reward / self.max_reward_magnitude)) * 100
        self.cumulative_damage += normalized_reward
        if terminated or truncated:
            info["custom_eval_metrics/damage"] = self.cumulative_damage
            info["custom_eval_metrics/effectors_kinematic_states"] = {k.name: round(v/(self.num_effectors * self.env.tick), 2)for k, v in self.effectors_kinematic_states_counters.items()}
            info["custom_eval_metrics/effectors_weapon_states"] = {k.name: round(v/(self.num_effectors * self.env.tick), 2) for k, v in self.effectors_weapon_states_counters.items()}
            info["custom_eval_metrics/effectors_weapon_states"]["UTILIZATION"] = round(
                (self.effectors_weapon_states_counters[EffectorWeaponState.SHOOTING]  + self.effectors_weapon_states_counters[EffectorWeaponState.RECHARGING])/(self.num_effectors * self.env.tick), 2
            )


        return observation, normalized_reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.cumulative_damage = 0
        self.effectors_kinematic_states_counters = {
            EffectorKinematicState.CHASING: 0,
            EffectorKinematicState.TRACKING: 0
        }
        self.effectors_weapon_states_counters = {
            EffectorWeaponState.READY: 0,
            EffectorWeaponState.SHOOTING: 0,
            EffectorWeaponState.RECHARGING: 0
        }
        self.max_reward_magnitude = info["max_reward_magnitude"]
        return observation, info

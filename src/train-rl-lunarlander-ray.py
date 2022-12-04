import gym
from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher


def create_environment(env_config):
    return gym.make("LunarLander-v2")


class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):
        register_env("LunarLander-v2", create_environment)

    def get_experiment_config(self):
        return {
            "training": {
                "env": "LunarLander-v2",
                "run": "DQN",
                "stop": {"training_iteration": 70},
                "config": {
                    "framework": "tf",
                    "gamma": 0.99,
                    "num_workers": (self.num_cpus - 1),
                    "num_gpus": self.num_gpus,
                    "batch_mode": "truncate_episodes",
                },
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()

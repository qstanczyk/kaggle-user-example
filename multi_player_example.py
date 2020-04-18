import os

from kaggle_environments import make, register, evaluate
from kaggle_environments.envs.football import football

env = make("football", debug=True, configuration={"scenario_name": "11_vs_11_stochastic", "team_1": 1, "team_2": 0, "episodeSteps": 100, "render": False, "save_video": True})
print(env.name, env.version)
print("Default Agents: ", *env.agents)

env.run(["run_right", "run_left"])
print("Video: %s" % football.get_video_path(env))
football.cleanup(env)
print("Logs stored in /tmp/football/%s" % env.id)


configuration = {"scenario_name": "11_vs_11_stochastic", "team_1": 1, "team_2": 0, "episodeSteps": 100, "render": False, "save_video": True}
agents = ["run_right", "run_left"]
rewards = evaluate("football", agents, configuration, steps=[], num_episodes=10)
## Broken: evaluate looks only on rewards from the last step.
# (or should we finish after a scored goal??)
print(rewards)
football.cleanup_all()

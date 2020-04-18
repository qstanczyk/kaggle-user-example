import os

from kaggle_environments import make, register
from kaggle_environments.envs.football import football

env = make("football", debug=True, configuration={"scenario_name": "11_vs_11_stochastic", "team_1": 1, "team_2": 1, "episodeSteps": 30, "render": True, "save_video": True})
print(env.name, env.version)
print("Default Agents: ", *env.agents)

env.run(["run_right", "run_left"])
football.cleanup(env)
print("Logs stored in /tmp/football/%s" % env.id)

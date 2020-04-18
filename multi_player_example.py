import os

from kaggle_environments import make, register
from kaggle_environments.envs.football import football

env = make("football", debug=True, configuration={"scenario_name": "11_vs_11_stochastic", "team_1": 1, "team_2": 0, "episodeSteps": 100, "render": False, "save_video": True})
print(env.name, env.version)
print("Default Agents: ", *env.agents)

env.run(["run_right", "run_left"])
print("Video: %s" % football.get_video_path(env))
football.cleanup(env)
print("Logs stored in /tmp/football/%s" % env.id)




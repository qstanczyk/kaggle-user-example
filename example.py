import os

from kaggle_environments import make, register
from kaggle_environments.envs.football import football

env = make("football", debug=True, configuration={"scenario_name": "11_vs_11_kaggle", "team_1": 11, "team_2": 11, "episodeSteps": 100, "render": True, "save_video": True, "agentExec":"LOCAL"})
print(env.name, env.version)
print("Default Agents: ", *env.agents)

env.run(["run_right", "run_left"])
football.cleanup(env)
print("Logs stored in /tmp/football/%s" % env.id)
print("Video: %s" % env.football_video_path)


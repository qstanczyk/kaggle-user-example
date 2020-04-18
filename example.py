from kaggle_environments import make, register

env = make("football", debug=True, configuration={"scenario_name": "11_vs_11_stochastic", "team_1": 1, "team_2": 1})
print(env.name, env.version)
print("Default Agents: ", *env.agents)

env.run(["run_right", "run_left"])
# env.render(mode="ipython", width=960, height=720)

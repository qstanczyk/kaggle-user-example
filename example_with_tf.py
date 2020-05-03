import os
import tensorflow as tf
from kaggle_environments import make, register
from kaggle_environments.envs.football import football
from gfootball.env import observation_preprocessing
from gfootball.env import wrappers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np


## Example with a super-simple keras model.

class DummyKerasModel(object):
  def __init__(self):
    model = Sequential()
    # very simple 1 layer network.
    model.add(Flatten(input_shape=(72, 96, 4)))
    model.add(Dense(19, activation='softmax'))
    model.build()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()
    self._model = model

  def train(self, data, labels):
    self._model.fit(data, labels, epochs=1, batch_size=1)

  def run_agent(self, obs, config, reward, info):
    print('About to start the agent')

    # Simple115 observation
    simple115_obs = wrappers.Simple115StateWrapper.convert_observation(obs.players_raw, True)
    # Or minimap observation.
    minimap = observation_preprocessing.generate_smm(obs.players_raw) 

    ## TODO: this should not be a batch dimension.
    print("Calling the model")
    action = np.argmax(self._model(minimap))
    print("Done")
    # you have to cast it back to int (from numpy.int64)
    return [int(action)]


model = DummyKerasModel()
# some dummy training.
model.train(np.zeros((10, 72, 96, 4)), np.ones((10,)))

env = make("football", debug=True, configuration={"scenario_name": "test_example_multiagent", "team_1": 1, "team_2": 1, "episodeSteps": 100, "render": False, "save_video": True})
print(env.name, env.version)
print("Default Agents: ", *env.agents)

env.run([model.run_agent, "run_left"])
football.cleanup(env)
print("Logs stored in /tmp/football/%s" % env.id)

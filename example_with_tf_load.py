import os
import tensorflow as tf
from kaggle_environments import make, register
from kaggle_environments.envs.football import football


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import IPython

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

def running_training():
  model = DummyKerasModel()
  # some dummy training.
  model.train(np.zeros((10, 72, 96, 4)), np.ones((10,)))
  tf.saved_model.save(model._model, "my_saved_model")



class LoadedModel(object):
  def __init__(self, model_path):
    self._model_path = model_path
    self._model = None

  def run_agent(self, obs, config, reward, info):
    if not self._model:
      # You must load the model on the first run_agent call (rather than in constructor)
      # As the agents can be called from a separate process.
      self._model = tf.saved_model.load(self._model_path)
    print('About to start the agent')
    minimap = np.reshape(np.array(obs['minimap']), (72, 96, 4))
    # Add a batch dimension
    minimap = np.expand_dims(minimap, axis=0)
    print("Calling the model")
    action = np.argmax(self._model(tf.cast(minimap, tf.float32)))
    print("Done")
    # you have to cast it back to int (from numpy.int64)
    return [int(action)]


def running_evaluation_in_notebook():
  loaded_model = LoadedModel("my_saved_model")
  print("!! Loaded %s" % loaded_model )
  env = make("football", debug=True, configuration={"scenario_name": "11_vs_11_stochastic", "team_1": 1, "team_2": 1, "episodeSteps": 30, "render": False, "save_video": True, "agentExec":"PROCESS"})
  print(env.name, env.version)
  print("Default Agents: ", *env.agents)

  env.run([loaded_model.run_agent, "run_left"])
  football.cleanup(env)
  print("Logs stored in /tmp/football/%s" % env.id)



print("Code that you run on your machine to train the agent.")
#running_training()

print("Code that you 'submit' / run via Kaggle notebook for competition.")
running_evaluation_in_notebook()

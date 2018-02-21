# Example of custom defined HMM and samping from the model
# 1- we define an HMM and all its parameters
# 2- we extract some samples from the defined model

import numpy as np
from hmmlearn import hmm
np.random.seed(42)

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

start_probability = np.array([0.6, 0.4])

transition_probability = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])

emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

X, Z = model.sample(5)
print "X:", X
print "Z:", Z

print "States:", ", ".join(map(lambda x: states[x], Z))
print "Performed actions:", ", ".join(map(lambda x: observations[x], np.squeeze(np.asarray(X))))


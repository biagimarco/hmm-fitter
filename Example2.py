# Example of fitted HMM and sampling
# 1- we create an HMM fitting it from data
# 2- we extract some samples from the fitted model

import numpy as np
from hmmlearn import hmm
np.random.seed(42)

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

train_data = [0, 2, 1, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0]

model = hmm.MultinomialHMM(n_components=n_states,  n_iter=100)
model.fit(np.array([train_data]).T)

print "start probs: ", model.startprob_
print "transmat: ", model.transmat_
print "emissionprob_", model.emissionprob_

X, Z = model.sample(5)
print "X:", X
print "Z:", Z

print "States:", ", ".join(map(lambda x: states[x], Z))
print "Performed actions:", ", ".join(map(lambda x: observations[x], np.squeeze(np.asarray(X))))


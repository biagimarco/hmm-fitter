# Example of fitted HMM and prediction
# 1- we create an HMM fitting it from data
# 2- we perform prediction based on the fitted model

import numpy as np
from hmmlearn import hmm
np.random.seed(42)

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

train_1 = [[0], [2], [1], [1], [2], [0], [1], [2], [0], [0], [0], [0], [0], [0]]
train_2 = [[0], [2], [1], [1], [2], [0], [1], [2], [0], [0]]


train_seq = np.concatenate([train_1, train_2])
train_lengths = [len(train_1), len(train_2)]
test_data = [0, 2, 1, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0]

print type(train_lengths)
print type(train_seq)

model = hmm.MultinomialHMM(n_components=n_states,  n_iter=100)
model.fit(train_seq, train_lengths)

print "start probs: ", model.startprob_
print "transmat: ", model.transmat_
print "emissionprob_", model.emissionprob_

R = model.predict(np.array([test_data]).T)

print "R:", R

print "Predicted states:", ", ".join(map(lambda x: states[x], R))
# Example of fitting HMM with distinct initial values
# and evaluating their score
#Note that 'init_params="te"' means that transitions 't' and emission 'e' probabilities are initialized
#randomly by the lib, while initial start probabilities are user specified

import numpy as np
from hmmlearn import hmm
np.random.seed(43)

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

train_data = [0, 2, 1, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0]
test_data = [0, 2, 1, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0]

model = hmm.MultinomialHMM(n_components=n_states,  n_iter=1000, init_params="te")
model.startprob_=np.array([0.6, 0.4])
model.fit(np.array([train_data]).T)

print "start probs: ", model.startprob_
print "transmat: ", model.transmat_
print "emissionprob_", model.emissionprob_

S = model.score(np.array([test_data]).T)

print "S:", S
print "Re model-------"

remodel = hmm.MultinomialHMM(n_components=n_states,  n_iter=1000, init_params="te")
remodel.startprob_=np.array([0.7, 0.3])
remodel.fit(np.array([train_data]).T)

print "start probs: ", remodel.startprob_
print "transmat: ", remodel.transmat_
print "emissionprob_", remodel.emissionprob_

S = remodel.score(np.array([test_data]).T)

print "S:", S

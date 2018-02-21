# This script get an input file containing sequences of observations and
# train a MultinomialHMM. The result is save on the specified output file

import numpy as np
import sys

from hmmlearn import hmm
np.random.seed(42)

ELEMENT_SEPARETOR = ','

#Function that allows to save the model to a file with our custom format
def saveModel(output_filename, model):
    with open(output_filename, 'w') as f:
        f.write("number of states:" + str(model.n_components) + "\n")

        f.write("start probs:\n")
        for p in model.startprob_:
            f.write(str(p) + '\n')

        f.write("transition matrix:\n")
        for i in model.transmat_:
            for j in i:
                f.write(str(j) + ', ')
            f.write('\n')

        f.write("emission probs [0,1]:\n")
        for i in model.emissionprob_:
            for j in i:
                f.write(str(j) + ', ')
            f.write('\n')

#Main routine
def main():
    #1- Read arguments(n_iterations, n_states, input_filename, output_filename)
    n_iterations = int(sys.argv[1])
    n_states = int(sys.argv[2])
    input_filename = sys.argv[3]
    output_filename = sys.argv[4]

    #2- Prepare training data
    with open(input_filename) as f:
        input_content = f.readlines()
    input_content = [x.strip() for x in input_content] #remove new lines
    input_content = [x.split(ELEMENT_SEPARETOR) for x in input_content] #split by comma

    train_lengths = [len(x) for x in input_content] #train sequence lengths
    train_seq = [j for i in input_content for j in i] # train concatenated sequences
    train_seq = [int(x) for x in train_seq] #convert to number
    train_seq = np.array(train_seq).reshape(len(train_seq), 1)

    #3- Train HMM
    model = hmm.MultinomialHMM(n_components=n_states,  n_iter=n_iterations)
    model.fit(train_seq, train_lengths)

    #4- Save HMM to file
    saveModel(output_filename, model)

main()
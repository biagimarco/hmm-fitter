# This script get an input file containing sequences of observations and
# train a MultinomialHMM. The result is save on the specified output file

import numpy as np
import sys
import time

from hmmlearn import hmm
np.random.seed(42)

ELEMENT_SEPARETOR = ','

#Decorator for timing
def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f s' % (f.func_name, (time2-time1))
        return ret
    return wrap

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

        f.write("emission probs:\n")
        for i in model.emissionprob_:
            for j in i:
                f.write(str(j) + ', ')
            f.write('\n')

@timing
def train_HMM(n_states, n_iterations, train_seq, train_lengths):
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=n_iterations)
    model.fit(train_seq, train_lengths)
    return model

#Main routine
@timing
def main():
    #1- Read arguments(n_iterations, n_states, input_filename, output_filename, n_multi_start)
    n_iterations = int(sys.argv[1])
    n_states = int(sys.argv[2])
    input_filename = sys.argv[3]
    output_filename = sys.argv[4]
    n_multi_start = int(sys.argv[5])

    #2- Prepare training data
    print "preprocessing training dataset"
    with open(input_filename) as f:
        input_content = f.readlines()
    input_content = [x.strip() for x in input_content] #remove new lines
    input_content = [x.split(ELEMENT_SEPARETOR) for x in input_content] #split by comma

    train_lengths = [len(x) for x in input_content] #train sequence lengths
    train_seq = [j for i in input_content for j in i] # train concatenated sequences
    train_seq = [int(x) for x in train_seq] #convert to number
    train_seq = np.array(train_seq).reshape(len(train_seq), 1)
    print "training dataset processed"

    #3- Train HMMs
    models = []
    for j in range(0, n_multi_start):
        np.random.seed(42+j)
        print "starting HMM training #" + str(1+j)
        model = train_HMM(n_states, n_iterations, train_seq, train_lengths)
        score = model.score(train_seq, train_lengths)
        print "HMM trained #" + str(1+j)
        print "Convergence informations: " + str(model.monitor_)
        models.append({"model": model, "score": score})

    #4- find best HMM
    best_model = min(models, key=lambda x: x['score'])
    print "Best score is: " + str(best_model["score"])

    #4- Save HMM to file
    print "saving HMM on file"
    saveModel(output_filename, best_model["model"])
    print "HMM saved"



main()
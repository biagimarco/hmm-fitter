# HMM fitter

- Use python 2
- Need python packages *scipy*, *numpy*, *sklearn* and *hmmlearn*
- Also a requirements.txt file is present

## Usage

*python HMMModelFitter.py <n_iterations> <n_states> <input_file> <output_file>*

Where
- <n_iterations>: number of iterations executed by the fitting algorithm
- <n_states>: number of hidden states
- <input_file>: input file name
- <output_file>: output file name

Note that the input file need to contains a line for each sequence. Elements of the same sequence are comma separated.

An example is the following:
*python HMMModelFitter.py 100 4 Dataset/inputDataset.dat Dataset/outputDataset.dat*


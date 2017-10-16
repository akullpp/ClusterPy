# ClusterPy

Bayesian Hidden Markov Model with Gibbs sampling

## Requirements

Python 2.7.3+

## Usage

```
./cluster.py <input file> [output file] <number of labels> <number of iterations> <alpha> <beta>
```

## Arguments

```
<input file> - File with sequences, one per line
[output file] - Output file
<number of labels> - Possible labels
<number of iterations> - Number of sampling iterations
<alpha> - Hyperparameter for transitions
<beta> - Hyperparameter for emissions
```

## Content

* bhmm.py - Gibbs sampling
* cluster.py - Entry point
* utility.py - Utility methods
* corpus.txt - toy corpus

## License

MIT

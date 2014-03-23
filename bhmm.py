from codecs import open
from random import randint, uniform
from collections import defaultdict
from math import log

from utility import change_count
from utility import get_value


class BHMM(object):
    """ Bayesian Hidden Markov Model with Gibbs sampling. """

    def __init__(self, args):
        # Input file
        self.fin = args.input
        # Output file
        self.fout = args.output
        # Number of possible labels
        self.labels = args.labels
        # Number of sampling iterations
        self.iterations = args.iterations
        # self.transition hyperparameter
        self.alpha = args.alpha
        # self.emission hyperparameter
        self.beta = args.beta
        # Lists of observations
        self.data = []
        # Uniform distribution of observations
        self.frequencies = defaultdict(float)
        # Delimits single observation
        self.delimiter = " "
        # Emission matrix: C(previous label, label)
        self.emission = defaultdict(int)
        # Transition Matrix: C(label, emission)
        self.transition = defaultdict(int)
        # Lists labels sequences
        self.sequences = []
        # Base probability
        self.base_probability = 1.0 / self.labels
        # Logarithmic likelihoods for each iteration
        self.likelihoods = []

    def __read_data(self):
        """ Creates a uniform distribution. """
        print "Reading corpus"

        with open(self.fin, encoding="utf8") as f:
            for line in f:
                # Sequence of observations
                unit = ["START"]

                for item in line.split(self.delimiter):
                    item = item.strip()
                    unit.append(item)
                    self.frequencies[item] += 1.0
                unit.append("END")
                self.data.append(unit)

    def __create_frequencies(self):
        """ Calculate relative frequency. """
        print "Creating frequencies"
        # Total number of observations
        total = sum(self.frequencies.values())

        for key in self.frequencies.keys():
            self.frequencies[key] /= total

    def __create_matrixes(self):
        """ Creates transition and emission matrix. """
        print "Creating matrixes"

        for unit in self.data:
            # Ordered list of hidden labels framed by -1
            sequence = [-1]

            for observation in unit[1:-1]:
                # Assign random label to observation
                label = randint(0, self.labels - 1)
                # Add C(label|previous label)
                change_count(self.transition, label, sequence[-1], 1)
                # Add C(emission|label)
                change_count(self.emission, observation, label, 1)
                sequence.append(label)
                # Last transition add C(-1|previous label)
            change_count(self.transition, "-1", sequence[-1], 1)
            sequence.append(-1)
            # Add sequence of observations list of sequences
            self.sequences.append(sequence)

    def __initialize_model(self):
        """ Initializes the HMM """
        print "Initializing model"
        self.__read_data()
        print "Corpus read"
        self.__create_frequencies()
        print "Frequencies created"
        self.__create_matrixes()
        print "Matrixes created"

    def __compute_probability(self, matrix, items, base, hyper):
        """ Calculating posterior.
        
        Arguments:
        matrix -- transition or emission
        items -- (hypothesis, evidence)
        base -- base probability
        hyper -- hyperparameter
        """
        x = get_value(matrix, items[0], items[1])
        y = get_value(matrix, items[1])

        return (x + base * hyper) / (y + hyper)

    def __sample_label(self, probabilities):
        """ Sample label.
        
        Arguments:
        probabilities -- probabilities of all labels
        """
        z = sum(probabilities)
        remaining = uniform(0, z)

        for probability in probabilities:
            remaining -= probability
            if remaining <= 0:
                return probabilities.index(probability)

    def __compute_label_probabilities(self, blanket):
        """ Computes the probability of each label.
        
        Arguments:
        blanket -- Markov blanket
        """
        _, previous_label, following_label, current_observation = blanket
        # Probabilities of each possible label
        probabilities = []

        for label in xrange(self.labels):
            # Chain rule
            probability = (self.__compute_probability(self.transition,
                                                      (label, previous_label),
                                                      self.base_probability,
                                                      self.alpha) *
                           self.__compute_probability(self.transition,
                                                      (following_label, label),
                                                      self.base_probability,
                                                      self.alpha) *
                           self.__compute_probability(self.emission,
                                                      (current_observation, label),
                                                      self.frequencies[current_observation],
                                                      self.beta))
            probabilities.append(probability)
        return probabilities

    def __write_labeled_data(self):
        """ Writes labeled data to output file. """
        print "Writing data"

        with open(self.fout, "w", encoding="utf8") as f:
            for i in xrange(len(self.data)):
                labeled_unit = []

                for j in xrange(len(self.sequences[i])):
                    labeled_unit.append("%s/%s" % (self.data[i][j],
                                                   self.sequences[i][j]))
                f.write("%s\n" % " ".join(labeled_unit[1:-1]))

    def draw_likelihood(self):
        """ Draw the logarithmic likelihood. """
        from matplotlib import pyplot

        print "Drawing likelihood"
        pyplot.plot(self.likelihoods[::self.iterations / 100])
        pyplot.show()

    def __compute_label_likelihood(self, blanket):
        """ Likelihood of label sequence.
        
        Arguments:
        blanket - Markov blanket
        """
        current_label, previous_label, following_label, current_observation = blanket
        # P(label|previous label)
        p = self.__compute_probability(self.transition,
                                       (current_label, previous_label),
                                       self.base_probability,
                                       self.alpha) * \
            self.__compute_probability(self.emission,
                                       (current_observation, current_label),
                                       self.frequencies[current_observation],
                                       self.beta)

        # Last transition
        if following_label == -1:
            # P(following label|label)
            p *= self.__compute_probability(self.transition,
                                            (following_label, current_label),
                                            self.base_probability,
                                            self.alpha)
        return p

    def __change_sample(self, blanket, i):
        """ Adds (i = 1) or removes (i = -1) a sample.
        
        Arguments:
        blanket -- affected labels
        i -- add or remove
        """
        current_label, previous_label, following_label, current_observation = blanket

        change_count(self.transition, current_label, previous_label, i)
        change_count(self.transition, following_label, current_label, i)
        change_count(self.emission, current_observation, current_label, i)

    def run(self):
        """ Gibbs sampling. """
        self.__initialize_model()
        print "Model initialized\nStarting iterations\n"

        for _ in xrange(self.iterations):
            likelihood = 0.0

            for i, sequence in enumerate(self.sequences):
                for j in xrange(1, len(self.sequences[i]) - 1):
                    # Markov blanket affected by changing label
                    blanket = [sequence[j],
                               sequence[j - 1],
                               sequence[j + 1],
                               self.data[i][j]]
                    # Remove sample
                    self.__change_sample(blanket, -1)
                    # Probabilities of each label
                    probabilities = self.__compute_label_probabilities(blanket)
                    # Sample current label
                    sequence[j] = self.__sample_label(probabilities)
                    # Update blanket
                    blanket[0] = sequence[j]
                    # Likelihood of current label
                    p = self.__compute_label_likelihood(blanket)
                    likelihood += log(p)
                    # Add sample
                    self.__change_sample(blanket, 1)
            print "Iteration %s \t Likelihood %f" % (_ + 1, likelihood)
            self.likelihoods.append(likelihood)
        print "\nIterations finished"
        self.__write_labeled_data()
        print "Data written"
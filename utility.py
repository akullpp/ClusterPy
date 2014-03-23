def parse_args():
    """ Parsing commandline arguments. """
    from argparse import ArgumentParser

    parser = ArgumentParser("./cluster.py")
    parser.add_argument("input", help="input file")
    parser.add_argument("output",
                        nargs="?",
                        help="output file",
                        default="out.txt")
    parser.add_argument("labels", type=int, help="number of labels")
    parser.add_argument("iterations",
                        type=int,
                        help="number of iterations")
    parser.add_argument("alpha",
                        type=float,
                        help="transition hyperparameter")
    parser.add_argument("beta", type=float, help="emission hyperparameter")
    return parser.parse_args()


def change_count(matrix, x, y, i):
    """ Change the count in a matrix.
    
    Arguments:
    matrix - transition or emission matrix
    x - emission or label
    y - label
    i - change in count
    """
    matrix["%s|%s" % (x, y)] += i
    matrix["%s" % y] += i


def get_value(matrix, *args):
    """ Returns the value to a key.
    
    Arguments:
    *args arbitrary number of arguments
    """
    if len(args) == 2:
        return matrix["%s|%s" % (args[0], args[1])]
    elif len(args) == 1:
        return matrix["%s" % (args[0])]
    else:
        raise Exception("Invalid argument list: " + str(args))

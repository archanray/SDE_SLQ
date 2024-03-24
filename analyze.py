import argparse
from src.aggregator import summarizer
from src.display_codes import plotter

def main(args):
    if args.methods == "all":
        methods = ["slq", "kd"]
    else:
        methods = args.methods
    errors = {}
    sds = {}
    blocks = {}
    for m in methods:
        errors[m], sds[m], blocks[m] = summarizer(args.dataset, m)
    plotter(errors, sds, blocks, methods, args.dataset)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SDE eval variables")
    parser.add_argument('--dataset', '-d',
                        dest='dataset', 
                        type=str, 
                        default="random", 
                        required=False,
                        help="choose datasets here")
    parser.add_argument('--methods', '-m',
                        dest='methods', 
                        type=str, 
                        default="all", 
                        required=False,
                        help="choose methdos to compare here")
    args = parser.parse_args()
    print(args)
    main(args)

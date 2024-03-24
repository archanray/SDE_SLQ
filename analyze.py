import argparse
from src.aggregator import summarizer
from src.display_codes import plotter

def main(args):
    if args.methods == "all":
        methods = ["SLQ", "BKDE"]
    else:
        methods = [args.methods]
    errors = {}
    p20s = {}
    p80s = {}
    blocks = {}
    for m in methods:
        errors[m], p20s[m], p80s[m], blocks[m] = summarizer(args.dataset, m)
    plotter(errors, p20s, p80s, blocks, methods, args.dataset)
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
                        default="SLQ", 
                        required=False,
                        choices=["SLQ", "BKDE", "all"],
                        help="choose methdos to compare here")
    args = parser.parse_args()
    print(args)
    main(args)

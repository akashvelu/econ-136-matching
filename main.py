import argparse

from matching.greedy import GreedyMatching


def parse_args():
    # TODO: can add even more arguments for ablation

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matching-algo",
        "-ma",
        type=str,
        default="greedy",
        choices=["greedy", "da"],
    )
    parser.add_argument(
        "--num-students", "-ns", type=int, default=100,
    )
    parser.add_argument(
        "--num-tutors", "-nt", type=int, default=100,
    )
    parser.add_argument(
        "--dim", "-d", type=int, default=100,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--student-pref-file", "-spf", type=str, default="")
    parser.add_argument("--tutor-pref-file", "-tpf", type=str, default="")

    args = parser.parse_args()

    return args


def main(args):
    raise NotImplementedError
    

if __name__ == "__main__":
    args = parse_args()
    main(args)

import argparse

from matching.greedy import GreedyMatching
from utils import generate_embeddings, get_cosine_similarities, get_oracle_pref
from matching.participants import Student, Tutor
import numpy as np
import os
import pickle

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
        "--tutor-num-slots", "-nts", type=int, default=1,
    )
    parser.add_argument(
        "--dim", "-d", type=int, default=100,
    )
    parser.add_argument(
        "--noise-k", "-nk", type=int, default=5,
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--student-pref-file", "-spf", type=str, default="")
    parser.add_argument("--tutor-pref-file", "-tpf", type=str, default="")

    args = parser.parse_args()

    return args


def run_greedy_matching(args, load_data_path=None, save_data=False):
    """
    Run greedy matching based on the setting specified in args.
    If save_data is True, save embeddings, similarity matrix, preferences, and the final match.
    TODO: if load_data_path is specified, load in the relevant information rather than re-initializing.
    """
    seed = args.seed
    num_students = args.num_students
    num_tutors = args.num_tutors
    num_tutor_slots = args.tutor_num_slots
    embed_dim = args.dim
    noise_k = args.noise_k

    student_embeds = generate_embeddings(num_students, embed_dim, store=False)
    tutor_embeds = generate_embeddings(num_tutors, embed_dim, store=False)

    sim_mat = get_cosine_similarities(student_embeds, tutor_embeds)

    oracle_student_prefs = get_oracle_pref(sim_mat)
    oracle_tutor_prefs = get_oracle_pref(sim_mat.T)

    students = [Student(i, student_embeds[i]) for i in range(num_students)]
    tutors = [Tutor(i, tutor_embeds[i], num_tutor_slots) for i in range(num_tutors)]

    student_prefs = np.stack([s.init_ranking_list for s in students])
    tutor_prefs = np.stack([t.init_ranking_list for t in tutors])

    greedy_matching = GreedyMatching(students, tutors, oracle_student_pref=oracle_student_prefs, noise_k=noise_k)
    greedy_match = greedy_matching.match()

    if save_data:
        # save embeddings, sim_mat, oracle_prefs, perturbed prefs, and the match
        exp_dir = "greedy_ns_" + str(num_students) + "_nt_" + str(num_tutors) + "_nts_" + str(num_tutor_slots) + "_ed_" + str(embed_dim) + "_nk_" + str(noise_k) + "_seed_" + str(seed)
        data_dir = os.getcwd() + '/data/' + exp_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(data_dir + "/student_embeds.npy", "wb") as f:
            np.save(f, student_embeds)
        with open(data_dir + "/tutor_embeds.npy", "wb") as f:
            np.save(f, tutor_embeds)
        with open(data_dir + "/student_oracle_prefs.npy", "wb") as f:
            np.save(f, oracle_student_prefs)
        with open(data_dir + "/tutor_oracle_prefs.npy", "wb") as f:
            np.save(f, oracle_tutor_prefs)
        with open(data_dir + "/perturbed_student_prefs.npy", "wb") as f:
            np.save(f, student_prefs)
        with open(data_dir + "/perturbed_tutor_prefs.npy", "wb") as f:
            np.save(f, tutor_prefs)
        with open(data_dir + "/sim_mat.npy", "wb") as f:
            np.save(f, sim_mat)
        with open(data_dir + "/match.pkl", "wb") as f:
            pickle.dump(greedy_match.matches, f)
        with open(data_dir + "/args.pkl", "wb") as f:
            pickle.dump(vars(args), f)

    return greedy_match


def main(args):
    match = run_greedy_matching(args, save_data=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import generate_embeddings
from utils import get_cosine_similarities
from utils import get_oracle_pref
from matching.participants import Student, Tutor
from matching.greedy import GreedyMatching
from matching.sampler import SamplerMatching
from matching.deferred_acceptance import DAAMatching


def plot(results, nts, title_sub):
    x = list(results[0][0]["greedy"].keys())
    # BP plot
    greedy_data = defaultdict(list)
    sampler_data = defaultdict(list)
    da_data = defaultdict(list)
    for result in results:
        for key in x:
            greedy_data[key].append(result[0]["greedy"][key])
            sampler_data[key].append(result[0]["sampler"][key])
            da_data[key].append(result[0]["da"][key])
    greedy_y = [np.mean(greedy_data[key]) for key in x]
    greedy_err = [np.std(greedy_data[key]) for key in x]
    sampler_y = [np.mean(sampler_data[key]) for key in x]
    sampler_err = [np.std(sampler_data[key]) for key in x]
    da_y = [np.mean(da_data[key]) for key in x]
    da_err = [np.std(da_data[key]) for key in x]
    plt.errorbar(x, greedy_y, yerr=greedy_err, label="Greedy")
    plt.errorbar(x, sampler_y, yerr=sampler_err, label="Sampler(5)")
    plt.errorbar(x, da_y, yerr=da_err, label="DA")
    plt.xlabel("(Num Students, Num Tutors)")
    plt.ylabel("Number of blocking pairs")
    plt.title(f"Number of blocking pairs vs growing {title_sub} (NTS={nts})")
    if title_sub == "matching pool":
        plt.savefig(f"plots/MP_bp_nts={nts}.png")
    if title_sub == "student/tutor ratio":
        plt.savefig(f"plots/STR_bp_nts={nts}.png")
    plt.clf()

    # Mean TL plot
    greedy_data = defaultdict(list)
    sampler_data = defaultdict(list)
    da_data = defaultdict(list)
    for result in results:
        for key in x:
            greedy_data[key].append(result[1]["greedy"][key])
            sampler_data[key].append(result[1]["sampler"][key])
            da_data[key].append(result[1]["da"][key])
    greedy_y = [np.mean(greedy_data[key]) for key in x]
    greedy_err = [np.std(greedy_data[key]) for key in x]
    sampler_y = [np.mean(sampler_data[key]) for key in x]
    sampler_err = [np.std(sampler_data[key]) for key in x]
    da_y = [np.mean(da_data[key]) for key in x]
    da_err = [np.std(da_data[key]) for key in x]
    plt.errorbar(x, greedy_y, yerr=greedy_err, label="Greedy")
    plt.errorbar(x, sampler_y, yerr=sampler_err, label="Sampler(5)")
    plt.errorbar(x, da_y, yerr=da_err, label="DA")
    plt.xlabel("(Num Students, Num Tutors)")
    plt.ylabel("Mean Tutor Load")
    plt.title(f"Mean Tutor Load vs growing matching pool (NTS={nts})")
    if title_sub == "matching pool":
        plt.savefig(f"plots/MP_mean_tl_nts={nts}.png")
    if title_sub == "student/tutor ratio":
        plt.savefig(f"plots/STR_mean_tl_nts={nts}.png")
    plt.clf()

    # Max TL plot
    greedy_data = defaultdict(list)
    sampler_data = defaultdict(list)
    da_data = defaultdict(list)
    for result in results:
        for key in x:
            greedy_data[key].append(result[2]["greedy"][key])
            sampler_data[key].append(result[2]["sampler"][key])
            da_data[key].append(result[2]["da"][key])
    greedy_y = [np.mean(greedy_data[key]) for key in x]
    greedy_err = [np.std(greedy_data[key]) for key in x]
    sampler_y = [np.mean(sampler_data[key]) for key in x]
    sampler_err = [np.std(sampler_data[key]) for key in x]
    da_y = [np.mean(da_data[key]) for key in x]
    da_err = [np.std(da_data[key]) for key in x]
    plt.errorbar(x, greedy_y, yerr=greedy_err, label="Greedy")
    plt.errorbar(x, sampler_y, yerr=sampler_err, label="Sampler(5)")
    plt.errorbar(x, da_y, yerr=da_err, label="DA")
    plt.xlabel("(Num Students, Num Tutors)")
    plt.ylabel("Max Tutor Load")
    plt.title(f"Max Tutor Load vs growing matching pool (NTS={nts})")
    if title_sub == "matching pool":
        plt.savefig(f"plots/MP_max_tl_nts={nts}.png")
    if title_sub == "student/tutor ratio":
        plt.savefig(f"plots/STR_max_tl_nts={nts}.png")
    plt.clf()


def init_matching(ns, nt, nts, dim):
    student_embeds = generate_embeddings(ns, dim, store=False)
    tutor_embeds = generate_embeddings(nt, dim, store=False)

    sim_mat = get_cosine_similarities(student_embeds, tutor_embeds)

    oracle_student_prefs = get_oracle_pref(sim_mat)
    oracle_tutor_prefs = get_oracle_pref(sim_mat.T)

    students = [
        Student(i, student_embeds[i], oracle_student_prefs[i])
        for i in range(ns)
    ]

    # Sample tutor slots uniform random. This mimics real life better
    tutors = [
        Tutor(
            i,
            tutor_embeds[i],
            oracle_tutor_prefs[i],
            np.random.uniform(1, nts + 1),
        )
        for i in range(nt)
    ]

    return students, tutors, oracle_student_prefs, oracle_tutor_prefs


def run_matcher(
    matching_algo,
    students,
    tutors,
    oracle_student_prefs,
    oracle_tutor_prefs,
    num_samples,
    noise_k,
):
    if matching_algo == "greedy":
        matcher = GreedyMatching(
            students,
            tutors,
            oracle_student_pref=oracle_student_prefs,
            noise_k=noise_k,
        )
    elif matching_algo == "sampler":
        matcher = SamplerMatching(
            students,
            tutors,
            num_samples=num_samples,
            oracle_student_pref=oracle_student_prefs,
            noise_k=noise_k,
        )
    elif matching_algo == "da":
        matcher = DAAMatching(
            students,
            tutors,
            oracle_student_pref=oracle_student_prefs,
            oracle_tutor_pref=oracle_tutor_prefs,
            noise_k=noise_k,
        )
    matches = matcher.match()
    blocking_pairs = matches.get_blocking_pairs()
    tutor_load_stats = matches.get_tutor_load_stats()
    return (
        len(blocking_pairs),
        tutor_load_stats["mean"],
        tutor_load_stats["max"],
    )


def run_experiment(ns_list, nt_list, nts, dim=100):
    bp = {"greedy": {}, "sampler": {}, "da": {}}
    mean_tl = {"greedy": {}, "sampler": {}, "da": {}}
    max_tl = {"greedy": {}, "sampler": {}, "da": {}}
    for ns, nt in zip(ns_list, nt_list):
        (
            students,
            tutors,
            oracle_student_prefs,
            oracle_tutor_prefs,
        ) = init_matching(ns, nt, nts, dim)

        # Set a reasonable noise threshold and sample size
        num_samples = 5
        noise_k = 5

        greedy_results = run_matcher(
            "greedy",
            students,
            tutors,
            oracle_student_prefs,
            oracle_tutor_prefs,
            num_samples,
            noise_k,
        )
        bp["greedy"][(ns, nt)] = greedy_results["bp"]
        mean_tl["greedy"][(ns, nt)] = greedy_results["mean_tl"]
        max_tl["greedy"][(ns, nt)] = greedy_results["max_tl"]

        sampler_results = run_matcher(
            "sampler",
            students,
            tutors,
            oracle_student_prefs,
            oracle_tutor_prefs,
            num_samples,
            noise_k,
        )
        bp["sampler"][(ns, nt)] = sampler_results["bp"]
        mean_tl["sampler"][(ns, nt)] = sampler_results["mean_tl"]
        max_tl["sampler"][(ns, nt)] = sampler_results["max_tl"]

        da_results = run_matcher(
            "da",
            students,
            tutors,
            oracle_student_prefs,
            oracle_tutor_prefs,
            num_samples,
            noise_k,
        )
        bp["da"][(ns, nt)] = da_results["bp"]
        mean_tl["da"][(ns, nt)] = da_results["mean_tl"]
        max_tl["da"][(ns, nt)] = da_results["max_tl"]

    return bp, mean_tl, max_tl


def run_seeds(ns_list, nt_list, nts, title_sub, seeds=[1, 2, 3], dim=100):
    results = []
    for seed in seeds:
        # Set seed
        np.random.seed(seed)
        # Run the experiment and get results
        result = run_experiment(ns_list, nt_list, nts, dim)
        results.append(result)

    # Plot results
    plot(results, nts, title_sub)


def run_matching_pool_exp():
    ns_list = [10, 20, 50, 100, 200]
    nt_list = [10, 20, 50, 100, 200]
    seeds = [1, 2, 3]
    nts_list = [3, 5]
    dim = 100
    title_sub = "matching pool"

    for nts in nts_list:
        run_seeds(ns_list, nt_list, nts, title_sub, seeds, dim)


def run_ratios_exp():
    ns_list = [50, 50, 50, 50]
    nt_list = [100, 50, 40, 20]
    seeds = [1, 2, 3]
    nts_list = [3, 5]
    dim = 100
    title_sub = "student/tutor ratio"

    for nts in nts_list:
        run_seeds(ns_list, nt_list, nts, title_sub, seeds, dim)

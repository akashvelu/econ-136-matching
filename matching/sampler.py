import pickle
import numpy as np

from utils import get_permutation
from matching.matches import Matches


class SamplerMatching:
    """
    Class to perform Sampler Matching and related utility
    methods.
    Sampler matching shows students a set of tutors and students
    select a match at "random" from this set.
    Randomness is skewed to higher ranked tutors.
    """

    def __init__(
        self,
        students,
        tutors,
        num_samples,
        oracle_student_pref=None,
        student_pref_file=None,
        noise_k=5,
    ):
        self.students = students
        self.tutors = tutors
        self.num_samples = num_samples

        assert self.num_samples in [5, 10]

        if self.num_samples == 5:
            self.sample_weights = [0.6, 0.25, 0.1, 0.025, 0.025]

        if self.num_samples == 10:
            self.sample_weights = [
                0.3,
                0.2,
                0.1,
                0.075,
                0.075,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
            ]

        assert sum(self.sample_weights) == 1.0

        if oracle_student_pref is None:
            assert student_pref_file is not None
            oracle_student_pref = pickle.load(open(student_pref_file, "rb"))

        self._get_preferences(oracle_student_pref, noise_k)

        self.matches = Matches(students, tutors)

    def _get_preferences(self, oracle_student_pref, noise_k):
        for student in self.students:
            ranking_list = get_permutation(
                oracle_student_pref[student.id], noise_k
            )
            student.set_noisy_rankings(ranking_list)

    def match(self):
        for student in self.students:
            # Find the top num_samples tutors who
            # still have an open slot.
            tutor_showcase = []
            for tutor_id in student.init_ranking_list:
                if self.tutors[tutor_id].has_slots():
                    tutor_showcase.append(tutor_id)
                if len(tutor_showcase) == self.num_samples:
                    break

            print(student.id, tutor_showcase)

            # Sample a tutor from this list with weights
            tutor_sample = np.random.choice(
                a=tutor_showcase, p=self.sample_weights[: len(tutor_showcase)]
            )
            # Assign the match
            student.current_match = tutor_sample
            self.tutors[tutor_sample].current_matches.append(student.id)
            self.matches.add_match(student.id, tutor_sample)

        return self.matches

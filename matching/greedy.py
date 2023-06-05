import pickle

from utils import get_permutation
from matching.matches import Matches


class GreedyMatching:
    """
    Class to perform Greedy Matching and related utility
    methods.
    """

    def __init__(
        self,
        students,
        tutors,
        oracle_student_pref=None,
        student_pref_file=None,
        noise_k=5,
    ):
        self.students = students
        self.tutors = tutors

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
            for tutor_id in student.init_ranking_list:
                tutor = self.tutors[tutor_id]
                if tutor.has_slots():
                    student.current_match = tutor_id
                    tutor.current_matches.append(student.id)
                    self.matches.add_match(student.id, tutor_id)
                    break

        return self.matches

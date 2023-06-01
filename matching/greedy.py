import pickle

from ..utils import get_permutation
from matches import Matches


class GreedyMatching:
    """
    Class to perform Greedy Matching and related utility
    methods.
    """

    def __init__(self, students, tutors, student_pref_file, noise_k=5):
        self.students = students
        self.tutors = tutors

        oracle_student_pref = pickle.load(open(student_pref_file, "rb"))
        self._get_preferences(oracle_student_pref, noise_k)

        self.matches = Matches()

    def _get_preferences(self, oracle_student_pref, noise_k):
        for student in self.students:
            ranking_list = get_permutation(
                oracle_student_pref[student.id], noise_k
            )
            student.set_init_rankings(ranking_list)

    def match(self):
        for student in self.students:
            for tutor_id in student.ranking_list:
                tutor = self.tutors[tutor_id]
                if tutor.has_slots():
                    student.current_match = tutor_id
                    tutor.current_matches.append(student.id)
                    self.matches.add_match(student.id, tutor_id)
                    break

        return self.matches

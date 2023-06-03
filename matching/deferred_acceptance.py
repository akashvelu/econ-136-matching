import pickle

from utils import get_permutation
from matching.matches import Matches


class DAAMatching:
    """
    Class to perform Greedy Matching and related utility
    methods.
    """

    def __init__(self, students, tutors, oracle_student_pref=None, oracle_tutor_pref=None,
                 student_pref_file=None, tutor_pref_file=None, noise_k=5):

        self.students = students
        self.tutors = tutors

        if oracle_student_pref is None:
            assert student_pref_file is not None
            oracle_student_pref = pickle.load(open(student_pref_file, "rb"))

        if oracle_tutor_pref is None:
            assert tutor_pref_file is not None
            oracle_tutor_pref = pickle.load(open(tutor_pref_file, "rb"))

        self._get_preferences(oracle_student_pref, noise_k, oracle_tutor_pref)

        self.matches = Matches()

    def _get_preferences(self, oracle_student_pref, noise_k, oracle_tutor_pref):
        for student in self.students:
            ranking_list = get_permutation(
                oracle_student_pref[student.id], noise_k
            )
            student.set_initial_rankings(ranking_list)

        for tutor in self.tutors:
            # No perturbations to tutor preferences (for now).
            tutor.set_initial_rankings(oracle_tutor_pref[tutor.id])

    def match(self):
        # Run deferred acceptance.
        all_matched = False
        while not all_matched:
            all_matched = True
            for student in self.students:
                if student.is_matched():
                    continue
                most_preferred_tutor_id = student.pop()
                most_preferred_tutor = self.tutors[most_preferred_tutor_id]
                # This function will already add student to the tutor matches, if accepted.
                accepted, replaced_student_id = most_preferred_tutor.process_proposal(student.id)
                if accepted:
                    student.current_match = most_preferred_tutor_id
                else:
                    all_matched = False
                if replaced_student_id is not None:
                    replaced_student = self.students[replaced_student_id]
                    # Replaced student is no longer matched.
                    replaced_student.current_match = None
                    all_matched = False

        # Finalize the matching.
        for student in self.students:
            assert student.current_match is not None
            self.matches.add_match(student.id, student.current_match)

        return self.matches

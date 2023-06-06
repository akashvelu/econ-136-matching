import numpy as np


class Participant:
    """
    Class that defines useful attributes and methods for
    students and tutors in the matching market
    """

    def __init__(self, id, embedding, oracle_ranking_list):
        self.id = id
        self.embedding = embedding
        self.oracle_ranking_list = oracle_ranking_list
        # These are potentially noisy versions
        self.init_ranking_list = None
        self.current_ranking_list = (
            None  # ranking list that will get modified during DAA
        )

    def set_noisy_rankings(self, ranking_list):
        assert self.init_ranking_list is None
        assert self.current_ranking_list is None
        self.init_ranking_list = ranking_list
        self.current_ranking_list = ranking_list

    def pop(self):
        x = self.current_ranking_list[0]
        self.current_ranking_list = self.current_ranking_list[1:]
        return x

    def top(self):
        return self.current_ranking_list[0]

    def get_oracle_rank(self, partner_idx):
        return np.where(self.oracle_ranking_list == partner_idx)[0][0]

    def get_noisy_rank(self, partner_idx):
        return np.where(self.init_ranking_list == partner_idx)[0][0]


class Student(Participant):
    """
    Class that defines useful attributes and methods
    for students in the matching market
    """

    def __init__(self, id, embedding, oracle_ranking_list):
        super().__init__(id, embedding, oracle_ranking_list)
        self.current_match = None

    def is_matched(self):
        return not (self.current_match is None)

    def reset(self):
        self.init_ranking_list = None
        self.current_ranking_list = None
        self.current_match = None


class Tutor(Participant):
    """
    Class that defines useful attributes and methods
    for tutors in the matching market
    """

    def __init__(self, id, embedding, oracle_ranking_list, num_slots):
        super().__init__(id, embedding, oracle_ranking_list)
        self.num_slots = num_slots
        self.current_matches = []

    @property
    def num_filled(self):
        return len(self.current_matches)

    def has_slots(self):
        return len(self.current_matches) < self.num_slots

    def reset(self):
        self.init_ranking_list = None
        self.current_ranking_list = None
        self.current_matches = []

    def process_proposal(self, student_id):
        # If the tutor has available slots, the student proposal can be accepted.
        if self.has_slots():
            self.current_matches.append(student_id)
            return True, None

        # If all slots are full, see if the proposing student is preferred over any existing match.
        # If so, replace the worst existing match with the proposing student.
        student_pref = self.get_noisy_rank(student_id)

        # Find the least-preferred student among current matches
        worst_match = -1
        worst_match_idx = None
        for i in range(len(self.current_matches)):
            match = self.current_matches[i]
            match_ind = self.get_noisy_rank(match)
            # a match is worse if the student appears later in the preference list.
            if match_ind > worst_match:
                worst_match = match_ind
                worst_match_idx = i

        # Check if the proposing student is preferred over the worst match; if so, replace the worst match with this student.
        if student_pref < worst_match:
            replaced_student_id = self.current_ranking_list[worst_match_idx]
            self.current_matches[worst_match_idx] = student_id
            return True, replaced_student_id
        else:
            return False, None

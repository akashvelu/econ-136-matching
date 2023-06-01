class Participant:
    """
    Class that defines useful attributes and methods for
    students and tutors in the matching market
    """

    def __init__(self, id, embedding):
        self.id = id
        self.embedding = embedding
        self.init_ranking_list = None
        self.current_ranking_list = None

    def set_initial_rankings(self, ranking_list):
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


class Student(Participant):
    """
    Class that defines useful attributes and methods
    for students in the matching market
    """

    def __init__(self, id, embedding, ranking_list=[]):
        super().__init__(id, embedding, ranking_list)
        self.current_match = None

    def is_matched(self):
        return not (self.current_match is None)


class Tutor(Participant):
    """
    Class that defines useful attributes and methods
    for tutors in the matching market 
    """

    def __init__(self, id, embedding, num_slots, ranking_list=[]):
        super().__init__(id, embedding, ranking_list)
        self.num_slots = num_slots
        self.current_matches = []

    @property
    def num_filled(self):
        return len(self.current_matches)

    def has_slots(self):
        return len(self.current_matches) < self.num_slots

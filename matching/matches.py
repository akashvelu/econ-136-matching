import pickle


class Matches:
    """
    Class that defines a matching and includes useful
    attributes and methods for the same.
    The `matching` dictionary holds 
    `student_idx:tutor_idx` matches
    """

    def __init__(self):
        self.matches = {}

    def get_blocking_pairs(self):
        raise NotImplementedError

    def add_match(self, student_id, tutor_id):
        self.matches[student_id] = tutor_id

    def get_tutor(self, student_idx):
        return self.matches[student_idx]

    def get_students(self, tutor_index):
        s = []

        for k, v in self.matches.items():
            if v == tutor_index:
                s.append(k)

        return s

    def is_efficient(self):
        raise NotImplementedError

    @property
    def num_matched_students(self):
        return len(self.matches)

    @property
    def num_matched_tutors(self):
        return len(set(self.matches.values()))

    def dump(self, filename):
        if filename.endswith(".csv"):
            with open(filename, "w") as fp:
                for key, value in self.matches.items():
                    fp.write(f"{key},{value}\n")
        else:
            pickle.dump(self.matches, open(filename, "wb"))

import pickle


class Matches:
    """
    Class that defines a matching and includes useful
    attributes and methods for the same.
    The `matching` dictionary holds
    `student_idx:tutor_idx` matches
    """

    def __init__(self, students, tutors):
        self.students = students
        self.tutors = tutors
        self.matches = {}

    def get_blocking_pairs(self):
        blocking_pairs = []
        for student in self.students:
            for tutor in self.tutors:
                # if there is a tutor the student prefers more to their current match
                if student.get_rank(tutor.id) < student.get_rank(
                    student.current_match
                ):
                    # check if the tutor prefers this student more to any of their current matches
                    for match in tutor.current_matches:
                        if tutor.get_rank(student.id) < tutor.get_rank(match):
                            # if yes, we have a blocking pair
                            blocking_pairs.append((student.id, tutor.id))

        return blocking_pairs

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

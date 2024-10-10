class Constraint:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def __str__(self):
        return "Constraint: " + self.name

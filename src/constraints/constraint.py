class Constraint:
    def __init__(self, name, constraint_type, constraint_value):
        self.name = name

    def __str__(self):
        return "Constraint: " + self.name

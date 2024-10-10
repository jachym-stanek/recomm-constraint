
class Algorithm:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def run(self, data):
        pass

    def __str__(self):
        return self.name + ': ' + self.description
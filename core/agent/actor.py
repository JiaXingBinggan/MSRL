"""
Implementation of general actor class
"""


class Actor:
    def __init__(self):
        super(Actor, self).__init__()

    def act(self, params):
        raise NotImplementedError("Method of generating action should be overridden by subclass.")

    def collect_transitions(self, transitions):
        raise NotImplementedError("Method of collecting transitions should be overridden by subclass.")
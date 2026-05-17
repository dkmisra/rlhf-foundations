class Dyck:
    
    def __init__(self, max_length=10):
        self.max_length = max_length

    def reset(self):
        self.state = ""

    def step(self, action):
        self.state += action
        return self.state

    def is_done(self):
        return len(self.state) == self.max_length
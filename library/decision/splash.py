import random

class Splash:
    # Splash is a random move..
    def __init__(self):
      pass
        
    def Decide(self, B, colour):
        return random.choice(B.get_moves())
    
    def Decide(self, B):
        return random.choice(B.get_moves())


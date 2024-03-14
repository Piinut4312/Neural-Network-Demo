"""
This class defines several learning rate schedulers for training nerual networks
"""

class LRScheduler:

    def __init__(self, learning_rate=0.5):
        self.steps = 0
        self.learning_rate = learning_rate

    def step(self):
        self.steps += 1
        return self.learning_rate
    
class ReciprocalLRScheduler(LRScheduler):

    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self):
        self.steps += 1
        return self.learning_rate/self.steps
    
class DecayLRScheduler(LRScheduler):

    def __init__(self, learning_rate, tao=1):
        super().__init__(learning_rate)
        self.tao = tao

    def step(self):
        self.steps += 1
        return self.learning_rate/(1+self.steps/self.tao)
    

LR_SCHEDULERS = {"Constant": LRScheduler, "Reciprocal": ReciprocalLRScheduler}
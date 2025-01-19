from brain_interface import SpaceshipBrain, Action, GameState
import math
import random

class RandomBrain(SpaceshipBrain):
    def __init__(self):
        self.current_action = None
        self.action_counter = 0
        self.action_duration = 5  # Number of frames to keep the same action

    @property
    def id(self) -> str:
        return "Random"

    def decide_what_to_do_next(self, game_state: GameState) -> Action:
        # Choose new random action after action_duration frames
        if self.current_action is None or self.action_counter >= self.action_duration:
            self.current_action = random.choice(list(Action))
            self.action_counter = 0
        
        self.action_counter += 1
        return self.current_action

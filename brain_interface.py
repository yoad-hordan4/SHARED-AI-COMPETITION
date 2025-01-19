from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict

class Action(Enum):
    ROTATE_RIGHT = 1
    ROTATE_LEFT = 2
    ACCELERATE = 3
    BRAKE = 4
    SHOOT = 5

@dataclass
class GameState:
    ships: List[Dict]            # List of ship info including position, angle, health, score
    bullets: List[Dict]          # List of bullet info including position, angle, owner_id
    gold_positions: List[Tuple[float, float]]
    asteroids: List[Dict]        # New: List of asteroid info including position and radius
    game_ticks: int              # Current game ticks
class SpaceshipBrain:
    @property
    def id(self) -> str:
        raise NotImplementedError()

    def decide_what_to_do_next(self, game_state: GameState) -> Action:
        raise NotImplementedError()
        
    def on_game_complete(self, final_state: GameState, won: bool):
        """Called when a game completes. Brains can implement this to handle end-of-game logic"""
        pass

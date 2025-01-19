from brain_interface import SpaceshipBrain, Action, GameState
import math
import time

class AggressiveHunterBrain(SpaceshipBrain):
    def __init__(self):
        self._id = "group2-Yoad-H"
        self.current_target_id = None
        self.optimal_range = 100  

    @property
    def id(self) -> str:
        return self._id

    def decide_what_to_do_next(self, game_state: GameState) -> Action:
        try:
            # Locate my ship
            my_ship = next(ship for ship in game_state.ships if ship['id'] == self.id)
        except StopIteration:
            return Action.ROTATE_RIGHT  # Default action if my ship is not found

        current_time = time.time()

        # Initialize health and movement time if not already set
        if not hasattr(self, 'last_health'):
            self.last_health = my_ship['health']
        if not hasattr(self, 'last_movement_time'):
            self.last_movement_time = current_time

        # Handle being hit: Move immediately if health decreases
        if my_ship['health'] < self.last_health:
            self.last_health = my_ship['health']
            self.last_movement_time = current_time
            return Action.ACCELERATE if my_ship['health'] % 2 == 0

        # Identify active enemies
        enemy_ships = [ship for ship in game_state.ships if ship['id'] != self.id and ship['health'] > 0]
        if not enemy_ships:
            self.current_target_id = None
            return Action.ROTATE_RIGHT  # Default if no enemies are left

        # Find the closest enemy
        closest_enemy = min(enemy_ships, key=lambda ship: math.hypot(ship['x'] - my_ship['x'], ship['y'] - my_ship['y']))
        self.current_target_id = closest_enemy['id']

        # Calculate relative position and angle to the target
        dx = closest_enemy['x'] - my_ship['x']
        dy = closest_enemy['y'] - my_ship['y']
        distance = math.hypot(dx, dy)
        target_line_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = (target_line_angle - my_ship['angle'] + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360

        # Shooting logic: shoot if the target is well-aligned
        angle_tolerance = 6  # Reduced tolerance for better precision
        if abs(angle_diff) < angle_tolerance:
            return Action.SHOOT

        # Handle inactivity: move if stagnant for 5+ seconds
        if current_time - self.last_movement_time > 5:
            self.last_movement_time = current_time
            return Action.ACCELERATE if distance > self.optimal_range else Action.ROTATE_LEFT

        # Movement logic: align precisely or close the gap
        if distance > self.optimal_range * 0.7:  # If far from the target
            if abs(angle_diff) > angle_tolerance:  # Rotate toward target
                self.last_movement_time = current_time
                return Action.ROTATE_RIGHT if angle_diff > 0 else Action.ROTATE_LEFT
            self.last_movement_time = current_time
            return Action.ACCELERATE

        # Default to aligning with the target when near the optimal range
        self.last_movement_time = current_time
        return Action.ROTATE_RIGHT if angle_diff > 0 else Action.ROTATE_LEFT

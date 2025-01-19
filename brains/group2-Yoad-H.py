from brain_interface import SpaceshipBrain, Action, GameState
import math

class AggressiveHunterBrain(SpaceshipBrain):
    def __init__(self):
        self._id = "group2-Yoad-H"
        self.current_target_id = None
        self.optimal_range = 100  

    @property
    def id(self) -> str:
        return self._id

    def decide_what_to_do_next(self, game_state: GameState) -> Action:
        # Find my ship
        try:
            my_ship = next(ship for ship in game_state.ships if ship['id'] == self.id)
        except StopIteration:
            return Action.ROTATE_RIGHT  # Default action if my ship isn't found

        # Find all enemy ships that aren't destroyed (health > 0)
        enemy_ships = [ship for ship in game_state.ships if ship['id'] != self.id and ship['health'] > 0]

        if not enemy_ships:
            self.current_target_id = None  # Reset target if no enemies are left
            return Action.ROTATE_RIGHT

        # Select the closest enemy ship
        closest_enemy = min(enemy_ships, key=lambda ship: math.hypot(ship['x'] - my_ship['x'], ship['y'] - my_ship['y']))
        self.current_target_id = closest_enemy['id']

        # Calculate relative position to the closest enemy
        dx = closest_enemy['x'] - my_ship['x']
        dy = closest_enemy['y'] - my_ship['y']
        distance = math.hypot(dx, dy)

        # Determine the angle to the target
        target_line_angle = math.degrees(math.atan2(dy, dx))

        # Calculate the angle difference and normalize it to -180 to 180
        angle_diff = (target_line_angle - my_ship['angle'] + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360

        # Shooting logic: shoot if the target is roughly aligned
        angle_tolerance = 20  # Increase tolerance for shooting
        if abs(angle_diff) < angle_tolerance:
            return Action.SHOOT

        # Aggressive movement logic: continuous motion
        if distance > self.optimal_range * 0.7:  # If far from the target
            if abs(angle_diff) > 20:  # Rotate toward the target
                if angle_diff > 0:
                    return Action.ROTATE_RIGHT
                else:
                    return Action.ROTATE_LEFT
            return Action.ACCELERATE  # Move closer

        if distance < self.optimal_range * 0.5:  # If too close, evade by rotating and moving
            if angle_diff > 0:
                return Action.ROTATE_LEFT  # Rotate left to change position
            else:
                return Action.ROTATE_RIGHT

        # Default to moving and rotating when near optimal range
        if abs(angle_diff) > 20:
            if angle_diff > 0:
                return Action.ROTATE_RIGHT
            else:
                return Action.ROTATE_LEFT

        return Action.ACCELERATE
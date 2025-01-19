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

        # Prioritize targets: closest or lowest health
        current_target = next((ship for ship in enemy_ships if ship['id'] == self.current_target_id), None)
        if not current_target or current_target['health'] <= 0:
            current_target = min(
                enemy_ships,
                key=lambda ship: (
                    math.hypot(ship['x'] - my_ship['x'], ship['y'] - my_ship['y']),  # Closest first
                    -ship['health']  # Then prioritize lowest health
                )
            )
            self.current_target_id = current_target['id']

        # Calculate relative position to target
        dx = current_target['x'] - my_ship['x']
        dy = current_target['y'] - my_ship['y']
        distance = math.hypot(dx, dy)

        # Predictive targeting: estimate future position
        target_speed = current_target.get('speed', 0)
        target_angle = math.radians(current_target.get('angle', 0))
        future_x = current_target['x'] + target_speed * math.cos(target_angle)
        future_y = current_target['y'] + target_speed * math.sin(target_angle)
        dx = future_x - my_ship['x']
        dy = future_y - my_ship['y']
        target_line_angle = math.degrees(math.atan2(dy, dx))

        # Calculate angle difference and normalize to -180 to 180
        angle_diff = (target_line_angle - my_ship['angle'] + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360

        # Shooting logic: broaden shooting range and prioritize firing
        angle_tolerance = 20  # Increased tolerance for shooting
        if abs(angle_diff) < angle_tolerance:
            return Action.SHOOT

        # Turn toward target
        if angle_diff > 0:
            return Action.ROTATE_RIGHT
        return Action.ROTATE_LEFT


#space_game.py
import pygame
import os
import importlib
import inspect
import math
import random
from brain_interface import SpaceshipBrain, Action, GameState
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np  # Import numpy for numerical operations
from helpers import cached_hypot

SPECIFIC_BRAINS_TO_RUN = [] #['Q-Learner', 'Defensive']
# Constants
TRAINING_MODE = False
TRAINING_MODE_GAMES = 100000

PLOT_UPDATE_INTERVAL = 1000
NUMBER_OF_BRAINS_TO_RUN = 7
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800

BORDER_LEFT = 50
BORDER_RIGHT = 350  # Wider right border
BORDER_TOP = 50
BORDER_BOTTOM = 50

GAME_WIDTH = SCREEN_WIDTH - BORDER_LEFT - BORDER_RIGHT
GAME_HEIGHT = SCREEN_HEIGHT - BORDER_TOP - BORDER_BOTTOM

FPS = 60
FIXED_DT = 0.016  # Fixed delta time for training mode

MAX_TICK_COUNT = 5000

ASTEROID_SPEED = 10
MAX_VELOCITY = 170
ACCELERATION = 50
FRICTION = 0.98
SHIP_BRAKE_FACTOR = 0.9  # Added brake factor
NUMBER_OF_ASTEROIDS = 13
HEALTH_FULL = 100
HEALTH_BULLET_DAMAGE = 4

GOLD_SPAWN_INTERVAL = 3000
GOLD_VALUE = 15
GOLD_SCATTER_FRACTION = 0.5  # Fraction of gold to scatter when ship is destroyed

LAST_SHIP_STANDING_MULTIPLIER_BONUS = 2  # Bonus for the last ship remaining
BULLET_HIT_SCORE = 10  # Score gained when hitting another ship
SHIP_DISTRUCTION_SCORE = 100  # Score gained when destroying another ship
SHIP_DESTROYED_ALL_SHIPS_BONUS = 20

GOLD_SCATTER_DISTANCE_MIN = 20
GOLD_SCATTER_DISTANCE_MAX = 50

INITIAL_GOLD_COUNT = 60

BULLET_SPEED = 300
BULLET_COOLDOWN = 1000
BULLET_SIZE = 3

SHIP_TURN_SPEED = 120 
SHIP_COLLISION_RADIUS = 20  # For collision detection
SHIP_COLLISION_DISTANCE = 30  # Collision radius between ships

SHIP_SIZE = 20  # For drawing
SHIP_SIDE_OFFSET = 10
SHIP_SIDE_ANGLE = 140

SHIP_HEALTH_BAR_WIDTH = 40
SHIP_HEALTH_BAR_HEIGHT = 5

SHIP_SCORE_DISPLAY_OFFSET_X = -10
SHIP_SCORE_DISPLAY_OFFSET_Y = -45

SHIP_BRAIN_ID_DISPLAY_OFFSET_X = -30
SHIP_BRAIN_ID_DISPLAY_OFFSET_Y = 20

SHIP_DESTROYED_COLOR = (128, 128, 128)
SHIP_ACTIVE_COLOR = (255, 255, 255)

GOLD_SIZE = 5  # Radius when drawing gold

BULLET_COLOR = (255, 100, 100)
GOLD_COLOR = (255, 215, 0)

LEADERBOARD_MAX_ENTRIES = 15

TRAINING_STATUS_INTERVAL = 100

IS_CONSTANT_STARTING_POSITIONS = False  # Set to True to use constant starting positions

ASTEROID_RADIUS = 35

# New: Define the Asteroid class
class Asteroid:
    def __init__(self, x: float, y: float, velocity_x: float, velocity_y: float, radius: int = ASTEROID_RADIUS):
        self.x = x
        self.y = y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.radius = radius  # Radius of the asteroid for collision and drawing

    def update_position(self, dt: float, border_left: int, border_right: int, border_top: int, border_bottom: int, screen_width: int, screen_height: int):
        self.x += self.velocity_x * dt
        self.y += self.velocity_y * dt

        # Wrap around the screen edges
        if self.x < border_left:
            self.x = screen_width - border_right - self.radius
        elif self.x > screen_width - border_right - self.radius:
            self.x = border_left + self.radius

        if self.y < border_top:
            self.y = screen_height - border_bottom - self.radius
        elif self.y > screen_height - border_bottom - self.radius:
            self.y = border_top + self.radius

class GameEnvironment:
    def __init__(self, training_mode=False):
        if not training_mode:
            pygame.init()
            self.screen_width = SCREEN_WIDTH
            self.screen_height = SCREEN_HEIGHT
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.font = pygame.font.Font(None, 36)
            
            # Ensure 'background.png' exists or handle missing file
            background_path = "background.png"
            if os.path.exists(background_path):
                self.background = pygame.image.load(background_path)
                self.background = pygame.transform.scale(self.background, (self.screen_width, self.screen_height))
            else:
                # Fill background with a solid color if image is missing
                self.background = None
                self.screen.fill((0, 0, 0))
        else:
            # Minimal initialization for training mode
            pygame.init()
            self.screen = None
            self.font = None
            self.background = None
            self.screen_width = SCREEN_WIDTH
            self.screen_height = SCREEN_HEIGHT
        self.training_mode = training_mode

class Spaceship:
    def __init__(self, brain: SpaceshipBrain, x: float, y: float):
        self.brain = brain
        self.x = x
        self.y = y
        self.velocity_x = 0
        self.velocity_y = 0
        self.max_velocity = MAX_VELOCITY
        self.acceleration = ACCELERATION
        self.friction = FRICTION
        self.angle = 0
        self.health = HEALTH_FULL
        self.score = 0
        self.gold_collected = 0
        self.id = brain.id
        self.last_shot_time = 0
        self.is_destroyed = False
        self.bullets_hit_count = 0  # New attribute to track bullet hits

class SpaceGame:
    def __init__(self, environment: GameEnvironment, wins_per_brain: dict):
        self.screen = environment.screen
        self.font = environment.font
        self.background = environment.background
        self.training_mode = environment.training_mode
        self.text_font = pygame.font.Font(None, 24)
        if not self.training_mode:
            self.clock = pygame.time.Clock()
        else:
            self.clock = None  # No need for clock in training mode

        self.ships = []
        self.gold_positions = []
        self.bullets = []
        self.asteroids = []  # New: List to hold asteroids
        self.last_gold_spawn_time = 0  # Initialize to 0 for correct spawning
        self.gold_spawn_interval = GOLD_SPAWN_INTERVAL
        self.tick_count = 0
        self.game_over = False
        self.wins_per_brain = wins_per_brain  # Reference to the shared wins counter

        # Define constants for screen and game area dimensions
        self.border_left = BORDER_LEFT
        self.border_right = BORDER_RIGHT  # Wider right border
        self.border_top = BORDER_TOP
        self.border_bottom = BORDER_BOTTOM
        self.screen_width = environment.screen_width
        self.screen_height = environment.screen_height
        self.game_width = GAME_WIDTH
        self.game_height = GAME_HEIGHT
        self.bonus_awarded = False  # Initialize the bonus flag

        # Initialize game_time to track elapsed game time in milliseconds
        self.game_time = 0

        # Generate starting positions if constant starting positions are enabled
        if IS_CONSTANT_STARTING_POSITIONS:
            starting_pos_rng = random.Random(42)  # Fixed seed for consistency
            self.starting_positions = [
                (
                    starting_pos_rng.randint(self.border_left + SHIP_SIZE, self.screen_width - self.border_right - SHIP_SIZE),  # Adjusted for SHIP_SIZE
                    starting_pos_rng.randint(self.border_top + SHIP_SIZE, self.screen_height - self.border_bottom - SHIP_SIZE)   # Adjusted for SHIP_SIZE
                )
                for _ in range(NUMBER_OF_BRAINS_TO_RUN)
            ]
        else:
            self.starting_positions = None

        self.load_brains()
        self.spawn_initial_gold()
        self.spawn_initial_asteroids()  # New: Spawn initial asteroids

    def load_brains(self):
        brains_dir = "brains"
        starting_pos_index = 0  # Initialize index for starting positions

        if not os.path.isdir(brains_dir):
            print(f"Brains directory '{brains_dir}' not found.")
            return

        for file in sorted(os.listdir(brains_dir)):
            if file.endswith(".py"):
                module_name = file[:-3]
                try:
                    module = importlib.import_module(f"brains.{module_name}")
                except Exception as e:
                    print(f"Error importing module '{module_name}': {e}")
                    continue

                # Find brain classes in the module
                for name, obj in inspect.getmembers(module):
                    if (len(self.ships) < NUMBER_OF_BRAINS_TO_RUN and inspect.isclass(obj) and
                        issubclass(obj, SpaceshipBrain) and
                            obj != SpaceshipBrain):
                        try:
                            brain = obj()
                        except Exception as e:
                            print(f"Error initializing brain '{name}': {e}")
                            continue
                        # Assign starting position
                        if SPECIFIC_BRAINS_TO_RUN and brain.id not in SPECIFIC_BRAINS_TO_RUN:
                            continue
                        if IS_CONSTANT_STARTING_POSITIONS and self.starting_positions:
                            x, y = self.starting_positions[starting_pos_index]
                            starting_pos_index += 1
                        else:
                            x = random.randint(self.border_left + SHIP_SIZE, self.screen_width - self.border_right - SHIP_SIZE)  # Adjusted for SHIP_SIZE
                            y = random.randint(self.border_top + SHIP_SIZE, self.screen_height - self.border_bottom - SHIP_SIZE)    # Adjusted for SHIP_SIZE
                        ship = Spaceship(brain, x=x, y=y)
                        self.ships.append(ship)
        random.shuffle(self.ships)

    def spawn_initial_asteroids(self, number_of_asteroids: int = NUMBER_OF_ASTEROIDS):
        """Spawn a fixed number of asteroids at the start of the game."""
        for _ in range(number_of_asteroids):
            x = random.randint(self.border_left + ASTEROID_RADIUS + SHIP_SIZE, self.screen_width - self.border_right - ASTEROID_RADIUS - SHIP_SIZE)  # Adjusted to prevent spawning too close to borders
            y = random.randint(self.border_top + ASTEROID_RADIUS + SHIP_SIZE, self.screen_height - self.border_bottom - ASTEROID_RADIUS - SHIP_SIZE)  # Adjusted to prevent spawning too close to borders
            # Assign slow velocities
            velocity_x = random.uniform(-ASTEROID_SPEED, ASTEROID_SPEED)  # Pixels per second
            velocity_y = random.uniform(-ASTEROID_SPEED, ASTEROID_SPEED)  # Pixels per second
            asteroid = Asteroid(x, y, velocity_x, velocity_y)
            self.asteroids.append(asteroid)

    def run(self):
        running = True
        while running:
            if self.training_mode:
                dt = FIXED_DT  # Fixed time step (~60 FPS)
            else:
                dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            
            # Update game_time based on dt
            self.game_time += dt * 1000  # Convert dt to milliseconds

            self.tick_count += 1
            # Determine current_time based on mode
            if self.training_mode:
                current_time = self.game_time
            else:
                current_time = pygame.time.get_ticks()

            if current_time - self.last_gold_spawn_time >= self.gold_spawn_interval:
                self.spawn_gold()
                self.last_gold_spawn_time = current_time

            if not self.training_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        return None  # Exit the run method

            # Check win conditions
            alive_ships = [ship for ship in self.ships if not ship.is_destroyed]
            if self.tick_count >= MAX_TICK_COUNT or len(alive_ships) <= 1:
                
                #if (self.tick_count < MAX_TICK_COUNT):
                #    print(f"Game ended before max ticks with {len(alive_ships)} alive ships after {self.tick_count} ticks.")
                winner = self.get_winner()
                if winner:
                    winnerId = winner.id
                    self.wins_per_brain[winnerId] = self.wins_per_brain.get(winnerId, 0) + 1

                # Notify all brains about game completion
                final_state = self.create_game_state(winner)
                for ship in self.ships:
                    try:
                        ship.brain.on_game_complete(final_state, ship == winner)
                    except Exception as e:
                        print(f"Error in brain '{ship.id}' on_game_complete: {e}")

                if not self.training_mode and winner:
                    self.show_winner(winner)
                    pygame.time.wait(1000)
                running = False
                if winner and winner.score == 0:
                    print("Winner has 0 score!")
                return winner

            # Update asteroids' positions
            for asteroid in self.asteroids:
                asteroid.update_position(dt, self.border_left, self.border_right, self.border_top, self.border_bottom, self.screen_width, self.screen_height)

            for ship in self.ships:
                if not ship.is_destroyed:
                    try:
                        game_state = self.create_game_state(ship)
                        action = ship.brain.decide_what_to_do_next(game_state)
                        self.process_action(ship, action, dt, current_time)  # Pass current_time
                    except Exception as e:
                        print(f"Error processing action for brain '{ship.id}': {e}")

            self.update_bullets(dt)
            self.check_collisions()

            if not self.training_mode:
                self.draw()

    def get_winner(self):
        max_score = max(ship.score for ship in self.ships)
        top_ships = [ship for ship in self.ships if ship.score == max_score]
        return random.choice(top_ships) if top_ships else None

# In space_game.py

    def create_game_state(self, current_ship: Spaceship) -> GameState:
        ships_data = [{
            'id': ship.id,
            'x': ship.x,
            'y': ship.y,
            'angle': ship.angle,
            'velocity_x': ship.velocity_x,  # Include velocity_x
            'velocity_y': ship.velocity_y,  # Include velocity_y
            'health': ship.health,
            'score': ship.score,
            'last_shot_time': ship.last_shot_time,
            'bullets_hit_count': ship.bullets_hit_count  # Include hit count
        } for ship in self.ships]

        bullets_data = [{
            'x': bullet['x'],
            'y': bullet['y'],
            'angle': bullet['angle'],
            'owner_id': bullet['owner'].id
        } for bullet in self.bullets]

        # Include asteroids in the game state
        asteroids_data = [{
            'x': asteroid.x,
            'y': asteroid.y,
            'radius': asteroid.radius
        } for asteroid in self.asteroids]

        return GameState(
            ships=ships_data,
            bullets=bullets_data,
            gold_positions=self.gold_positions,
            asteroids=asteroids_data,
            game_ticks=self.game_time  # Ensure game_time is set correctly
        )


    def show_winner(self, winner):
        if self.screen and self.font:
            # Save screenshot
            timestamp = pygame.time.get_ticks()
            screenshot_path = f"game_result_{timestamp}.png"
            pygame.image.save(self.screen, screenshot_path)
            print(f"\nScreenshot saved as: {screenshot_path}")
            
            #self.screen.fill((0, 0, 40))
            winner_text = self.font.render(f"Ship {winner.id} is the winner!", True, GOLD_COLOR)
            score_text = self.font.render(f"Final Score: {winner.score}", True, GOLD_COLOR)

            text_rect = winner_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 20))
            score_rect = score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 20))

            self.screen.blit(winner_text, text_rect)
            self.screen.blit(score_text, score_rect)

            # Print leaderboard to console
            print("\nFinal Leaderboard:")
            print("-----------------")
            sorted_ships = sorted(self.ships, key=lambda x: x.score, reverse=True)
            for i, ship in enumerate(sorted_ships, 1):
                status = "Destroyed" if ship.is_destroyed else "Active"
                print(f"{i}. {ship.id}: Score={ship.score}, Hits={ship.bullets_hit_count}, Status={status}")


            

            pygame.display.flip()


    def check_collisions(self):
        # Check bullet hits
        for bullet in self.bullets[:]:
            # Check collision with ships
            for ship in self.ships:
                if ship is not bullet['owner'] and not ship.is_destroyed:
                    if math.dist((bullet['x'], bullet['y']), (ship.x, ship.y)) < SHIP_COLLISION_RADIUS:
                        ship.health -= HEALTH_BULLET_DAMAGE
                        if bullet in self.bullets:
                            self.bullets.remove(bullet)
                        bullet['owner'].score += BULLET_HIT_SCORE
                        bullet['owner'].bullets_hit_count += 1  # Increment hit counter

                        if ship.health <= 0 and not ship.is_destroyed:
                            bullet['owner'].score += SHIP_DISTRUCTION_SCORE
                            ship.is_destroyed = True
                            self.scatter_gold(ship)

                            # Award to all living ships if a ship is destroyed
                            for other_ship in self.ships:
                                if not other_ship.is_destroyed:
                                    other_ship.score += SHIP_DESTROYED_ALL_SHIPS_BONUS

                            # Check if only one ship remains after this destruction
                            alive_ships = [s for s in self.ships if not s.is_destroyed]
                            if len(alive_ships) == 1 and not self.bonus_awarded:
                                surviving_ship = alive_ships[0]
                                surviving_ship.score *= LAST_SHIP_STANDING_MULTIPLIER_BONUS  # Award bonus
                                self.bonus_awarded = True  # Ensure bonus is only awarded once
                                #print(f"Bonus awarded to Ship {surviving_ship.id} for being the last ship remaining.")

            # New: Check collision with asteroids
            for asteroid in self.asteroids:
                if math.dist((bullet['x'], bullet['y']), (asteroid.x, asteroid.y)) < asteroid.radius:
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    break  # Bullet destroyed, no need to check other asteroids

        # Check gold collection
        for ship in self.ships:
            if not ship.is_destroyed:
                for gold_pos in self.gold_positions[:]:
                    if math.dist((ship.x, ship.y), gold_pos) < SHIP_COLLISION_RADIUS:
                        self.gold_positions.remove(gold_pos)
                        ship.score += GOLD_VALUE
                        ship.gold_collected += 1

        # Check collisions between ships
        for i in range(len(self.ships)):
            ship_a = self.ships[i]
            if ship_a.is_destroyed:
                continue
            for j in range(i + 1, len(self.ships)):
                ship_b = self.ships[j]
                if ship_b.is_destroyed:
                    continue
                # Check if ships are colliding
                dx = ship_b.x - ship_a.x
                dy = ship_b.y - ship_a.y
                distance = cached_hypot(dx, dy)
                if distance < SHIP_COLLISION_DISTANCE:
                    # Ships are colliding, resolve collision
                    overlap = SHIP_COLLISION_DISTANCE - distance
                    if distance == 0:
                        # Ships are in the same position; choose random direction
                        angle = random.uniform(0, 2 * math.pi)
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                    else:
                        dx /= distance
                        dy /= distance
                    # Move ships apart equally
                    ship_a.x -= dx * overlap / 2
                    ship_a.y -= dy * overlap / 2
                    ship_b.x += dx * overlap / 2
                    ship_b.y += dy * overlap / 2

                    # Ensure ships are within bounds considering SHIP_SIZE
                    ship_a.x = max(self.border_left + SHIP_SIZE, min(ship_a.x, self.screen_width - self.border_right - SHIP_SIZE))
                    ship_a.y = max(self.border_top + SHIP_SIZE, min(ship_a.y, self.screen_height - self.border_bottom - SHIP_SIZE))
                    ship_b.x = max(self.border_left + SHIP_SIZE, min(ship_b.x, self.screen_width - self.border_right - SHIP_SIZE))
                    ship_b.y = max(self.border_top + SHIP_SIZE, min(ship_b.y, self.screen_height - self.border_bottom - SHIP_SIZE))

        # New: Check collisions between ships and asteroids
        for ship in self.ships:
            if ship.is_destroyed:
                continue
            for asteroid in self.asteroids:
                distance = math.dist((ship.x, ship.y), (asteroid.x, asteroid.y))
                if distance < SHIP_COLLISION_RADIUS + asteroid.radius:
                    
                    # Optional: Adjust ship's position to prevent overlapping
                    overlap = SHIP_COLLISION_RADIUS + asteroid.radius - distance
                    if distance == 0:
                        # Ships are in the same position as asteroid; choose random direction
                        angle = random.uniform(0, 2 * math.pi)
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                    else:
                        dx = (ship.x - asteroid.x) / distance
                        dy = (ship.y - asteroid.y) / distance
                    # Move the ship out of collision
                    ship.x += dx * overlap
                    ship.y += dy * overlap

                    # Ensure ship is within bounds considering SHIP_SIZE
                    ship.x = max(self.border_left + SHIP_SIZE, min(ship.x, self.screen_width - self.border_right - SHIP_SIZE))
                    ship.y = max(self.border_top + SHIP_SIZE, min(ship.y, self.screen_height - self.border_bottom - SHIP_SIZE))
                    
                    # Since asteroid should not move, we do not alter its position or velocity
                    # If multiple collisions occur, additional handling might be necessary

    def spawn_gold(self):
        x = random.randint(self.border_left + GOLD_SIZE, self.screen_width - self.border_right - GOLD_SIZE)  # Adjusted for GOLD_SIZE
        y = random.randint(self.border_top + GOLD_SIZE, self.screen_height - self.border_bottom - GOLD_SIZE)    # Adjusted for GOLD_SIZE
        self.gold_positions.append((x, y))

    def spawn_initial_gold(self):
        for _ in range(INITIAL_GOLD_COUNT):
            self.spawn_gold()

    def update_bullets(self, dt):
        for bullet in self.bullets[:]:  # Create copy to safely remove bullets
            bullet['x'] += bullet['speed'] * math.cos(math.radians(bullet['angle'])) * dt
            bullet['y'] += bullet['speed'] * math.sin(math.radians(bullet['angle'])) * dt

            # Remove bullets that enter the border area considering BULLET_SIZE
            if (bullet['x'] < self.border_left - BULLET_SIZE or bullet['x'] > self.screen_width - self.border_right + BULLET_SIZE or
                bullet['y'] < self.border_top - BULLET_SIZE or bullet['y'] > self.screen_height - self.border_bottom + BULLET_SIZE):
                self.bullets.remove(bullet)

    def process_action(self, ship: Spaceship, action: Action, dt: float, current_time: float):
        if ship.is_destroyed or ship.health <= 0:
            return

        if action == Action.ROTATE_RIGHT:
            ship.angle += SHIP_TURN_SPEED * dt  # Adjusted rotation speed
            ship.angle %= 360       # Normalize angle
        elif action == Action.ROTATE_LEFT:
            ship.angle -= SHIP_TURN_SPEED * dt  # Adjusted rotation speed
            ship.angle %= 360       # Normalize angle
        elif action == Action.ACCELERATE:
            # Add acceleration
            ship.velocity_x += ship.acceleration * math.cos(math.radians(ship.angle)) * dt
            ship.velocity_y += ship.acceleration * math.sin(math.radians(ship.angle)) * dt

            # Limit velocity
            speed = math.sqrt(ship.velocity_x**2 + ship.velocity_y**2)
            if speed > ship.max_velocity and speed > 0:
                ship.velocity_x = (ship.velocity_x / speed) * ship.max_velocity
                ship.velocity_y = (ship.velocity_y / speed) * ship.max_velocity
        elif action == Action.SHOOT:
            # Check shooting cooldown
            if current_time - ship.last_shot_time >= BULLET_COOLDOWN:
                bullet = {
                    'x': ship.x + SHIP_SIZE * math.cos(math.radians(ship.angle)),
                    'y': ship.y + SHIP_SIZE * math.sin(math.radians(ship.angle)),
                    'angle': ship.angle,
                    'speed': BULLET_SPEED,
                    'owner': ship
                }
                self.bullets.append(bullet)
                ship.last_shot_time = current_time
        elif action == Action.BRAKE:
            ship.velocity_x *= SHIP_BRAKE_FACTOR
            ship.velocity_y *= SHIP_BRAKE_FACTOR

        # Apply velocity and friction
        ship.velocity_x *= ship.friction ** dt  # Adjusted for delta time
        ship.velocity_y *= ship.friction ** dt  # Adjusted for delta time
        ship.x += ship.velocity_x * dt
        ship.y += ship.velocity_y * dt

        # Keep ships within game area bounds considering SHIP_SIZE
        ship.x = max(self.border_left + SHIP_SIZE, min(ship.x, self.screen_width - self.border_right - SHIP_SIZE))
        ship.y = max(self.border_top + SHIP_SIZE, min(ship.y, self.screen_height - self.border_bottom - SHIP_SIZE))

        # Optional: Add assertions to catch NaN values
        assert not math.isnan(ship.angle), "ship.angle is NaN"
        assert not math.isnan(ship.x), "ship.x is NaN"
        assert not math.isnan(ship.y), "ship.y is NaN"
        assert not math.isnan(ship.velocity_x), "ship.velocity_x is NaN"
        assert not math.isnan(ship.velocity_y), "ship.velocity_y is NaN"

    def scatter_gold(self, ship: Spaceship):
        gold_to_scatter = int(ship.gold_collected * GOLD_SCATTER_FRACTION)  # Scatter 50% of collected gold
        for _ in range(gold_to_scatter):
            scatter_distance = random.randint(GOLD_SCATTER_DISTANCE_MIN, GOLD_SCATTER_DISTANCE_MAX)
            scatter_angle = random.uniform(0, 360)
            x = ship.x + scatter_distance * math.cos(math.radians(scatter_angle))
            y = ship.y + scatter_distance * math.sin(math.radians(scatter_angle))

            # Keep gold within game area bounds considering GOLD_SIZE
            x = max(self.border_left + GOLD_SIZE, min(x, self.screen_width - self.border_right - GOLD_SIZE))
            y = max(self.border_top + GOLD_SIZE, min(y, self.screen_height - self.border_bottom - GOLD_SIZE))
            self.gold_positions.append((x, y))

        ship.gold_collected //= 2  # Reduce collected gold by 50%

    def draw_look_ahead_cone(self, ship, cone_angle, max_distance, color):
        # Convert angles to radians
        ship_angle_rad = math.radians(ship.angle)
        cone_angle_rad = math.radians(cone_angle)
        
        # Calculate cone points
        left_angle = ship_angle_rad - cone_angle_rad/2
        right_angle = ship_angle_rad + cone_angle_rad/2
        
        # Calculate end points of cone lines
        left_x = ship.x + max_distance * math.cos(left_angle)
        left_y = ship.y + max_distance * math.sin(left_angle)
        right_x = ship.x + max_distance * math.cos(right_angle)
        right_y = ship.y + max_distance * math.sin(right_angle)
        
        # Draw the cone lines
        pygame.draw.line(self.screen, color, (ship.x, ship.y), (left_x, left_y), 1)
        pygame.draw.line(self.screen, color, (ship.x, ship.y), (right_x, right_y), 1)
        # Draw arc to connect the lines
        pygame.draw.arc(self.screen, color, 
                    (ship.x - max_distance, ship.y - max_distance, 
                        max_distance * 2, max_distance * 2),
                    -right_angle, -left_angle, 1)

    def draw(self):
        if self.background:
            # Draw background image
            self.screen.blit(self.background, (0, 0))
        else:
            # If background is None, fill with a solid color
            self.screen.fill((0, 0, 0))

        # Draw border around the game area
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.border_left, self.border_top,
                          self.game_width, self.game_height), 2)
        # Draw asteroids
        for asteroid in self.asteroids:
            pygame.draw.circle(self.screen, (128, 128, 128),
                               (int(asteroid.x), int(asteroid.y)), asteroid.radius)

        # Draw each spaceship
        for ship in self.ships:
            # Set color based on ship status
            ship_color = SHIP_DESTROYED_COLOR if ship.is_destroyed else SHIP_ACTIVE_COLOR

            # Draw spaceship triangle
            ship_points = [
                (ship.x + SHIP_SIZE * math.cos(math.radians(ship.angle)),
                 ship.y + SHIP_SIZE * math.sin(math.radians(ship.angle))),
                (ship.x + SHIP_SIDE_OFFSET * math.cos(math.radians(ship.angle + SHIP_SIDE_ANGLE)),
                 ship.y + SHIP_SIDE_OFFSET * math.sin(math.radians(ship.angle + SHIP_SIDE_ANGLE))),
                (ship.x + SHIP_SIDE_OFFSET * math.cos(math.radians(ship.angle - SHIP_SIDE_ANGLE)),
                 ship.y + SHIP_SIDE_OFFSET * math.sin(math.radians(ship.angle - SHIP_SIDE_ANGLE)))
            ]
            pygame.draw.polygon(self.screen, ship_color, ship_points)

            # Draw health bar (even for destroyed ships)
            health_width = SHIP_HEALTH_BAR_WIDTH * (ship.health / HEALTH_FULL)
            pygame.draw.rect(self.screen, (255, 0, 0),
                             (ship.x - SHIP_HEALTH_BAR_WIDTH / 2, ship.y - 30, SHIP_HEALTH_BAR_WIDTH, SHIP_HEALTH_BAR_HEIGHT))
            pygame.draw.rect(self.screen, (0, 255, 0),
                             (ship.x - SHIP_HEALTH_BAR_WIDTH / 2, ship.y - 30, health_width, SHIP_HEALTH_BAR_HEIGHT))

            # Draw score
            font = self.text_font
            score_text = font.render(str(ship.score), True, ship_color)
            self.screen.blit(score_text, (ship.x + SHIP_SCORE_DISPLAY_OFFSET_X, ship.y + SHIP_SCORE_DISPLAY_OFFSET_Y))

            # Draw bullets hit count
            bullets_hit_text = font.render(f"Hits: {ship.bullets_hit_count}", True, ship_color)
            self.screen.blit(bullets_hit_text, (ship.x + SHIP_SCORE_DISPLAY_OFFSET_X, ship.y + SHIP_SCORE_DISPLAY_OFFSET_Y + 20))

            # Draw brain ID
            brain_id_text = font.render(f"{ship.id}", True, ship_color)
            self.screen.blit(brain_id_text, (ship.x + SHIP_BRAIN_ID_DISPLAY_OFFSET_X, ship.y + SHIP_BRAIN_ID_DISPLAY_OFFSET_Y))

        # # Draw look ahead cones for Q-learning ships
        # for ship in self.ships:
        #     if "Learner" in ship.id and not ship.is_destroyed:
        #         # Draw different cones with different colors
        #         LOOK_AHEAD_CONE_ANGLE = 15       # Cone of vision in degrees
        #         LOOK_AHEAD_CONE_ANGLE_ASTEROID = 20
        #         MAX_LOOK_AHEAD_DISTANCE = 500    # Maximum distance to look ahead for objects
        #         self.draw_look_ahead_cone(ship, LOOK_AHEAD_CONE_ANGLE, MAX_LOOK_AHEAD_DISTANCE, (0, 255, 0))  # Green for normal cone
        #         self.draw_look_ahead_cone(ship, LOOK_AHEAD_CONE_ANGLE_ASTEROID, MAX_LOOK_AHEAD_DISTANCE, (255, 165, 0))  # Orange for asteroid cone

        # Draw gold pieces
        for gold_pos in self.gold_positions:
            pygame.draw.circle(self.screen, GOLD_COLOR,
                               (int(gold_pos[0]), int(gold_pos[1])), GOLD_SIZE)


        # Draw bullets
        for bullet in self.bullets:
            pygame.draw.circle(self.screen, BULLET_COLOR,
                               (int(bullet['x']), int(bullet['y'])), BULLET_SIZE)

        # Draw leaderboard in the right border area
        leaderboard_x = self.screen_width - self.border_right + 10
        leaderboard_y = self.border_top + 10
        leaderboard_width = self.border_right - 20
        pygame.draw.rect(self.screen, (30, 30, 30),
                         (self.screen_width - self.border_right + 5, self.border_top + 5,
                          leaderboard_width, self.screen_height - self.border_top - self.border_bottom - 10))
        leaderboard_text = self.font.render("Leaderboard", True, (255, 255, 255))
        self.screen.blit(leaderboard_text, (leaderboard_x, leaderboard_y))

        sorted_ships = sorted(self.ships, key=lambda x: x.score, reverse=True)[:LEADERBOARD_MAX_ENTRIES]
        for i, ship in enumerate(sorted_ships, 1):
            leaderboard_y += 25
            ship_text = self.font.render(f"{i}. {ship.id}: {ship.score}", True,
                                         (255, 255, 255) if not ship.is_destroyed else (128, 128, 128))
            self.screen.blit(ship_text, (leaderboard_x, leaderboard_y))

        # In the draw method, right after drawing the FPS counter:
        if self.clock:
            fps = int(self.clock.get_fps())
            fps_text = self.font.render(f"FPS: {fps}", True, (255, 255, 255))
            self.screen.blit(fps_text, (10, 10))
            
            # Add ticks remaining counter
            ticks_remaining = MAX_TICK_COUNT - self.tick_count
            ticks_text = self.font.render(f"Ticks remaining: {ticks_remaining}", True, (255, 255, 255))
            self.screen.blit(ticks_text, (150, 10))


        pygame.display.flip()

def update_plot(fig, ax_wins, ax_scores, ax_ticks, plot_x, plot_history_wins, plot_history_avg, plot_history_ticks, game_winners, game_scores, game_ticks, interval_size=10):
    """
    Updates the matplotlib plots with the number of wins, average scores, and average ticks per brain in each interval using grouped bar charts.
    """
    total_intervals = len(game_winners) // interval_size
    new_intervals = total_intervals - len(plot_x)  # Calculate only new intervals

    if new_intervals <= 0:
        return  # No new intervals to process

    for _ in range(new_intervals):
        start = len(plot_x) * interval_size  # Start index for new interval
        end = start + interval_size
        current_interval_winners = game_winners[start:end]
        current_interval_scores = game_scores[start:end]
        current_interval_ticks = game_ticks[start:end]

        # Update interval number
        plot_x.append(len(plot_x) + 1)  # Ensure cumulative interval numbering

        # Count wins per brain in the current interval
        interval_counts = {}
        for winner_id in current_interval_winners:
            interval_counts[winner_id] = interval_counts.get(winner_id, 0) + 1

        # Count scores per brain in the current interval
        interval_scores_per_brain = {}
        for game in current_interval_scores:
            for brain_id, score in game.items():
                interval_scores_per_brain.setdefault(brain_id, []).append(score)

        # Compute average ticks for the current interval
        if current_interval_ticks:
            avg_ticks = sum(current_interval_ticks) / len(current_interval_ticks)
        else:
            avg_ticks = 0
        plot_history_ticks.append(avg_ticks)

        # Identify all brains present so far
        all_brains = set(interval_counts.keys()).union(interval_scores_per_brain.keys()).union(plot_history_wins.keys())

        for brain_id in all_brains:
            # Update win counts
            win_count = interval_counts.get(brain_id, 0)
            if brain_id not in plot_history_wins:
                # Initialize history with zeros for previous intervals
                plot_history_wins[brain_id] = [0] * (len(plot_x) - 1)
            plot_history_wins[brain_id].append(win_count)

            # Update average scores
            if brain_id in interval_scores_per_brain:
                avg_score = sum(interval_scores_per_brain[brain_id]) / len(interval_scores_per_brain[brain_id])
            else:
                avg_score = 0
            if brain_id not in plot_history_avg:
                # Initialize history with zeros for previous intervals
                plot_history_avg[brain_id] = [0] * (len(plot_x) - 1)
            plot_history_avg[brain_id].append(avg_score)

    # Clear the previous plots
    ax_wins.clear()
    ax_scores.clear()
    ax_ticks.clear()

    # Prepare for Grouped Bar Charts
    brain_ids = sorted(plot_history_wins.keys())
    num_brains = len(brain_ids)
    indices = np.arange(1, len(plot_x) + 1)
    bar_width = 0.8 / num_brains  # Adjust bar width based on the number of brains

    # Colors for different brains
    colors = plt.cm.get_cmap('tab10', num_brains)

    # Plot Wins per Brain as Grouped Bar Charts
    for i, brain_id in enumerate(brain_ids):
        counts = plot_history_wins[brain_id]
        ax_wins.bar(indices + i * bar_width, counts, bar_width, label=brain_id, color=colors(i))

    ax_wins.set_xlabel('Interval Number')
    ax_wins.set_ylabel('Number of Wins')
    ax_wins.set_title(f'Number of Wins per Brain in Each {interval_size} Game Interval')
    ax_wins.legend(title="Brain ID")
    ax_wins.set_xticks(indices + bar_width * (num_brains - 1) / 2)
    ax_wins.set_xticklabels(plot_x)
    ax_wins.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Plot Average Scores per Brain as Grouped Bar Charts
    for i, brain_id in enumerate(brain_ids):
        avg_scores = plot_history_avg.get(brain_id, [0] * len(plot_x))
        ax_scores.bar(indices + i * bar_width, avg_scores, bar_width, label=brain_id, color=colors(i))

    ax_scores.set_xlabel('Interval Number')
    ax_scores.set_ylabel('Average Score')
    ax_scores.set_title(f'Average Score per Brain in Each {interval_size} Game Interval')
    ax_scores.legend(title="Brain ID")
    ax_scores.set_xticks(indices + bar_width * (num_brains - 1) / 2)
    ax_scores.set_xticklabels(plot_x)
    ax_scores.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Plot Average Ticks per Interval as Line Chart
    ax_ticks.plot(plot_x, plot_history_ticks, marker='o', linestyle='-', color='b')
    ax_ticks.set_xlabel('Interval Number')
    ax_ticks.set_ylabel('Average Number of Ticks')
    ax_ticks.set_title(f'Average Number of Ticks per Game in Each {interval_size} Game Interval')
    ax_ticks.set_xticks(plot_x)
    ax_ticks.set_xticklabels(plot_x)
    ax_ticks.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout for better spacing
    fig.tight_layout()

    # Refresh the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

# Main function to run the games
def main(training_mode=False, num_games=1):
    environment = GameEnvironment(training_mode)
    wins_per_brain = {}
    game_winners = []  # List to track the winner of each game
    game_scores = []    # List to track scores of all brains per game
    game_ticks = []     # List to track tick counts per game

    # Initialize plotting variables if in training mode
    if training_mode:
        plt.ion()  # Turn on interactive mode
        fig, (ax_wins, ax_scores, ax_ticks) = plt.subplots(3, 1, figsize=(15, 9))  # Increased figsize for better visibility
        plot_history_wins = {}    # Dictionary to hold win history per brain
        plot_history_avg = {}     # Dictionary to hold average score history per brain
        plot_history_ticks = []   # List to hold average ticks per interval
        plot_x = []               # List to hold interval numbers
        fig.tight_layout(pad=3.0)  # Adjust layout
        fig.show()

    for game_num in range(num_games):
        game = SpaceGame(environment, wins_per_brain)
        winner = game.run()

        # Collect winner information
        if winner:
            game_winners.append(winner.id)  # Track the winner
            print(f"Game {game_num + 1} winner: Ship {winner.id} with score {winner.score} after {game.tick_count} ticks. Alive ships: {len([ship for ship in game.ships if not ship.is_destroyed])}")

        # Collect scores of all ships in the game
        scores_this_game = {}
        for ship in game.ships:
            scores_this_game[ship.id] = ship.score
        game_scores.append(scores_this_game)

        # Collect tick count of the game
        game_ticks.append(game.tick_count)

        if training_mode and (game_num + 1) % PLOT_UPDATE_INTERVAL == 0:
            # Update the plot every PLOT_UPDATE_INTERVAL games
            update_plot(fig, ax_wins, ax_scores, ax_ticks, plot_x, plot_history_wins, plot_history_avg, plot_history_ticks, game_winners, game_scores, game_ticks, interval_size=PLOT_UPDATE_INTERVAL)
            plt.pause(0.001)  # Allow the plot to update

        if training_mode and (game_num + 1) % TRAINING_STATUS_INTERVAL == 0:
            print(f"Completed {game_num + 1} games.")

    # After all games have been played
    if training_mode:
        print("\nTraining completed.")
        for brain_id, wins in wins_per_brain.items():
            print(f"Brain {brain_id} won {wins} games.({wins / num_games * 100:.2f}% win rate)")
        
        # Print the number of games in which the game ticks were less than the max tick count
        print(f"Number of games with less than {MAX_TICK_COUNT} ticks: {len([tick_count for tick_count in game_ticks if tick_count < MAX_TICK_COUNT])}")
        # Notify all brains that training is complete
        for ship in game.ships:
            try:
                ship.brain.on_training_complete()
            except Exception as e:
                print(f"Error in brain '{ship.id}' on_training_complete: {e}")
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Keep the plot open

    if not training_mode:
        pygame.quit()

if __name__ == "__main__":
    num_games = 1
    if TRAINING_MODE:
        num_games = TRAINING_MODE_GAMES
    main(training_mode=TRAINING_MODE, num_games=num_games)

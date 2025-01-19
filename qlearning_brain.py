# import os
# import math
# import random
# import uuid
# import pickle
# from brain_interface import SpaceshipBrain, Action, GameState
# from space_game import (
#     HEALTH_FULL, BORDER_LEFT, BORDER_RIGHT, BORDER_TOP, BORDER_BOTTOM,
#     SCREEN_WIDTH, SCREEN_HEIGHT, TRAINING_MODE, MAX_TICK_COUNT, BULLET_COOLDOWN,
#     MAX_VELOCITY
# )
# from helpers import cached_atan2_degrees, cached_hypot
# from datetime import datetime

# # ============================
# # Constants
# # ============================

# SHOULD_TRAIN = TRAINING_MODE

# # Q-Learning Parameters
# INITIAL_EPSILON = 1.0       # Initial exploration rate
# LEARNING_RATE = 0.01        # Alpha: Learning rate
# DISCOUNT_FACTOR = 0.95      # Gamma: Discount factor
# MIN_EPSILON = 0.01          # Minimum exploration rate
# EPSILON_DECAY = 0.999       # Epsilon decay rate

# # Action Parameters
# ACTION_DURATION_FRAMES = 1  # Number of frames to keep the same action

# # State Detection Parameters
# MAX_LOOK_AHEAD_DISTANCE = 1000    # Maximum distance to look ahead for objects
# LOOK_AHEAD_CONE_ANGLE = 15       # Cone of vision in degrees

# NEAR_DISTANCE_THRESHOLD = 50
# MEDIUM_DISTANCE_THRESHOLD = 100

# # Reward Parameters
# SCORE_REWARD_FACTOR = 1        # Multiplier for score-based rewards
# BULLET_HIT_REWARD_FACTOR = 20  # Multiplier for bullet hit-based rewards

# # File Naming Templates
# RUN_ID = uuid.uuid4().hex
# CURRENT_DATETIME = datetime.now().strftime("%Y%m%d-%H%M%S")
# CURRENT_FILE_NAME = os.path.basename(__file__).split('.')[0]
# Q_TABLE_FILENAME = f'q_table-{CURRENT_FILE_NAME}-{CURRENT_DATETIME}-{RUN_ID}.pkl'
# STATS_FILE_TEMPLATE = f'qlearning_stats-{CURRENT_FILE_NAME}-{RUN_ID}.txt'

# # State Binning Parameters
# # Angle Difference Bins: Ranges from -180 to 180 degrees
# ANGLE_DIFF_BINS = [-180, -10, -5, 0, 5, 10, 180]

# # Distance Bins for Enemy, Border, Gold, and Asteroid
# DISTANCE_BINS_ENEMY = [0, 50, 150, 400, 600, MAX_LOOK_AHEAD_DISTANCE]
# DISTANCE_LABELS_ENEMY = ["enemy-very-near", "enemy-near", "enemy-medium", "enemy-far", "enemy-very-far"]

# DISTANCE_BINS_BORDER = [0, 50, 150, 400, 600, MAX_LOOK_AHEAD_DISTANCE]
# DISTANCE_LABELS_BORDER = ["border-very-near", "border-near", "border-medium", "border-far", "border-very-far"]

# DISTANCE_BINS_GOLD = [0, 50, 150, 400, 600, MAX_LOOK_AHEAD_DISTANCE]
# DISTANCE_LABELS_GOLD = ["gold-very-near", "gold-near", "gold-medium", "gold-far", "gold-very-far"]

# # **Add Distance Bins and Labels for Asteroid**
# DISTANCE_BINS_ASTEROID = [0, 50, 150, 400, 600, MAX_LOOK_AHEAD_DISTANCE]
# DISTANCE_LABELS_ASTEROID = ["asteroid-very-near", "asteroid-near", "asteroid-medium", "asteroid-far", "asteroid-very-far"]

# # Speed Bins
# SPEED_BINS = [0, 50, 100, MAX_VELOCITY]
# SPEED_LABELS = ["speed-very-low", "speed-low", "speed-medium", "speed-high"]

# # Health Bins
# HEALTH_BINS = [0, 50, 100]
# HEALTH_LABELS = ["health-low", "health-high"]

# def generate_angle_labels(entity_type):
#     """
#     Generates dynamic angle labels based on ANGLE_DIFF_BINS.
    
#     Args:
#         entity_type (str): The type of entity (e.g., 'enemy', 'border', 'gold', 'velocity', 'asteroid').
    
#     Returns:
#         list: A list of dynamically generated angle labels.
#     """
#     labels = []
#     for i in range(len(ANGLE_DIFF_BINS)-1):
#         from_angle = ANGLE_DIFF_BINS[i]
#         to_angle = ANGLE_DIFF_BINS[i+1]
#         label = f"angle_{entity_type}-from_{from_angle}_to_{to_angle}"
#         labels.append(label)
#     return labels

# ANGLE_DIFF_LABELS_ENEMY = generate_angle_labels("enemy")
# ANGLE_DIFF_LABELS_BORDER = generate_angle_labels("border")
# ANGLE_DIFF_LABELS_GOLD = generate_angle_labels("gold")
# ANGLE_DIFF_LABELS_VELOCITY = generate_angle_labels("velocity")  # For agent's own velocity angle difference
# # **Add Angle Difference Labels for Asteroid**
# ANGLE_DIFF_LABELS_ASTEROID = generate_angle_labels("asteroid")

# # ============================
# # QLearningBrain Class
# # ============================

# class QLearningBrain(SpaceshipBrain):
#     def __init__(self):
#         self._id = 'Q-Learner'
#         self.q_table = {}
#         self.epsilon = INITIAL_EPSILON
#         self.alpha = LEARNING_RATE
#         self.gamma = DISCOUNT_FACTOR
#         self.prev_state = None
#         self.prev_action = None
#         self.prev_score = 0
#         self.prev_health = HEALTH_FULL
#         self.prev_bullets_hit_count = 0  # New attribute to track bullet hits
#         self.epsilon_min = MIN_EPSILON
#         self.epsilon_decay = EPSILON_DECAY
#         self.stats_file = STATS_FILE_TEMPLATE
#         self.current_action = None
#         self.action_counter = 0
#         self.action_duration = ACTION_DURATION_FRAMES
#         self.episode_length = 0

#         self.episode_stats = {
#             'score': 0,
#             'reward': 0,
#             'epsilon': self.epsilon
#         }
#         if TRAINING_MODE and SHOULD_TRAIN:
#             # Load Q-table and epsilon if the file exists
#             if os.path.exists(Q_TABLE_FILENAME):
#                 with open(Q_TABLE_FILENAME, 'rb') as f:
#                     try:
#                         data = pickle.load(f)
#                         self.q_table = data.get('q_table', {})
#                         self.epsilon = data.get('epsilon', INITIAL_EPSILON)
#                         print(f"Loaded Q-table and epsilon from {Q_TABLE_FILENAME}. Current epsilon: {self.epsilon}")
#                     except (pickle.UnpicklingError, EOFError) as e:
#                         print(f"Failed to load Q-table from {Q_TABLE_FILENAME}: {e}")
#                         self.q_table = {}
#                         self.epsilon = INITIAL_EPSILON
#             else:
#                 print(f"No existing Q-table found. Starting fresh with epsilon: {self.epsilon}")
#         else:
#             # Search for the latest pkl file and load it
#             pkl_files = [f for f in os.listdir() if f.endswith('.pkl')]
#             if pkl_files:
#                 latest_file = max(pkl_files, key=os.path.getctime)
#                 if latest_file:
#                     with open(latest_file, 'rb') as f:
#                         data = pickle.load(f)
#                         self.q_table = data.get('q_table', {})
#                         self.epsilon = data.get('epsilon', INITIAL_EPSILON)
#                         #print(f"Loaded Q-table and epsilon from {latest_file}. Current epsilon: {self.epsilon}")

#     @property
#     def id(self) -> str:
#         return self._id

#     def calcRewardAndUpdateQTable(self, game_state: GameState, end_of_game: bool, won: bool):
#         if self.prev_state is not None and self.prev_action is not None:
#             reward = self.get_reward(game_state, end_of_game, won)
#             self.episode_stats['reward'] += reward
#             self.update_q_table(self.prev_state, self.prev_action, reward, self.get_state(game_state))

#     def decide_what_to_do_next(self, game_state: GameState) -> Action:
#         self.episode_length += 1
#         # Calculate reward every frame
#         if SHOULD_TRAIN:
#             self.calcRewardAndUpdateQTable(game_state, False, False)

#         # Choose new action only after action_duration frames
#         if self.current_action is None or self.action_counter >= self.action_duration:
#             current_state = self.get_state(game_state)

#             # Choose action using epsilon-greedy policy
#             if random.random() < self.epsilon:
#                 self.current_action = random.choice(list(Action))
#             else:
#                 q_values = self.q_table.get(current_state, {a: 0 for a in Action})
#                 max_q = max(q_values.values())
#                 actions_with_max_q = [a for a, q in q_values.items() if q == max_q]
#                 self.current_action = random.choice(actions_with_max_q)

#             self.action_counter = 0

#             # Update previous state and action
#             self.prev_state = current_state
#             self.prev_action = self.current_action

#         self.action_counter += 1

#         # Update previous score, health, and bullets_hit_count
#         current_ship = self.get_current_ship(game_state)
#         if current_ship:
#             self.prev_score = current_ship['score']
#             self.prev_health = current_ship['health']
#             self.prev_bullets_hit_count = current_ship.get('bullets_hit_count', 0)  # Update hit count
#         else:
#             self.prev_score = 0
#             self.prev_health = 0
#             self.prev_bullets_hit_count = 0

#         # Note: Removed epsilon decay from here

#         # Update episode stats
#         self.episode_stats['epsilon'] = self.epsilon

#         return self.current_action

#     def get_state(self, game_state: GameState):
#         current_ship = self.get_current_ship(game_state)
#         if not current_ship:
#             return None

#         x = current_ship['x']
#         y = current_ship['y']
#         ship_angle = current_ship['angle'] % 360

#         # Initialize state variables
#         angle_diff_enemy = "angle_enemy-no-data"
#         distance_enemy = "enemy-no-data"

#         angle_diff_border = "angle_border-no-data"
#         distance_border = "border-no-data"

#         angle_diff_gold = "angle_gold-no-data"
#         distance_gold = "gold-no-data"

#         # **Initialize asteroid state variables**
#         angle_diff_asteroid = "angle_asteroid-no-data"
#         distance_asteroid = "asteroid-no-data"

#         speed_label = "speed-no-data"
#         angle_diff_velocity = "angle_velocity-no-data"

#         health_label = "health-no-data"
#         can_shoot_label = "can_shoot-no-data"

#         # =============================
#         # Angle Difference and Distance to Enemy
#         # =============================
#         closest_enemy = None
#         min_enemy_distance = float('inf')
#         for ship in game_state.ships:
#             if ship['id'] != current_ship['id'] and ship['health'] > 0:
#                 dx = ship['x'] - x
#                 dy = ship['y'] - y
#                 dist = cached_hypot(dx, dy)
#                 if dist < min_enemy_distance:
#                     min_enemy_distance = dist
#                     closest_enemy = ship

#         if closest_enemy and min_enemy_distance < MAX_LOOK_AHEAD_DISTANCE:
#             # Calculate desired angle to face the enemy
#             desired_angle = math.degrees(math.atan2(closest_enemy['y'] - y, closest_enemy['x'] - x)) % 360
#             angle_diff_raw = (desired_angle - ship_angle + 180) % 360 - 180  # Normalize to [-180, 180]

#             # Bin the angle difference
#             angle_diff_enemy_binned = self.bin_angle_difference(angle_diff_raw, "enemy")

#             # Bin the distance
#             distance_enemy = self.bin_distance(min_enemy_distance, DISTANCE_BINS_ENEMY, DISTANCE_LABELS_ENEMY, "enemy")

#             angle_diff_enemy = angle_diff_enemy_binned
        
#         # =============================
#         # Angle Difference and Distance to Gold
#         # =============================
#         closest_gold = None
#         min_gold_distance = float('inf')
#         for gold_pos in game_state.gold_positions:
#             dx = gold_pos[0] - x
#             dy = gold_pos[1] - y
#             dist = cached_hypot(dx, dy)
#             if dist < min_gold_distance:
#                 min_gold_distance = dist
#                 closest_gold = gold_pos

#         if closest_gold and min_gold_distance < MAX_LOOK_AHEAD_DISTANCE:
#             # Calculate desired angle to face the gold
#             desired_angle = math.degrees(math.atan2(closest_gold[1] - y, closest_gold[0] - x)) % 360
#             angle_diff_raw = (desired_angle - ship_angle + 180) % 360 - 180  # Normalize to [-180, 180]

#             # Bin the angle difference
#             angle_diff_gold_binned = self.bin_angle_difference(angle_diff_raw, "gold")

#             # Bin the distance
#             distance_gold = self.bin_distance(min_gold_distance, DISTANCE_BINS_GOLD, DISTANCE_LABELS_GOLD, "gold")

#             angle_diff_gold = angle_diff_gold_binned

#         # =============================
#         # **Angle Difference and Distance to Asteroid**
#         # =============================
#         closest_asteroid = None
#         min_asteroid_distance = float('inf')
#         for asteroid in game_state.asteroids:
#             dx = asteroid['x'] - x
#             dy = asteroid['y'] - y
#             dist = cached_hypot(dx, dy)
#             if dist < min_asteroid_distance:
#                 min_asteroid_distance = dist
#                 closest_asteroid = asteroid

#         if closest_asteroid and min_asteroid_distance < MAX_LOOK_AHEAD_DISTANCE:
#             # Calculate desired angle to face the asteroid
#             desired_angle = math.degrees(math.atan2(closest_asteroid['y'] - y, closest_asteroid['x'] - x)) % 360
#             angle_diff_raw = (desired_angle - ship_angle + 180) % 360 - 180  # Normalize to [-180, 180]

#             # Bin the angle difference
#             angle_diff_asteroid_binned = self.bin_angle_difference(angle_diff_raw, "asteroid")

#             # Bin the distance
#             distance_asteroid = self.bin_distance(min_asteroid_distance, DISTANCE_BINS_ASTEROID, DISTANCE_LABELS_ASTEROID, "asteroid")

#             angle_diff_asteroid = angle_diff_asteroid_binned

#         # =============================
#         # Angle Difference and Distance to Border
#         # =============================
#         GAME_AREA_LEFT = BORDER_LEFT
#         GAME_AREA_RIGHT = SCREEN_WIDTH - BORDER_RIGHT
#         GAME_AREA_TOP = BORDER_TOP
#         GAME_AREA_BOTTOM = SCREEN_HEIGHT - BORDER_BOTTOM

#         rad_angle = math.radians(ship_angle)
#         dx_border = math.cos(rad_angle)
#         dy_border = math.sin(rad_angle)

#         distance_x = distance_y = float('inf')

#         # Calculate distance to vertical borders (left and right)
#         if dx_border > 0:
#             distance_x = (GAME_AREA_RIGHT - x) / dx_border
#         elif dx_border < 0:
#             distance_x = (GAME_AREA_LEFT - x) / dx_border

#         # Calculate distance to horizontal borders (top and bottom)
#         if dy_border > 0:
#             distance_y = (GAME_AREA_BOTTOM - y) / dy_border
#         elif dy_border < 0:
#             distance_y = (GAME_AREA_TOP - y) / dy_border

#         distance_border_ahead = min(distance_x, distance_y)

#         if distance_border_ahead < MAX_LOOK_AHEAD_DISTANCE:
#             # Calculate desired angle to face the border point
#             target_x = x + math.cos(rad_angle) * distance_border_ahead
#             target_y = y + math.sin(rad_angle) * distance_border_ahead
#             desired_angle = math.degrees(math.atan2(target_y - y, target_x - x)) % 360
#             angle_diff_raw = (desired_angle - ship_angle + 180) % 360 - 180  # Normalize to [-180, 180]

#             # Bin the angle difference
#             angle_diff_border_binned = self.bin_angle_difference(angle_diff_raw, "border")

#             # Bin the distance
#             distance_border = self.bin_distance(distance_border_ahead, DISTANCE_BINS_BORDER, DISTANCE_LABELS_BORDER, "border")

#             angle_diff_border = angle_diff_border_binned

#         # =============================
#         # Agent's own speed and velocity angle difference
#         # =============================
#         velocity_x = current_ship['velocity_x']
#         velocity_y = current_ship['velocity_y']
#         ship_speed = cached_hypot(velocity_x, velocity_y)

#         # Bin the speed
#         for i in range(len(SPEED_BINS)-1):
#             if SPEED_BINS[i] <= ship_speed < SPEED_BINS[i+1]:
#                 speed_label = SPEED_LABELS[i]
#                 break
#         else:
#             speed_label = SPEED_LABELS[-1]  # If speed exceeds max bin

#         # Calculate the angle difference between ship's facing angle and velocity angle
#         velocity_angle = math.degrees(math.atan2(velocity_y, velocity_x)) % 360
#         angle_diff_raw = (velocity_angle - ship_angle + 180) % 360 - 180  # Normalize to [-180, 180]

#         # Bin the angle difference
#         angle_diff_velocity = self.bin_angle_difference(angle_diff_raw, "velocity")

#         # =============================
#         # Agent's own health status
#         # =============================
#         health = current_ship['health']
#         health_percentage = (health / HEALTH_FULL) * 100

#         # Bin the health
#         for i in range(len(HEALTH_BINS)-1):
#             if HEALTH_BINS[i] <= health_percentage < HEALTH_BINS[i+1]:
#                 health_label = HEALTH_LABELS[i]
#                 break
#         else:
#             health_label = HEALTH_LABELS[-1]  # If health exceeds max bin

#         # =============================
#         # Can Shoot Status
#         # =============================
#         current_time = game_state.game_ticks
#         can_shoot = (current_time - current_ship['last_shot_time']) >= BULLET_COOLDOWN
#         can_shoot_label = "can_shoot" if can_shoot else "cannot_shoot"

#         # =============================
#         # Compile the State Tuple
#         # =============================
#         state = (
#             angle_diff_enemy,
#             distance_enemy,
#             angle_diff_border,
#             distance_border,
#             angle_diff_gold,
#             distance_gold,
#             angle_diff_asteroid,  # **Include angle difference to asteroid**
#             distance_asteroid,    # **Include distance to asteroid**
#             speed_label,
#             angle_diff_velocity,
#             health_label,
#             can_shoot_label
#         )

#         return state

#     def bin_angle_difference(self, angle_diff, entity_type):
#         """
#         Bins the angle difference and returns the corresponding label.
        
#         Args:
#             angle_diff (float): The angle difference in degrees.
#             entity_type (str): The type of entity ('enemy', 'border', 'gold', 'velocity', 'asteroid').
        
#         Returns:
#             str: The binned angle label.
#         """
#         labels = {
#             "enemy": ANGLE_DIFF_LABELS_ENEMY,
#             "border": ANGLE_DIFF_LABELS_BORDER,
#             "gold": ANGLE_DIFF_LABELS_GOLD,
#             "velocity": ANGLE_DIFF_LABELS_VELOCITY,
#             "asteroid": ANGLE_DIFF_LABELS_ASTEROID  # **Add asteroid labels**
#         }
#         for i in range(len(ANGLE_DIFF_BINS)-1):
#             if ANGLE_DIFF_BINS[i] <= angle_diff < ANGLE_DIFF_BINS[i+1]:
#                 return labels[entity_type][i]
#         # Handle the edge case where angle_diff == 180
#         return labels[entity_type][-1]

#     def bin_distance(self, distance, bins, labels, entity_type):
#         """
#         Bins the distance and returns the corresponding label.
        
#         Args:
#             distance (float): The distance value.
#             bins (list): The list of bin edges.
#             labels (list): The list of labels corresponding to the bins.
#             entity_type (str): The type of entity ('enemy', 'border', 'gold', 'asteroid').
        
#         Returns:
#             str: The binned distance label.
#         """
#         for i in range(len(bins)-1):
#             if bins[i] <= distance < bins[i+1]:
#                 return labels[i]
#         return labels[-1]

#     def get_reward(self, game_state: GameState, end_of_game: bool, won: bool):
#         current_ship = self.get_current_ship(game_state)
#         if not current_ship:
#             return -4000  # Large negative reward for being destroyed

#         score = current_ship['score']
#         score_change = score - self.prev_score

#         bullets_hit_count = current_ship.get('bullets_hit_count', 0)
#         bullets_hit_change = bullets_hit_count - self.prev_bullets_hit_count  # Calculate change in bullet hits

#         reward = 0
#         if score_change > 0:
#             # Reward for increase in score
#             reward += score_change * SCORE_REWARD_FACTOR
#         else:
#             reward -= 0.1  # Small negative reward each step to encourage faster wins

#         if bullets_hit_change > 0:
#             # Reward for successful bullet hits
#             reward += bullets_hit_change * BULLET_HIT_REWARD_FACTOR

#         if end_of_game and not won:
#             reward -= 1000  # Large negative reward for losing the game

#         if end_of_game and won:
#             reward += 4000  # Large positive reward for winning the game
#             reward += (1 - (self.episode_length / MAX_TICK_COUNT)) * 4000  # Extra reward for completing in shorter time

#         return reward

#     def update_q_table(self, state, action, reward, next_state):
#         if state is None or next_state is None:
#             return

#         prev_q_values = self.q_table.get(state, {a: 0 for a in Action})
#         next_q_values = self.q_table.get(next_state, {a: 0 for a in Action})
#         max_future_q = max(next_q_values.values())
#         prev_q = prev_q_values[action]

#         # Q-learning formula
#         new_q = prev_q + self.alpha * (reward + self.gamma * max_future_q - prev_q)
#         prev_q_values[action] = new_q
#         self.q_table[state] = prev_q_values

#     def get_current_ship(self, game_state: GameState):
#         for ship in game_state.ships:
#             if ship['id'] == self.id:
#                 return ship
#         return None

#     def save_q_table(self):
#         data = {
#             'q_table': self.q_table,
#             'epsilon': self.epsilon
#         }
#         with open(Q_TABLE_FILENAME, 'wb') as f:
#             pickle.dump(data, f)

#     def on_game_complete(self, final_state: GameState, won: bool):
#         self.calcRewardAndUpdateQTable(final_state, True, won)
#         if TRAINING_MODE and SHOULD_TRAIN:
#             # Get final stats for this episode
#             current_ship = self.get_current_ship(final_state)
#             if current_ship:
#                 self.episode_stats['score'] = current_ship['score']

#             # Log stats to file
#             with open(self.stats_file, 'a') as f:
#                 f.write(f"Score: {self.episode_stats['score']}, ")
#                 f.write(f"Total Reward: {self.episode_stats['reward']}, ")
#                 f.write(f"Epsilon: {self.epsilon}, ")
#                 f.write(f"Won: {won}\n")

#             # Print current episode stats
#             print(f"Q-Learning Episode Stats - Score: {self.episode_stats['score']}, "
#                   f"Reward: {self.episode_stats['reward']}, "
#                   f"Epsilon: {self.epsilon}", f"Number of states: {len(self.q_table)}")

#             # Reset stats for next episode
#             self.episode_stats = {
#                 'score': 0,
#                 'reward': 0,
#                 'epsilon': self.epsilon  # Update to current epsilon
#             }
#             self.episode_length = 0
#             self.prev_bullets_hit_count = 0  # Reset bullet hit count

#             # Decay epsilon after the game completes
#             self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

#             # Update episode stats with decayed epsilon
#             self.episode_stats['epsilon'] = self.epsilon

#             # Save the updated Q-table and epsilon
#             self.save_q_table()

#     def on_training_complete(self):
#         """Called when all training games are completed. Brains can implement this to handle end-of-training logic"""
#         pass

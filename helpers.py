from functools import lru_cache
import math

# Define cached versions of math functions
@lru_cache(maxsize=10000)
def cached_atan2_degrees(dy, dx):
    return math.degrees(math.atan2(dy, dx))

@lru_cache(maxsize=10000)
def cached_hypot(dx, dy):
    return math.hypot(dx, dy)
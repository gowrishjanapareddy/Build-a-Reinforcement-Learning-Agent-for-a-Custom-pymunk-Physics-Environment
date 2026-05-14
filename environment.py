import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pymunk
import pymunk.pygame_util
import pygame
import math

class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, reward_type='shaped', render_mode=None):
        super().__init__()
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.dt = 1.0 / 60.0
        
        # Observation space: cart x, cart vx, pole1 angle, pole1 omega, pole2 angle, pole2 omega
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Action space: Continuous force to the cart
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Pygame setup
        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.draw_options = None

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Double Inverted Pendulum")
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        elif self.render_mode == "rgb_array":
            pygame.init()
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.space = None
        self.cart_body = None
        self.pole1_body = None
        self.pole2_body = None
        
        # Environment constraints
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 100
        self.force_mag = 1500.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.space = pymunk.Space()
        self.space.gravity = (0, -900) # Negative y is down in pymunk coords if we set it up, but pymunk default is y up. We'll use y up.
        
        # Track
        static_body = self.space.static_body
        static_body.position = (self.screen_width / 2, self.screen_height / 2)
        track_shape = pymunk.Segment(static_body, (-400, 0), (400, 0), 2)
        track_shape.friction = 0.1
        track_shape.filter = pymunk.ShapeFilter(categories=0b1000)
        self.space.add(track_shape)

        # Cart
        cart_size = (60, 30)
        self.cart_body = pymunk.Body(self.cart_mass, pymunk.moment_for_box(self.cart_mass, cart_size))
        self.cart_body.position = (self.screen_width / 2, self.screen_height / 2)
        cart_shape = pymunk.Poly.create_box(self.cart_body, cart_size)
        cart_shape.friction = 0.1
        cart_shape.color = pygame.Color("blue")
        cart_shape.filter = pymunk.ShapeFilter(categories=0b0001, mask=0b1000) # Only collide with track
        
        # Constrain cart to track
        groove = pymunk.GrooveJoint(static_body, self.cart_body, (-400, 0), (400, 0), (0, 0))
        self.space.add(self.cart_body, cart_shape, groove)

        # Pole 1
        pole1_size = (10, self.pole_length)
        self.pole1_body = pymunk.Body(self.pole_mass, pymunk.moment_for_box(self.pole_mass, pole1_size))
        # Initial position slightly perturbed
        self.pole1_body.position = (self.screen_width / 2, self.screen_height / 2 + self.pole_length / 2)
        self.pole1_body.angle = self.np_random.uniform(low=-0.05, high=0.05)
        pole1_shape = pymunk.Poly.create_box(self.pole1_body, pole1_size)
        pole1_shape.color = pygame.Color("red")
        pole1_shape.filter = pymunk.ShapeFilter(categories=0b0010, mask=0b0000) # No collisions
        
        # Connect pole 1 to cart
        pivot1 = pymunk.PivotJoint(self.cart_body, self.pole1_body, self.cart_body.position)
        self.space.add(self.pole1_body, pole1_shape, pivot1)

        # Pole 2
        pole2_size = (10, self.pole_length)
        self.pole2_body = pymunk.Body(self.pole_mass, pymunk.moment_for_box(self.pole_mass, pole2_size))
        self.pole2_body.position = (
            self.pole1_body.position.x - (self.pole_length / 2) * math.sin(self.pole1_body.angle),
            self.pole1_body.position.y + (self.pole_length / 2) * math.cos(self.pole1_body.angle) + self.pole_length / 2
        )
        self.pole2_body.angle = self.np_random.uniform(low=-0.05, high=0.05)
        pole2_shape = pymunk.Poly.create_box(self.pole2_body, pole2_size)
        pole2_shape.color = pygame.Color("green")
        pole2_shape.filter = pymunk.ShapeFilter(categories=0b0100, mask=0b0000) # No collisions
        
        # Connect pole 2 to pole 1
        pivot2_pos = (
            self.pole1_body.position.x - (self.pole_length / 2) * math.sin(self.pole1_body.angle),
            self.pole1_body.position.y + (self.pole_length / 2) * math.cos(self.pole1_body.angle)
        )
        pivot2 = pymunk.PivotJoint(self.pole1_body, self.pole2_body, pivot2_pos)
        self.space.add(self.pole2_body, pole2_shape, pivot2)

        return self._get_obs(), {}

    def _get_obs(self):
        cart_x = self.cart_body.position.x - (self.screen_width / 2)
        cart_vx = self.cart_body.velocity.x
        
        # Pymunk angles: 0 is right, pi/2 is up. We want upright to be 0 for our reward calculation logic easily,
        # but let's stick to providing raw angles and handling cosine. In pymunk, since y is up, 
        # a box created vertically with create_box has angle 0 when it's horizontal.
        # Wait, if we use pymunk.Poly.create_box, the box's long side is along Y axis if we pass (width, height) = (10, 100).
        # When body.angle = 0, the box is upright! Let's verify: a box (10, 100) has width 10, height 100.
        # So upright is angle 0.
        
        theta1 = self.pole1_body.angle
        omega1 = self.pole1_body.angular_velocity
        theta2 = self.pole2_body.angle
        omega2 = self.pole2_body.angular_velocity
        
        return np.array([cart_x, cart_vx, theta1, omega1, theta2, omega2], dtype=np.float32)

    def step(self, action):
        force = float(action[0]) * self.force_mag
        self.cart_body.apply_force_at_local_point((force, 0), (0, 0))
        
        self.space.step(self.dt)
        
        obs = self._get_obs()
        cart_x, cart_vx, theta1, omega1, theta2, omega2 = obs
        
        # Reward calculation
        # Upright is angle 0. cos(0) = 1.
        baseline_reward = math.cos(theta1) + math.cos(theta2)
        
        if self.reward_type == 'baseline':
            reward = baseline_reward
        else: # shaped
            # Center penalty: encourage staying near center
            center_penalty = abs(cart_x) * 0.005 # reduced from 0.1 as cart_x is in pixels (can be up to 400)
            
            # Velocity penalty: encourage stability
            velocity_penalty = (abs(omega1) + abs(omega2)) * 0.01
            
            # Action penalty: encourage less effort
            action_penalty = (float(action[0])**2) * 0.001
            
            reward = baseline_reward - center_penalty - velocity_penalty - action_penalty

        # Check termination condition
        terminated = False
        # Terminate if cart goes off screen
        if abs(cart_x) > (self.screen_width / 2 - 30):
            terminated = True
            reward -= 10.0 # Penalty for failing
        
        # Terminate if poles fall too far (e.g., below horizontal)
        if abs(theta1) > math.pi / 2 or abs(theta2) > math.pi / 2:
            terminated = True
            reward -= 10.0
            
        truncated = False
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
            
        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Double Inverted Pendulum")
                self.clock = pygame.time.Clock()
                self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            elif self.render_mode == "rgb_array":
                pygame.init()
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
                self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
                
        # pymunk y-axis is up, pygame y-axis is down. DrawOptions handles this automatically if configured.
        # By default DrawOptions has a transform, but pymunk 6+ uses standard coordinates (y up).
        # We need to manually flip y in rendering or use transform.
        # Actually pymunk pygame_util handles the transformation if `positive_y_is_up` is true.
        # Let's explicitly clear and draw.
        self.screen.fill(pygame.Color("white"))
        
        # Pygame uses y-down. Pymunk 6 by default uses y-up.
        self.draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES

        self.space.debug_draw(self.draw_options)
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)
        elif self.render_mode == "rgb_array":
            # Extract rgb array from pygame surface
            # Note: flip y because pygame saves surface with y-down
            rgb_array = pygame.surfarray.array3d(self.screen)
            rgb_array = np.transpose(rgb_array, (1, 0, 2)) # (height, width, channels)
            return rgb_array

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

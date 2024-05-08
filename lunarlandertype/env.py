import gym
from gym import spaces 
import pygame
import sys
import numpy as np
from Boat import Boat
from FlowField import FlowField

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 50
GRID_SIZE = 100 # This is only for adding a turbulent flow field. It does not mean the environement is discrete. 

# Load background image
background = pygame.image.load("assets/background.jpg")  # Replace with your image path
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

class Environment(gym.Env):
    def __init__(self, start_pos: tuple = (100, 500), end_pos: tuple = (1000, 700), max_steps: int = 1500):
        super(Environment, self).__init__()
        pygame.init()
        self.max_steps = max_steps
        self.current_step = 0
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.prev_dist_to_goal = np.sqrt((self.end_pos[0] - self.start_pos[0])**2 + (self.end_pos[1] - self.start_pos[1])**2)

        # reward policy
        self.reward_policy = {
            "goal_reached": 1500,
            "step": -1,
            "border": -100,
            "closer_to_goal": 1
        }

        # action space = (left thruster force, right thruster force)
        self.action_space = spaces.Box(
            low=np.array([-10000.0, -10000.0], dtype=np.float32),
            high=np.array([10000.0, 10000.0], dtype=np.float32),
        )

        # observation space = (x, y, angle, velocity_x, velocity_y, flow_direction (currently removed), dist_to_goal)
        self.observation_space = spaces.Box(
            # low=np.array([0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(np.float32),
            low=np.array([0, 0, 0.0, 0.0, 0.0, 0.0]).astype(np.float32),
            # high=np.array([WIDTH, HEIGHT, 360.0, 500.0, 2.0 * np.pi, 1445.0]).astype(np.float32)
            high=np.array([WIDTH, HEIGHT, 360.0, 500.0, 500.0, 1445.0]).astype(np.float32)
        )

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.Font(None, 40)
        pygame.display.set_caption("")

        self.reset()

    def _check_termination_conditions(self, reward):
        done = False
        if self.current_step >= self.max_steps:
            done = True
            reward += self.reward_policy["step"]
        elif self.agent.rect.x < 0 or self.agent.rect.x > WIDTH or self.agent.rect.y < 0 or self.agent.rect.y > HEIGHT:
            done = True
            reward += self.reward_policy["border"]
        elif self.agent.rect.colliderect(self.end_pos[0], self.end_pos[1], 50, 50):
            done = True
            reward += self.reward_policy["goal_reached"]
        return done, reward


    def step(self, action, dt=1/60):
        """
        left thruster force,
        right thruster force
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.current_step += 1
        self.agent.move(self.flow_field, dt, action)

        # Update the environment based on the given action
        self.all_sprites.update()

        # default reward
        dist_to_goal = np.sqrt((self.end_pos[0] - self.agent.rect.x)**2 + (self.end_pos[1] - self.agent.rect.y)**2)
        if dist_to_goal < self.prev_dist_to_goal:
            reward = self.reward_policy["closer_to_goal"]
        else:
            reward = -self.reward_policy["closer_to_goal"]
        done = False

        self.prev_dist_to_goal = dist_to_goal

        # Check for termination conditions
        done, reward = self._check_termination_conditions(reward)

        # Return observation, reward, done, info
        # observation = (self.agent.rect.x, self.agent.rect.y, self.agent.angle, self.agent.velocity_x, self.agent.velocity_y, self.flow_field.get_flow_direction(self.agent.rect.x, self.agent.rect.y), dist_to_goal)
        observation = (self.agent.rect.x, self.agent.rect.y, self.agent.angle, self.agent.velocity_x, self.agent.velocity_y, dist_to_goal)

        return np.array(observation, dtype=np.float32), reward, done, {}


    def reset(self):
        # Reset the environment to its initial state
        # Return the initial observation
        self.current_step = 0

        # remove previous sprites
        self.all_sprites = pygame.sprite.Group()

        # initialize flow field
        self.flow_field = FlowField(WIDTH, HEIGHT, GRID_SIZE)
        self.flow_field.create_flow_field()
        self.flow_field.draw_arrows(self.screen)

        boat = Boat(self.start_pos[0], self.start_pos[1])
        self.agent = boat
        self.all_sprites.add(boat)
        self.all_sprites.update()

        dist_to_goal = np.sqrt((self.end_pos[0] - self.agent.rect.x)**2 + (self.end_pos[1] - self.agent.rect.y)**2)
        self.prev_dist_to_goal = dist_to_goal

        # observation = (self.agent.rect.x, self.agent.rect.y, self.agent.angle, self.agent.velocity_x, self.agent.velocity_y, self.flow_field.get_flow_direction(self.agent.rect.x, self.agent.rect.y), dist_to_goal)
        observation = (self.agent.rect.x, self.agent.rect.y, self.agent.angle, self.agent.velocity_x, self.agent.velocity_y, dist_to_goal)

        return np.array(observation, dtype=np.float32), {}
    
    def render(self):
        self.screen.blit(background, (0, 0))

        # Draw start and end markers
        self.screen.blit(self.font.render("A", True, (255, 0, 0)), (self.start_pos[0], self.start_pos[1]))
        self.screen.blit(self.font.render("B", True, (255, 0, 0)), (self.end_pos[0], self.end_pos[1]))

        self.all_sprites.draw(self.screen)
        self.flow_field.draw_arrows(self.screen)
        pygame.display.flip()

    def close(self):
        # Cleanup any resources when the environment is closed
        pygame.quit()
        sys.exit()


# Example of a training loop
if __name__ == '__main__':
    env = Environment()
    clock = pygame.time.Clock()  # Initialize the clock
    for episode in range(100):
        observation = env.reset()
        done = False
        while not done:
            # action = env.action_space.sample()  # Replace with your agent's action
            action = [500, 5000]
            
            # Update the clock
            dt = clock.tick(FPS) / 1000  # Convert milliseconds to seconds

            observation, reward, done, info = env.step(action)

            # Your training logic goes here

            env.render()

    env.close()

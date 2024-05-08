import pygame
import numpy as np

class FlowField:
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        # self.velocity = 150
        self.velocity = 0
        self.grid_size = grid_size
        self.arrow_size = (20, 20)  # Set the size of the arrow image
        self.create_flow_field()
        # Load the arrow image
        self.arrow_image = pygame.image.load("assets/arrow.png")
        self.arrow_image = pygame.transform.scale(self.arrow_image, (20, 20))


    def create_flow_field(self):
        # Initialize a flow field with a general direction and slight variations in angle
        flow_field = []
        # base_angle = np.random.uniform(0, 2 * np.math.pi)  # General direction
        base_angle = 0 # in radians

        for _ in range(self.width // self.grid_size):
            row = []
            # angle_variation = np.random.uniform(-np.pi / 8, np.pi / 8)  # Angle variation
            angle_variation = 0

            for _ in range(self.height // self.grid_size):
                angle = base_angle + angle_variation
                row.append(angle)

                # Update angle for the next cell with a slight variation
                # angle_variation += np.random.uniform(-np.pi / 16, np.pi / 16)
                angle_variation = 0

            flow_field.append(row)

        self.flow_field = flow_field


    def get_flow_direction(self, x, y):
        # Get the flow direction at a specific position
        grid_x = x // self.grid_size
        grid_y = y // self.grid_size
        if 0 <= grid_x < len(self.flow_field) and 0 <= grid_y < len(self.flow_field[0]):
            return self.flow_field[grid_x][grid_y]
        return 0  # Default to no flow

    def draw_arrows(self, screen):
        # Draw arrows on the screen to visualize the flow field
        for x in range(0, self.width, self.grid_size):
            for y in range(0, self.height, self.grid_size):
                direction = self.flow_field[x // self.grid_size][y // self.grid_size]
                arrow_length = min(self.grid_size // 2, 20)
                arrow_tip = (
                    x + arrow_length * np.cos(direction),
                    y - arrow_length * np.sin(direction),
                )

                # Calculate the angle of rotation for the arrow image
                arrow_rotation = np.degrees(-direction)

                # Rotate and draw the arrow image at the arrow's position
                rotated_arrow = pygame.transform.rotate(self.arrow_image, arrow_rotation)
                screen.blit(rotated_arrow, (arrow_tip[0] - self.arrow_size[0] // 2, arrow_tip[1] - self.arrow_size[1] // 2))


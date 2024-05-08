import pygame
import numpy as np
from FlowField import FlowField

class Boat(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load("assets/boat.png")  # Replace with your boat sprite path
        self.original_image = pygame.transform.scale(self.original_image, (70, 70))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 90
        self.velocity = 0
        self.velocity_x = 0
        self.velocity_y = 0
        self.MASS = 100
        self.FORCE = 5000
        self.TURN_ANGLE = 20

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.move(2)
        if keys[pygame.K_DOWN]:
            self.move(-2)
        if keys[pygame.K_LEFT]:
            self.angle -= 20 
        if keys[pygame.K_RIGHT]:
            self.angle += 20 

        # Rotate the boat image
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)


    def move(self, flow_field: FlowField, dt: int, action: int):
        # Get the flow direction at the current position
        flow_direction = flow_field.get_flow_direction(self.rect.x, self.rect.y)
        
        # Calculate the force components based on the flow direction
        flow_vel_x = flow_field.velocity * np.cos(flow_direction)
        flow_vel_y = flow_field.velocity * np.sin(flow_direction)

        total_force_y = 0
        total_force_x = 0

        if action == 0:
            total_force_x += self.FORCE * np.cos(np.radians(self.angle))  # Thrust force
            total_force_y += self.FORCE * np.sin(np.radians(self.angle))

        elif action == 1:
            total_force_x += (-self.FORCE) * np.cos(np.radians(self.angle))
            total_force_y += (-self.FORCE) * np.sin(np.radians(self.angle))

        elif action == 2:
            self.angle -= self.TURN_ANGLE

        elif action == 3:
            self.angle += self.TURN_ANGLE

        # Calculate the displacement using kinematic equations
        displacement_x = (self.velocity_x + flow_vel_x) * dt + 0.5 * (total_force_x) / self.MASS * dt**2
        displacement_y = (self.velocity_y + flow_vel_y) * dt + 0.5 * (total_force_y) / self.MASS * dt**2

        # Update the velocity using the change in force
        self.velocity_x += (total_force_x) / self.MASS * dt
        self.velocity_y += (total_force_y) / self.MASS * dt
        
        # Update the boat's position
        self.rect.x += displacement_x
        self.rect.y += displacement_y


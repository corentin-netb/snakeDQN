# snake_ai.py
import random
import copy
import pygame
import torch
from dqn_dual import DQN_agent
import math
import wandb

import matplotlib.pyplot as plt

class SnakeGame:
    def __init__(self, agent):
        self.drawing = False
        self.step_reward = 0
        self.food_reward = 0
        self.number_of_games = 0
        self.step = 0
        pygame.init()
        self.GAME_WIDTH = 280
        self.GAME_HEIGHT = 280
        self.GRID_SIZE = 20
        self.HEADER_HEIGHT = 40
        self.OBSTACLE_COLOR = (34, 49, 29)
        self.BACKGROUND_COLOR = (117, 162, 89)
        self.FOOD_COLOR = (138, 24, 24)
        self.SNAKE_COLOR = (34, 49, 29)
        self.TEXT_COLOR = (117, 162, 89)
        self.TIMEOUT_STEPS = 40
        self.training_gap_counter = 0

        self.game_display = pygame.display.set_mode((self.GAME_WIDTH, self.GAME_HEIGHT + self.HEADER_HEIGHT))
        pygame.display.set_caption('Snake Game')
        self.font = pygame.font.Font("assets/PressStart2P-Regular.ttf", 20)
        self.clock = pygame.time.Clock()

        self.snake_x = self.GAME_WIDTH // 2
        self.snake_y = self.GAME_HEIGHT // 2
        self.snake_x_change = 0
        self.snake_y_change = 0
        self.snake_pos = [
            (self.snake_x, self.snake_y),
            (self.snake_x - self.GRID_SIZE, self.snake_y),
            (self.snake_x - 2 * self.GRID_SIZE, self.snake_y)
        ]
        self.food_pos = self.spawn_food()
        self.food_spawn = True
        self.score = 0
        self.high_score = 0
        self.quit = False
        self.game_over = False
        self.dqn_agent = agent
        self.action = 4
        self.previous_action = 4
        self.reward = 0

        wandb.init(project="snake-dqn", config={
            'model': "CNN",
            'learning_rate': self.dqn_agent.lr,
            'epsilon': self.dqn_agent.epsilon,
            'eta': self.dqn_agent.eta
        })

    def spawn_food(self):
        valid_food_positions = [(x, y) for x in range(self.GRID_SIZE, self.GAME_WIDTH - self.GRID_SIZE, self.GRID_SIZE)
                                for y in range(self.GRID_SIZE, self.GAME_HEIGHT - self.GRID_SIZE, self.GRID_SIZE)]
        return random.choice(valid_food_positions)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True

    def move_snake(self, action):

        # Check if the action is opposite to the current direction
        if (self.action == 1 and self.previous_action == 2) or \
                (self.action == 2 and self.previous_action == 1) or \
                (self.action == 3 and self.previous_action == 4) or \
                (self.action == 4 and self.previous_action == 3):
            # Ignore the opposite direction
            action = self.previous_action
        else:
            a=2


        if action == 1:  # Up
            self.snake_x_change = 0
            self.snake_y_change = -self.GRID_SIZE
        elif action == 2:  # Down
            self.snake_x_change = 0
            self.snake_y_change = self.GRID_SIZE
        elif action == 3:  # Left
            self.snake_x_change = -self.GRID_SIZE
            self.snake_y_change = 0
        elif action == 4:  # Right
            self.snake_x_change = self.GRID_SIZE
            self.snake_y_change = 0

        self.snake_x += self.snake_x_change
        self.snake_y += self.snake_y_change
        self.snake_pos.insert(0, (self.snake_x, self.snake_y))
        self.reward += self.calculate_distance_reward()
        self.previous_action = action

        # Eating the food
        if self.snake_x == self.food_pos[0] and self.snake_y == self.food_pos[1]:
            self.score += 1
            self.food_reward = 50.0
            self.step_reward = 0
            self.food_spawn = True
            self.training_gap_counter = 0
        else:
            self.snake_pos.pop()
            self.training_gap_counter += 1

        if self.food_spawn:
            self.food_pos = self.spawn_food()
            self.food_spawn = False

    def calculate_distance_reward(self):
        # Distance Reward Component
        distance = abs(math.dist(self.snake_pos[0], self.food_pos)) / 14
        reward = 50 / (distance + 1)
        return reward

    def check_game_over(self):
        if (
            self.snake_x < self.GRID_SIZE
            or self.snake_x >= self.GAME_WIDTH - self.GRID_SIZE
            or self.snake_y < self.GRID_SIZE
            or self.snake_y >= self.GAME_HEIGHT - self.GRID_SIZE
            or (self.snake_x, self.snake_y) in self.snake_pos[1:]
        ):
            if self.score > self.high_score:
                self.high_score = self.score
            return True
        else:
            return False

    def draw_game(self):
        self.game_display.fill(self.OBSTACLE_COLOR)
        pygame.draw.rect(self.game_display, self.BACKGROUND_COLOR,
                         (self.GRID_SIZE, self.GRID_SIZE + self.HEADER_HEIGHT, self.GAME_WIDTH - 2 * self.GRID_SIZE,
                          self.GAME_HEIGHT - 2 * self.GRID_SIZE))
        pygame.draw.rect(self.game_display, self.OBSTACLE_COLOR, (0, 0, self.GAME_WIDTH, self.HEADER_HEIGHT))

        for pos in self.snake_pos:
            pygame.draw.rect(self.game_display, self.SNAKE_COLOR,
                             pygame.Rect(pos[0], pos[1] + self.HEADER_HEIGHT, self.GRID_SIZE, self.GRID_SIZE))

        pygame.draw.rect(self.game_display, self.FOOD_COLOR,
                         pygame.Rect(self.food_pos[0], self.food_pos[1] + self.HEADER_HEIGHT, self.GRID_SIZE,
                                     self.GRID_SIZE))

        # Score text in a larger font
        score_text = self.font.render(f"Score: {self.score}", True, self.TEXT_COLOR)
        self.game_display.blit(score_text, (20, 20))

        # Reward text in a smaller font
        small_font = pygame.font.Font("assets/PressStart2P-Regular.ttf", 10)
        reward_text = small_font.render(f"Reward: {self.reward:.2f}", True, self.TEXT_COLOR)
        self.game_display.blit(reward_text, (20, 45))

        # Number of games text in a smaller font
        small_font = pygame.font.Font("assets/PressStart2P-Regular.ttf", 8)
        reward_text = small_font.render(f"N°{self.number_of_games}", True, self.TEXT_COLOR)
        self.game_display.blit(reward_text, (200, 50))

        pygame.display.update()

    def run_game(self):
        while True:
            self.handle_events()

            state = self.get_game_state()

            # For the first step, always set the action to UP (1)
            if self.step == 0:
                self.action = 1
            else:
                self.action = self.dqn_agent.select_action(state)
            previous_state = copy.deepcopy(self.dqn_agent.state_input)
            self.move_snake(self.action)
            game_over = self.check_game_over()

            distance_reward = self.calculate_distance_reward()

            self.step += 1

            if game_over:
                game_over_penalty = -50.0
                self.reward = distance_reward + game_over_penalty
            else:
                self.step_reward -= 0.1
                self.reward = distance_reward + self.step_reward + self.food_reward
                self.food_reward = 0

            # Add the condition to reset the game when self.step reaches a certain amount
            if self.step >= 100:
                self.reset_game()

            if self.drawing == True:
                self.draw_game()

            M = 4 + len(self.snake_pos) // 2

            if self.training_gap_counter > M:
                consequent_state = self.get_game_state()

                next_state = copy.deepcopy(self.dqn_agent.state_input)
                next_state.pop(0)
                next_state.append(torch.tensor(consequent_state))
                self.dqn_agent.add_experience(previous_state, self.action, self.reward, next_state, game_over)
                self.dqn_agent.update_q_network(batch_size=4, gamma=0.8)
                self.dqn_agent.update_target_network(update_frequency=100)

                if game_over:
                    self.reset_game()

                if self.dqn_agent.total_steps % 1000 == 0 and self.dqn_agent.total_steps != 0:
                    torch.save(self.dqn_agent.model.state_dict(), 'saved_model.pth')

            else:
                self.reward = 0

            self.clock.tick(10)

    def reset_game(self):
        # Log relevant information to wandb
        wandb.log({
            'Total Steps': self.dqn_agent.total_steps,
            'Score': self.score,
            'Steps by Game': self.step,
        })

        if self.score > self.high_score:
            self.high_score = self.score

        self.score = 0
        self.reward = 0
        self.step_reward = 0
        self.number_of_games += 1
        self.step = 0

        # Set the snake's initial position and direction
        self.snake_x, self.snake_y = self.GAME_WIDTH // 2, self.GAME_HEIGHT // 2
        self.snake_x_change, self.snake_y_change = 0, 0
        self.snake_pos = [
            (self.snake_x, self.snake_y),
            (self.snake_x - self.GRID_SIZE, self.snake_y),
            (self.snake_x - 2 * self.GRID_SIZE, self.snake_y)
        ]

        # Ensure the initial direction is consistent
        self.action = 1
        self.previous_action = 1

        # Respawn food at a new location
        self.food_pos = self.spawn_food()
        self.food_spawn = True

        self.game_over = False

    def get_game_state(self):
        game_board = pygame.Surface((self.GAME_WIDTH - 2 * self.GRID_SIZE, self.GAME_HEIGHT - 2 * self.GRID_SIZE))
        game_board.fill(self.BACKGROUND_COLOR)

        for pos in self.snake_pos:
            pygame.draw.rect(game_board, self.SNAKE_COLOR,
                             pygame.Rect(pos[0] - self.GRID_SIZE, pos[1] - self.GRID_SIZE, self.GRID_SIZE,
                                         self.GRID_SIZE))

        pygame.draw.rect(game_board, self.FOOD_COLOR,
                         pygame.Rect(self.food_pos[0] - self.GRID_SIZE, self.food_pos[1] - self.GRID_SIZE,
                                     self.GRID_SIZE, self.GRID_SIZE))

        game_board = pygame.transform.rotate(game_board, 90)  # Rotate the game board to the correct orientation
        game_board = pygame.transform.flip(game_board, False, True)  # Flip vertically to correct mirroring
        game_board = pygame.transform.scale(game_board, (64, 64))  # Resize to 64x64
        game_state = pygame.surfarray.array3d(game_board)

        return game_state


if __name__ == '__main__':
    dqn_agent = DQN_agent()

    # Load the saved model weights
    dqn_agent.model.load_model_weights('saved_model.pth')

    game = SnakeGame(dqn_agent)
    game.run_game()
    wandb.finish()

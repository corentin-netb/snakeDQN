import pygame
import random

class SnakeGame:
    def __init__(self):
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

        self.game_display = pygame.display.set_mode((self.GAME_WIDTH, self.GAME_HEIGHT + self.HEADER_HEIGHT))
        pygame.display.set_caption('Snake Game')
        self.font = pygame.font.Font("assets/PressStart2P-Regular.ttf", 20)
        self.clock = pygame.time.Clock()

        self.snake_x = self.GAME_WIDTH // 2
        self.snake_y = self.GAME_HEIGHT // 2
        self.snake_x_change = 0
        self.snake_y_change = 0
        self.snake_pos = [(self.snake_x, self.snake_y)]
        self.food_pos = self.spawn_food()
        self.food_spawn = True
        self.score = 0
        self.high_score = 0
        self.game_over = False

    def spawn_food(self):
        valid_food_positions = [(x, y) for x in range(self.GRID_SIZE, self.GAME_WIDTH - self.GRID_SIZE, self.GRID_SIZE)
                                for y in range(self.GRID_SIZE, self.GAME_HEIGHT - self.GRID_SIZE, self.GRID_SIZE)]
        return random.choice(valid_food_positions)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.snake_x_change = 0
                    self.snake_y_change = -self.GRID_SIZE
                if event.key == pygame.K_DOWN:
                    self.snake_x_change = 0
                    self.snake_y_change = self.GRID_SIZE
                if event.key == pygame.K_LEFT:
                    self.snake_x_change = -self.GRID_SIZE
                    self.snake_y_change = 0
                if event.key == pygame.K_RIGHT:
                    self.snake_x_change = self.GRID_SIZE
                    self.snake_y_change = 0

    def move_snake(self):
        self.snake_x += self.snake_x_change
        self.snake_y += self.snake_y_change
        self.snake_pos.insert(0, (self.snake_x, self.snake_y))

        if self.snake_x == self.food_pos[0] and self.snake_y == self.food_pos[1]:
            self.score += 1
            self.food_spawn = True
        else:
            self.snake_pos.pop()

        if self.food_spawn:
            self.food_pos = self.spawn_food()
            self.food_spawn = False

    def check_game_over(self):
        if (self.snake_x < self.GRID_SIZE or self.snake_x >= self.GAME_WIDTH - self.GRID_SIZE or
            self.snake_y < self.GRID_SIZE or self.snake_y >= self.GAME_HEIGHT - self.GRID_SIZE or
            (self.snake_x, self.snake_y) in self.snake_pos[1:]):
            if self.score > self.high_score:
                self.high_score = self.score
            self.score = 0
            self.snake_x, self.snake_y = self.GAME_WIDTH // 2, self.GAME_HEIGHT // 2
            self.snake_x_change, self.snake_y_change = 0, 0
            self.snake_pos = [(self.snake_x, self.snake_y)]

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
                         pygame.Rect(self.food_pos[0], self.food_pos[1] + self.HEADER_HEIGHT, self.GRID_SIZE, self.GRID_SIZE))

        score_text = self.font.render(str(self.score), True, self.TEXT_COLOR)
        self.game_display.blit(score_text, (20, 20))
        pygame.display.update()

    def run_game(self):
        while not self.game_over:
            self.handle_events()
            self.move_snake()
            self.check_game_over()
            self.draw_game()
            self.clock.tick(10)

        pygame.quit()

if __name__ == '__main__':
    game = SnakeGame()
    game.run_game()

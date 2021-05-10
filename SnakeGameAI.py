from collections import namedtuple
import random
import pygame
import numpy as np

pygame.init()
font = pygame.font.Font('Arial.ttf', 25)
location = namedtuple("loc", "x, y")

class Snake:
    def __init__(self, width, height):
        pygame.display.set_caption("Snake Game AI")
        self.width = width
        self.reward = 0
        self.height = height
        self.AIDirection = 3
        self.currentDirection = 3
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset_game()

    def AIMove(self, action):
        if (np.array_equal(action, [1, 0, 0])):
            if (self.currentDirection != 0):
                self.AIDirection = self.currentDirection
        elif (np.array_equal(action, [0, 1, 0])):
            if (self.AIDirection == 1):
                self.AIDirection = 3
            elif (self.AIDirection == 2):
                self.AIDirection = 4
            elif (self.AIDirection == 3):
                self.AIDirection = 2
            elif (self.AIDirection == 4):
                self.AIDirection = 1
            else:
                raise Exception("ERROR: move function direction error")
        else:
            if (self.AIDirection == 1):
                self.AIDirection = 4
            elif (self.AIDirection == 2):
                self.AIDirection = 3
            elif (self.AIDirection == 3):
                self.AIDirection = 1
            elif (self.AIDirection == 4):
                self.AIDirection = 2
            else:
                raise Exception("ERROR: move function direction error")
        headX = self.head.x
        headY = self.head.y
        if (self.AIDirection == 1):  # Left
            if (self.currentDirection != 2):
                headX -= 40
                self.currentDirection = 1
            else:
                headX += 40
                self.currentDirection = 2
        elif (self.AIDirection == 2):  # Right
            if (self.currentDirection != 1):
                headX += 40
                self.currentDirection = 2
            else:
                headX -= 40
                self.currentDirection = 1
        elif (self.AIDirection == 3):  # Up
            if (self.currentDirection != 4):
                headY -= 40
                self.currentDirection = 3
            else:
                headY += 40
                self.currentDirection = 4
        elif (self.AIDirection == 4):  # Down
            if (self.currentDirection != 3):
                headY += 40
                self.currentDirection = 4
            else:
                headY -= 40
                self.currentDirection = 3
        else:
            raise Exception("ERROR: move function direction error")
        self.head = location(headX, headY)

    def checkCollision(self, Square = None):
        if (Square is None):
            if (self.head.x > self.width - 40):
                return True
            if (self.head.x < 0):
                return True
            if (self.head.y > self.height - 40):
                return True
            if (self.head.y < 0):
                return True
            if (self.head in self.snake[1:]):
                return True
            return False
        if (Square.x > self.width - 40):
            return True
        if (Square.x < 0):
            return True
        if (Square.y > self.height - 40):
            return True
        if (Square.y < 0):
            return True
        if (Square in self.snake[1:]):
            return True
        return False

    def placeApple(self):
        red = pygame.Color(255, 0, 0)
        tempApple = pygame.Rect(self.food.x, self.food.y, 40, 40)
        pygame.draw.rect(self.display, red, tempApple)

    def placeScore(self):
        black = pygame.Color(0, 0, 0)
        text = font.render("Current Score: " + str(self.score), True, black)
        self.display.blit(text, [(self.width / 2) - 90, self.height - 30])

    def placeSnake(self):
        green = pygame.Color(0, 128, 0)
        for snakeBody in self.snake:
            tempSnake = pygame.Rect(snakeBody.x, snakeBody.y, 40, 40)
            pygame.draw.rect(self.display, green, tempSnake)

    def updateGame(self):
        grey = pygame.Color(180, 180, 180)
        self.display.fill(grey)
        self.placeScore()
        self.placeApple()
        self.placeSnake()
        pygame.display.flip()

    def placeFood(self):
        mapWidth = (self.width - 40)
        mapHeight = (self.height - 40)
        foodX = random.randint(0, mapWidth // 40)
        foodY = random.randint(0, mapHeight // 40)
        tempFood = location(foodX * 40, foodY * 40)
        if (tempFood in self.snake):
            self.placeFood()
        self.food = tempFood

    def AIControl(self, action):
        self.checkQuit()
        self.iteration += 1
        self.reward = 0
        self.AIMove(action)
        self.snake.insert(0, self.head)
        if (self.checkCollision()):
            return True, self.score, -10
        if (self.iteration > 100*len(self.snake)):
            return True, self.score, -10
        self.checkSnakeFood()
        self.updateGame()
        self.clock.tick(10000000)
        return False, self.score, self.reward

    def checkSnakeFood(self):
        if self.head == self.food:
            self.score += 1
            self.reward += 10
            self.placeFood()
        else:
            self.snake.pop()

    def reset_game(self):
        self.AIDirection = 3
        self.currentDirection = 3
        self.head = location(self.width / 2, self.height / 2)
        snakeBody1 = location(self.head.x - 40, self.head.y)
        snakeBody2 = location(self.head.x - (40 + 40), self.head.y)
        self.snake = [self.head, snakeBody1, snakeBody2]
        self.score = 0
        self.reward = 0
        self.food = None
        self.placeFood()
        self.iteration = 0

    def checkQuit(self):
        for input in pygame.event.get():
            if (input.type == pygame.QUIT):
                pygame.quit()
                quit()
from collections import namedtuple
import random
import pygame

pygame.init()
font = pygame.font.Font('C:\Windows\Fonts\Arial.ttf', 25)
Point = namedtuple("Point", "x, y")
squareSize = 20
class Snake:
    def move(self, direction):
        headX = self.head.x
        headY = self.head.y
        if(self.currentDirection == 0):
            if (direction == 1 ): # Left
                headX += squareSize
                self.currentDirection = 1
            elif (direction == 2): # Right
                headX -= squareSize
                self.currentDirection = 2
            elif (direction == 3): # Up
                headY += squareSize
                self.currentDirection = 3
            elif (direction == 4): # Down
                headY -= squareSize
                self.currentDirection = 4
            else:
                raise Exception("ERROR: move function direction error")
        else:
            if (direction == 1 ): # Left
                if(self.currentDirection != 2):
                    headX += squareSize
                    self.currentDirection = 1
                else:
                    headX -= squareSize
                    self.currentDirection = 2
            elif (direction == 2): # Right
                if(self.currentDirection != 1):    
                    headX -= squareSize
                    self.currentDirection = 2
                else:
                    headX += squareSize
                    self.currentDirection = 1
            elif (direction == 3): # Up
                if(self.currentDirection != 4):    
                    headY += squareSize
                    self.currentDirection = 3
                else:
                    headY -= squareSize
                    self.currentDirection = 4
            elif (direction == 4): # Down
                if(self.currentDirection != 3):
                    headY -= squareSize
                    self.currentDirection = 4
                else:
                    headY += squareSize
                    self.currentDirection = 3
            else:
                raise Exception("ERROR: move function direction error")
        self.head = Point(headX, headY)

    def checkCollision(self):
        if (self.head.x > self.width - squareSize or self.head.x < 0 or self.head.y > self.height - squareSize or self.head.y < 0):
            return True
        if (self.head in self.snake[1]):
            return True
        return False

    def updateGame(self):
        self.display.fill(pygame.Color(255, 255, 255))
        for pt in self.snake:
            pygame.draw.rect(self.display, pygame.Color(0, 0, 255), pygame.Rect(pt.x, pt.y, squareSize, squareSize))
            pygame.draw.rect(self.display, pygame.Color(0, 0, 200), pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, pygame.Color(255, 0, 0), pygame.Rect(self.food.x, self.food.y, squareSize, squareSize))
        text = font.render("Score: " + str(self.score), True, pygame.Color(0, 0, 0))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def placeFood(self):
        foodX = random.randint(0, (self.width - squareSize)//squareSize) * squareSize
        foodY = random.randint(0, (self.height - squareSize)//squareSize) * squareSize
        self.food = Point(foodX, foodY)
        if self.food in self.snake:
            self.placeFood()

    def gameInputs(self):
        for input in pygame.event.get():
            if (input.type == pygame.QUIT):
                pygame.quit()
                quit()
            if input.type == pygame.KEYDOWN:
                if input.key == pygame.K_LEFT:
                    self.direction = 2
                elif input.key == pygame.K_RIGHT:
                    self.direction = 1
                elif input.key == pygame.K_UP:
                    self.direction = 4
                elif input.key == pygame.K_DOWN:
                    self.direction = 3
        self.move(self.direction)
        self.snake.insert(0, self.head)
        if(self.checkGameStatus()):
            return True, self.score
        self.checkFood()
        self.updateGame()
        self.clock.tick(20)
        return False, self.score

    def checkGameStatus(self):
        if(self.checkCollision()):
            return True

    def checkFood(self):
        if self.head == self.food:
            self.score += 1
            self.placeFood()
        else:
            self.snake.pop()

    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.direction = 1
        self.head = Point((self.width/2), (self.height/2))
        self.snake = [self.head, Point(self.head.x - squareSize, self.head.y), Point(self.head.x - squareSize + squareSize, self.head.y)]
        self.score = 0
        self.food = None
        self.currentDirection = 0
        self.placeFood()

if __name__ == '__main__':
    print("test")
    game = Snake(480, 640)
    while True:
        gameover, score = game.gameInputs()
        if gameover:
            break
    print("Gameover: Final Score", score)
    pygame.quit()

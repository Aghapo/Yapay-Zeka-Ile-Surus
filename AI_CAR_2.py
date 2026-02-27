import pygame 
import sys
import math

pygame.init()

WIDTH , HEIGHT = 800 , 600 
screen  = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("AI CAR VERSION 2")
clock = pygame.time.Clock()
FPS = 60

#Car sınıfı
class Car : 
    def __init__(self, x , y):
        self.x = x
        self.y = y
        self.angle = 0 
        self.velocity = 0
        self.accerelation = 0.9
        self.max_speed = 20
        self.friction = 1
        self.turn_speed = 5

    def update(self, keys): #Hızlanma ve yavaşlama
        if keys[pygame.K_UP]:
            self.velocity += self.accerelation
        if keys[pygame.K_DOWN]:
            self.velocity -= self.accerelation

        #Maksimum hız
        self.velocity = max(-self.max_speed, min(self.max_speed,self.velocity)) 

        #Sürtünme
        self.velocity *= self.friction

        if abs(self.velocity) > 0.1 : 
            if keys[pygame.K_LEFT]: 
                self.angle = self.angle + self.turn_speed
            if keys[pygame.K_RIGHT]:
                self.angle = self.angle - self.turn_speed

        self.x += math.cos(math.radians(self.angle)) * self.velocity
        self.y -= math.sin(math.radians(self.angle)) * self.velocity
        
    def draw (self, screen) : 
        pygame.draw.circle(screen, (0,200, 255), (int(self.x), int(self.y)), 10)

        end_x = self.x + math.cos(math.radians(self.angle)) * 20 
        end_y = self.y - math.sin(math.radians(self.angle)) * 20 
        pygame.draw.line(screen, (255, 100 ,0 ), (self.x, self.y) ,(int(end_x),int(end_y)),2)




Car = Car(400,300)
while True : 
    for game in pygame.event.get():
        if game.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    keys = pygame.key.get_pressed()
    Car.update(keys)

    screen.fill((30,30,30))
    Car.draw(screen)
    pygame.display.flip()
    clock.tick(FPS)
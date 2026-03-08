import pygame
import sys
import math
import numpy as np 

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Cars - Dönen Araba")
clock = pygame.time.Clock()
FPS = 60

WALLS = [
    ((100, 100), (700, 100)),
    ((700, 100), (700, 500)),
    ((700, 500), (100, 500)),
    ((100, 500), (100, 100)),
]

def line_intersection(p1, p2, p3, p4):
    x1, y1= p1; x2, y2 = p2
    x3, y3 = p3 ; x4, y4 = p4

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

    if denom == 0 :
        return None
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    
    if 0 < t < 1 and 0 < u < 1:
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        return (x, y)
    return None

    
class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.velocity = 0
        self.acceleration = 0.2
        self.max_speed = 5
        self.friction = 0.95
        self.turn_speed = 3

        # Araba görseli — basit renkli dikdörtgen
        self.image = pygame.Surface((40, 20), pygame.SRCALPHA)
        self.image.fill((0, 200, 255))
        # Ön tarafı belirtmek için küçük kırmızı işaret
        pygame.draw.rect(self.image, (255, 50, 50), (30, 5, 10, 10))

        #Sensör açıları
        self.sensor_angles = [-90, -45, 0 , 45, 90]
        self.sensor_length = 150
        self.sensor_readings = [1.0] * len(self.sensor_angles)


    def cast_sensor(self):
        self.sensor_readings = []
        for s_angle in self.sensor_angles:
            angle_rad = math.radians(self.angle + s_angle)
            end_x = self.x + math.cos(angle_rad) * self.sensor_length
            end_y = self.y - math.sin(angle_rad) * self.sensor_length

            closest = None
            closest_dist = self.sensor_length

            for wall in WALLS : 
                hit = line_intersection((self.x, self.y), (end_x, end_y), wall[0], wall[1])
                if hit :
                    dist = math.hypot(hit[0]-self.x, hit[1]-self.y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest = hit
            self.sensor_readings.append(closest_dist / self.sensor_length)

            if closest :
                pygame.draw.line(screen, (255, 50, 50),(int(self.x), int(self.y)), (int(closest[0]), int(closest[1])), 1)
                pygame.draw.circle(screen, (255, 255, 0),(int(closest[0]), int(closest[1])), 4)
            else:
                pygame.draw.line(screen, (0, 255, 100),(int(self.x), int(self.y)), (int(end_x), int(end_y)), 1)
    def update(self, keys):
        if keys[pygame.K_UP]:
            self.velocity += self.acceleration
        if keys[pygame.K_DOWN]:
            self.velocity -= self.acceleration

        self.velocity = max(-self.max_speed, min(self.max_speed, self.velocity))
        self.velocity *= self.friction

        if abs(self.velocity) > 0.1:
            if keys[pygame.K_LEFT]:
                self.angle += self.turn_speed
            if keys[pygame.K_RIGHT]:
                self.angle -= self.turn_speed

        self.x += math.cos(math.radians(self.angle)) * self.velocity
        self.y -= math.sin(math.radians(self.angle)) * self.velocity

    def draw(self, screen):
        self.cast_sensor()
        rotated = pygame.transform.rotate(self.image, self.angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect.topleft)



car = Car(400, 300)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    car.update(keys)

    screen.fill((30, 30, 30))

    # Duvar
    for wall in WALLS:
        pygame.draw.line(screen, (200, 200, 200), wall[0], wall[1], 3)

    car.draw(screen)

    # Sensör değerleri
    font = pygame.font.SysFont(None, 24)
    for i, val in enumerate(car.sensor_readings):
        text = font.render(f"S{i}: {val:.2f}", True, (255,255,255))
        screen.blit(text, (10, 10 + i*20))

    pygame.display.flip()
    clock.tick(FPS)
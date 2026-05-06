import pygame
import sys
import math
import numpy as np
import os
from neat import NEATPopulation

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Cars - NEAT")
clock = pygame.time.Clock()
FPS = 60

TRACK_OUTER = [(130,100),(313,38),(547,54),(702,146),(742,331),(697,472),(583,539),(249,538),(76,417),(66,202)]
TRACK_INNER = [(220,190),(344,127),(548,151),(630,241),(634,337),(617,394),(493,449),(286,421),(186,381),(173,247)]

fitness_history = []

CHECKPOINTS = [
    (int((o[0]+i[0])/2), int((o[1]+i[1])/2))
    for o, i in zip(TRACK_OUTER, TRACK_INNER)
]

def make_segments(points):
    segments = []
    for i in range(len(points)):
        segments.append((points[i], points[(i+1) % len(points)]))
    return segments

OUTER_SEGMENTS = make_segments(TRACK_OUTER)
INNER_SEGMENTS = make_segments(TRACK_INNER)
ALL_WALLS      = OUTER_SEGMENTS + INNER_SEGMENTS

def line_intersection(p1, p2, p3, p4):
    x1,y1 = p1; x2,y2 = p2
    x3,y3 = p3; x4,y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    if 0 < t < 1 and 0 < u < 1:
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))
    return None

def save_graph():
    if not fitness_history:
        return
    import matplotlib.pyplot as plt
    generations = [d["generation"] for d in fitness_history]
    bests       = [d["best"] for d in fitness_history]
    avgs        = [d["avg"] for d in fitness_history]
    species_counts = [d["species"] for d in fitness_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(generations, bests, label="En iyi fitness", color="cyan",   linewidth=2)
    ax1.plot(generations, avgs,  label="Ortalama fitness", color="orange", linewidth=2, linestyle="--")
    ax1.set_xlabel("Nesil")
    ax1.set_ylabel("Fitness")
    ax1.set_title("AI Cars NEAT — Öğrenme Eğrisi")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(generations, species_counts, color="lime", linewidth=2)
    ax2.set_xlabel("Nesil")
    ax2.set_ylabel("Tür Sayısı")
    ax2.set_title("Tür Çeşitliliği")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ogrenme_egrisi.png", dpi=150)
    print("Grafik kaydedildi:", os.path.abspath("ogrenme_egrisi.png"))
    plt.show()

class Car:
    def __init__(self, x, y, genome):
        self.x       = x
        self.y       = y
        self.angle   = 0
        self.velocity = 0
        self.acceleration = 0.2
        self.max_speed    = 5
        self.friction     = 0.95
        self.turn_speed   = 3
        self.alive        = True
        self.fitness      = 0
        self.checkpoint_index    = 0
        self.checkpoints_passed  = 0
        self.frames_alive        = 0
        self.frames_since_checkpoint = 0
        self.slow_frames  = 0
        self.sensor_lines = []
        self.genome       = genome

        self.image = pygame.Surface((40, 20), pygame.SRCALPHA)
        self.image.fill((0, 200, 255))
        pygame.draw.rect(self.image, (255, 50, 50), (30, 5, 10, 10))

        self.sensor_angles  = [-90, -45, 0, 45, 90]
        self.sensor_length  = 150
        self.sensor_readings = [1.0] * 5

    def cast_sensors(self):
        self.sensor_lines    = []
        self.sensor_readings = []

        for s_angle in self.sensor_angles:
            angle_rad = math.radians(self.angle + s_angle)
            end_x = self.x + math.cos(angle_rad) * self.sensor_length
            end_y = self.y - math.sin(angle_rad) * self.sensor_length

            closest_point = None
            closest_dist  = self.sensor_length

            for wall in ALL_WALLS:
                hit = line_intersection(
                    (self.x, self.y), (end_x, end_y),
                    wall[0], wall[1]
                )
                if hit:
                    dist = math.hypot(hit[0] - self.x, hit[1] - self.y)
                    if dist < closest_dist:
                        closest_dist  = dist
                        closest_point = hit

            self.sensor_readings.append(closest_dist / self.sensor_length)
            draw_end = closest_point if closest_point else (end_x, end_y)
            self.sensor_lines.append((draw_end, closest_point is not None))

    def get_corners(self):
        angle_rad = math.radians(-self.angle)
        w, h   = 18, 9
        cos_a  = math.cos(angle_rad)
        sin_a  = math.sin(angle_rad)
        corners = []
        for dx, dy in [(-w,-h),(w,-h),(w,h),(-w,h)]:
            cx = self.x + dx * cos_a - dy * sin_a
            cy = self.y + dx * sin_a + dy * cos_a
            corners.append((cx, cy))
        return corners

    def check_checkpoint(self):
        target = CHECKPOINTS[self.checkpoint_index]
        dist   = math.hypot(self.x - target[0], self.y - target[1])
        if dist < 80:
            self.checkpoints_passed += 1
            self.checkpoint_index    = (self.checkpoint_index + 1) % len(CHECKPOINTS)
            self.frames_since_checkpoint = 0

    def check_collision(self):
        corners = self.get_corners()
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i+1) % 4]
            for wall in ALL_WALLS:
                if line_intersection(p1, p2, wall[0], wall[1]):
                    self.alive = False
                    return

    def update(self):
        if not self.alive:
            return

        if abs(self.velocity) < 0.5:
            self.slow_frames += 1
        else:
            self.slow_frames = 0
        if self.slow_frames > 120:
            self.alive = False
            return

        self.cast_sensors()

        speed_normalized = abs(self.velocity) / self.max_speed
        inputs   = self.sensor_readings + [speed_normalized]
        decision = self.genome.forward(inputs)

        if decision[0] > 0.5:
            self.velocity += self.acceleration
        else:
            self.velocity -= self.acceleration

        if decision[1] > 0.5:
            self.angle += self.turn_speed
        else:
            self.angle -= self.turn_speed

        self.velocity = max(-self.max_speed, min(self.max_speed, self.velocity))
        self.velocity *= self.friction

        self.x += math.cos(math.radians(self.angle)) * self.velocity
        self.y -= math.sin(math.radians(self.angle)) * self.velocity

        self.frames_alive            += 1
        self.frames_since_checkpoint += 1

        if self.frames_since_checkpoint > 300:
            self.alive = False
            return

        self.fitness = self.checkpoints_passed * 1000 + (self.frames_alive / 20)

        # Genome fitness'ını güncelle — NEAT bunu kullanıyor
        self.genome.fitness = self.fitness

        self.check_checkpoint()
        self.check_collision()

    def draw(self, is_best=False):
        if not self.alive:
            return

        for (end_x, end_y), hit in self.sensor_lines:
            color = (255, 50, 50) if hit else (0, 255, 100)
            pygame.draw.line(screen, color,
                (int(self.x), int(self.y)), (int(end_x), int(end_y)), 1)
            if hit:
                pygame.draw.circle(screen, (255, 255, 0), (int(end_x), int(end_y)), 3)

        # En iyi araba altın rengi, diğerleri mavi
        img = pygame.Surface((40, 20), pygame.SRCALPHA)
        if is_best:
            img.fill((255, 215, 0))   # altın
            pygame.draw.rect(img, (255, 50, 50), (30, 5, 10, 10))
        else:
            img.fill((0, 200, 255))   # mavi
            pygame.draw.rect(img, (255, 50, 50), (30, 5, 10, 10))

        rotated = pygame.transform.rotate(img, self.angle)
        rect    = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect.topleft)

        corners = self.get_corners()
        color   = (255, 215, 0) if is_best else (255, 100, 0)
        pygame.draw.polygon(screen, color,
            [(int(c[0]), int(c[1])) for c in corners], 1)

# --- Ana Simülasyon ---
START_X, START_Y = CHECKPOINTS[0]
INPUT_SIZE       = 6   # 5 sensör + 1 hız
OUTPUT_SIZE      = 2   # gaz, dönüş
POPULATION_SIZE  = 20

population = NEATPopulation(POPULATION_SIZE, INPUT_SIZE, OUTPUT_SIZE)
cars       = [Car(START_X, START_Y, g) for g in population.genomes]
font       = pygame.font.SysFont(None, 28)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            save_graph()
            sys.exit()

    for car in cars:
        car.update()

    if all(not car.alive for car in cars):
        fitnesses   = [car.fitness for car in cars]
        best_fitness = max(fitnesses)
        avg_fitness  = sum(fitnesses) / len(fitnesses)
        species_count = len(population.species)

        fitness_history.append({
            "generation": population.generation,
            "best":       best_fitness,
            "avg":        avg_fitness,
            "species":    species_count
        })

        print(f"Nesil {population.generation} — En iyi: {best_fitness:.0f} — "
              f"Ort: {avg_fitness:.0f} — Tür: {species_count}")

        population.evolve()
        cars = [Car(START_X, START_Y, g) for g in population.genomes]

    screen.fill((20, 20, 20))

    pygame.draw.polygon(screen, (60, 60, 60), TRACK_OUTER)
    pygame.draw.polygon(screen, (20, 20, 20), TRACK_INNER)
    pygame.draw.lines(screen, (180, 180, 180), True, TRACK_OUTER, 2)
    pygame.draw.lines(screen, (180, 180, 180), True, TRACK_INNER, 2)

    for i, cp in enumerate(CHECKPOINTS):
        color = (255, 255, 0) if i == 0 else (100, 100, 255)
        pygame.draw.circle(screen, color, cp, 8)

    # En iyi arabayı bul
    alive_cars  = [c for c in cars if c.alive]
    best_car    = max(cars, key=lambda c: c.fitness)

    for car in cars:
        car.draw(is_best=(car is best_car))

    alive_count = len(alive_cars)
    best        = max(car.fitness for car in cars)
    screen.blit(font.render(f"Nesil: {population.generation}",    True, (255,255,255)), (10, 10))
    screen.blit(font.render(f"Hayatta: {alive_count}",             True, (255,255,255)), (10, 35))
    screen.blit(font.render(f"En iyi: {best:.0f}",                True, (255,255,255)), (10, 60))
    screen.blit(font.render(f"Tür sayısı: {len(population.species)}", True, (255,255,255)), (10, 85))

    pygame.display.flip()
    clock.tick(FPS)
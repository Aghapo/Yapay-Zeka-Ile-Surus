import pygame
import sys
import math
import numpy as np

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Cars - Yarış Pisti")
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
ALL_WALLS = OUTER_SEGMENTS + INNER_SEGMENTS

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

class NeuralNetwork:
    def __init__(self, input_size=5, hidden_size=20, output_size=2):
        self.w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.w2 = np.random.uniform(-1, 1, (hidden_size, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        hidden = self.sigmoid(np.dot(inputs, self.w1))
        return self.sigmoid(np.dot(hidden, self.w2))

class Car:
    def __init__(self, x, y, nn=None):
        self.x = x
        self.y = y
        self.angle = 0
        self.velocity = 0
        self.acceleration = 0.2
        self.max_speed = 5
        self.friction = 0.95
        self.turn_speed = 3
        self.alive = True
        self.fitness = 0
        self.checkpoint_index = 0
        self.checkpoints_passed = 0
        self.frames_alive = 0
        self.frames_since_checkpoint = 0
        self.sensor_lines = []

        self.nn = nn if nn else NeuralNetwork()

        self.image = pygame.Surface((40, 20), pygame.SRCALPHA)
        self.image.fill((0, 200, 255))
        pygame.draw.rect(self.image, (255, 50, 50), (30, 5, 10, 10))

        self.sensor_angles = [-90, -45, 0, 45, 90]
        self.sensor_length = 150
        self.sensor_readings = [1.0] * 5

    def cast_sensors(self):
        self.sensor_lines = []
        self.sensor_readings = []
        for s_angle in self.sensor_angles:
            angle_rad = math.radians(self.angle + s_angle)
            end_x = self.x + math.cos(angle_rad) * self.sensor_length
            end_y = self.y - math.sin(angle_rad) * self.sensor_length

            closest = None
            closest_dist = self.sensor_length

            for wall in ALL_WALLS:
                hit = line_intersection(
                    (self.x, self.y), (end_x, end_y),
                    wall[0], wall[1]
                )
                if hit:
                    dist = math.hypot(hit[0]-self.x, hit[1]-self.y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest = hit

            self.sensor_readings.append(closest_dist / self.sensor_length)
            self.sensor_lines.append((end_x, end_y, closest))

    def check_checkpoint(self):
        target = CHECKPOINTS[self.checkpoint_index]
        dist = math.hypot(self.x - target[0], self.y - target[1])
        if dist < 80:
            self.checkpoints_passed += 1
            self.checkpoint_index = (self.checkpoint_index + 1) % len(CHECKPOINTS)
            self.frames_since_checkpoint = 0

    def check_collision(self):
        for wall in ALL_WALLS:
            hit = line_intersection(
                (self.x - 10, self.y), (self.x + 10, self.y),
                wall[0], wall[1]
            )
            if hit:
                self.alive = False
                return
            hit = line_intersection(
                (self.x, self.y - 10), (self.x, self.y + 10),
                wall[0], wall[1]
            )
            if hit:
                self.alive = False

    def update(self):
        if not self.alive:
            return

        self.cast_sensors()

        inputs = np.array(self.sensor_readings)
        decision = self.nn.forward(inputs)

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

        self.frames_alive += 1
        self.frames_since_checkpoint += 1

        if self.frames_since_checkpoint > 300:
            self.alive = False

        self.fitness = self.checkpoints_passed * 1000 + self.frames_alive

        self.check_checkpoint()
        self.check_collision()

    def draw(self):
        if not self.alive:
            return

        for end_x, end_y, closest in self.sensor_lines:
            if closest:
                pygame.draw.line(screen, (255, 50, 50),
                    (int(self.x), int(self.y)), (int(closest[0]), int(closest[1])), 1)
            else:
                pygame.draw.line(screen, (0, 255, 100),
                    (int(self.x), int(self.y)), (int(end_x), int(end_y)), 1)

        rotated = pygame.transform.rotate(self.image, self.angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect.topleft)

        end_x = self.x + math.cos(math.radians(self.angle)) * 30
        end_y = self.y - math.sin(math.radians(self.angle)) * 30
        pygame.draw.line(screen, (255,255,0), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)

class GeneticAlgorithm:
    def select(self, fitnesses, top_k=5):
        return np.argsort(fitnesses)[::-1][:top_k]

    def mutate(self, nn, mutation_rate=0.1):
        child = NeuralNetwork()
        child.w1 = nn.w1 + np.random.uniform(-mutation_rate, mutation_rate, nn.w1.shape)
        child.w2 = nn.w2 + np.random.uniform(-mutation_rate, mutation_rate, nn.w2.shape)
        return child

    def new_generation(self, cars, fitnesses, generation):
        best_indices = self.select(fitnesses)
        new_cars = []

        best_car = cars[best_indices[0]]
        elite = Car(START_X, START_Y, best_car.nn)
        new_cars.append(elite)

        mutation_rate = max(0.05, 0.3 - generation * 0.005)

        for i in range(1, 20):
            parent = cars[best_indices[i % len(best_indices)]]
            child_nn = self.mutate(parent.nn, mutation_rate)
            new_cars.append(Car(START_X, START_Y, child_nn))

        return new_cars

START_X, START_Y = CHECKPOINTS[0]

ga = GeneticAlgorithm()
cars = [Car(START_X, START_Y) for _ in range(20)]
generation = 1
font = pygame.font.SysFont(None, 28)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
        # Grafik çiz
            import matplotlib.pyplot as plt
            generations = [d["generation"] for d in fitness_history]
            bests = [d["best"] for d in fitness_history]
            avgs = [d["avg"] for d in fitness_history]
            
            plt.figure(figsize=(10, 5))
            plt.plot(generations, bests, label="En iyi fitness", color="cyan", linewidth=2)
            plt.plot(generations, avgs, label="Ortalama fitness", color="orange", linewidth=2, linestyle="--")
            plt.xlabel("Nesil")
            plt.ylabel("Fitness")
            plt.title("AI Cars — Öğrenme Eğrisi")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("ogrenme_egrisi.png", dpi=150)
            plt.show()
            
            pygame.quit()
            sys.exit()

    for car in cars:
        car.update()

    if all(not car.alive for car in cars):
        fitnesses = [car.fitness for car in cars]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)  # ortalama da izle
        mutation_rate = max(0.05, 0.3 - generation * 0.005)
        
        # Veriyi kaydet
        fitness_history.append({
            "generation": generation,
            "best": best_fitness,
            "avg": avg_fitness
        })
        
        print(f"Nesil {generation} — En iyi: {best_fitness:.0f} — Ort: {avg_fitness:.0f} — Mutasyon: {mutation_rate:.3f}")
        cars = ga.new_generation(cars, fitnesses, generation)
        generation += 1

    screen.fill((20, 20, 20))

    pygame.draw.polygon(screen, (60, 60, 60), TRACK_OUTER)
    pygame.draw.polygon(screen, (20, 20, 20), TRACK_INNER)
    pygame.draw.lines(screen, (180, 180, 180), True, TRACK_OUTER, 2)
    pygame.draw.lines(screen, (180, 180, 180), True, TRACK_INNER, 2)

    for i, cp in enumerate(CHECKPOINTS):
        color = (255, 255, 0) if i == 0 else (100, 100, 255)
        pygame.draw.circle(screen, color, cp, 8)

    for car in cars:
        car.draw()

    alive_count = sum(1 for car in cars if car.alive)
    best = max(car.fitness for car in cars)
    screen.blit(font.render(f"Nesil: {generation}", True, (255,255,255)), (10, 10))
    screen.blit(font.render(f"Hayatta: {alive_count}", True, (255,255,255)), (10, 35))
    screen.blit(font.render(f"En iyi: {best:.0f}", True, (255,255,255)), (10, 60))

    pygame.display.flip()
    clock.tick(FPS)
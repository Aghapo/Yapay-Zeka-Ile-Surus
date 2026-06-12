import pygame
import sys
import math
import numpy as np
import os
import pickle
import random # Patlama efektleri için eklendi
from neat import NEATPopulation

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Cars - Premium Simülatör")
clock = pygame.time.Clock()
FPS = 60

# --- HARİTA (SEVİYE) SİSTEMİ ---
MAPS = [
    {
        "outer": [(130,100),(313,38),(547,54),(702,146),(742,331),(697,472),(583,539),(249,538),(76,417),(66,202)],
        "inner": [(220,190),(344,127),(548,151),(630,241),(634,337),(617,394),(493,449),(286,421),(186,381),(173,247)],
        "obstacles": []
    },
    {
        "outer": [(50,50), (750,50), (750,550), (50,550)], 
        "inner": [(200,200), (600,200), (600,400), (200,400)],
        # Pistin tam ortasında duran engel blokları!
        "obstacles": [ [(350,100), (450,100), (450,150), (350,150)], [(350,450), (450,450), (450,500), (350,500)] ]
    },
    {
        "outer": [(50,50), (750,50), (750,250), (300,250), (300,350), (750,350), (750,550), (50,550)],
        "inner": [(150,150), (650,150), (650,170), (200,170), (200,430), (650,430), (650,450), (150,450)],
        "obstacles": []
    },
    {
        "outer": [(50, 50), (750, 50), (750, 550), (450, 550), (450, 250), (300, 250), (300, 550), (50, 550)],
        "inner": [(150, 150), (650, 150), (650, 450), (550, 450), (550, 150), (200, 150), (200, 450), (150, 450)],
        "obstacles": [ [(350, 100), (400, 100), (400, 130), (350, 130)] ] # Zorlu dönüşte bir engel
    }
]

TARGET_LAPS = 3 
current_map_index = 0
TRACK_OUTER, TRACK_INNER, ALL_WALLS, CHECKPOINTS, START_X, START_Y = [], [], [], [], 0, 0
OBSTACLES = []

# --- PARÇACIK (PATLAMA) SİSTEMİ ---
particles = []

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-4, 4)
        self.vy = random.uniform(-4, 4)
        self.timer = random.randint(15, 30)
        # Ateş renkleri (Sarı, Turuncu, Kırmızı)
        self.color = (255, random.randint(50, 200), 0)
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.timer -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.timer > 0 and self.size > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

def make_segments(points):
    return [(points[i], points[(i+1) % len(points)]) for i in range(len(points))]

def load_map(index):
    global TRACK_OUTER, TRACK_INNER, ALL_WALLS, CHECKPOINTS, START_X, START_Y, OBSTACLES
    track = MAPS[index % len(MAPS)]
    TRACK_OUTER = track["outer"]
    TRACK_INNER = track["inner"]
    OBSTACLES = track.get("obstacles", [])
    
    OUTER_SEGMENTS = make_segments(TRACK_OUTER)
    INNER_SEGMENTS = make_segments(TRACK_INNER)
    
    # Engelleri de duvar listesine ekliyoruz ki sensörler algılasın
    OBS_SEGMENTS = []
    for obs in OBSTACLES:
        OBS_SEGMENTS.extend(make_segments(obs))
        
    ALL_WALLS = OUTER_SEGMENTS + INNER_SEGMENTS + OBS_SEGMENTS
    CHECKPOINTS = [(int((o[0]+i[0])/2), int((o[1]+i[1])/2)) for o, i in zip(TRACK_OUTER, TRACK_INNER)]
    START_X, START_Y = CHECKPOINTS[0]

load_map(current_map_index)
fitness_history = []

def line_intersection(p1, p2, p3, p4):
    x1,y1 = p1; x2,y2 = p2
    x3,y3 = p3; x4,y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0: return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    if 0 < t < 1 and 0 < u < 1: return (x1 + t*(x2-x1), y1 + t*(y2-y1))
    return None

class Car:
    def __init__(self, x, y, genome):
        self.x = x
        self.y = y
        self.angle = 0
        self.velocity = 0
        self.acceleration = 0.25
        self.max_speed = 7 
        self.friction = 0.95
        self.turn_speed = 4.5 
        self.max_reverse_speed = 2.0

        self.stuck_frames = 0 
        self.min_dist_to_target = 10000

        self.alive = True
        self.completed = False
        self.fitness = 0
        self.checkpoint_index = 0
        self.checkpoints_passed = 0
        
        self.speed_bonus = 0 
        self.frames_since_checkpoint = 0
        self.slow_frames = 0
        self.sensor_lines = []
        self.trail = [] 
        self.genome = genome
        
        self.current_accel = 0
        self.current_turn = 0

        self.sensor_angles = [-90, -45, 0, 45, 90] 
        self.sensor_length = 200 
        self.sensor_readings = [1.0] * 5

    def cast_sensors(self):
        self.sensor_lines = []
        self.sensor_readings = []
        for s_angle in self.sensor_angles:
            angle_rad = math.radians(self.angle + s_angle)
            end_x = self.x + math.cos(angle_rad) * self.sensor_length
            end_y = self.y - math.sin(angle_rad) * self.sensor_length
            closest_point = None
            closest_dist = self.sensor_length

            for wall in ALL_WALLS:
                hit = line_intersection((self.x, self.y), (end_x, end_y), wall[0], wall[1])
                if hit:
                    dist = math.hypot(hit[0] - self.x, hit[1] - self.y)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_point = hit

            self.sensor_readings.append(closest_dist / self.sensor_length)
            draw_end = closest_point if closest_point else (end_x, end_y)
            self.sensor_lines.append((draw_end, closest_point is not None))

    def get_corners(self):
        angle_rad = math.radians(-self.angle)
        w, h = 18, 9
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        corners = []
        for dx, dy in [(-w,-h),(w,-h),(w,h),(-w,h)]:
            cx = self.x + dx * cos_a - dy * sin_a
            cy = self.y + dx * sin_a + dy * cos_a
            corners.append((cx, cy))
        return corners

    def check_checkpoint(self):
        target = CHECKPOINTS[self.checkpoint_index]
        dist = math.hypot(self.x - target[0], self.y - target[1])
        if dist < 80:
            self.checkpoints_passed += 1

            self.stuck_frames = 0
            self.min_dist_to_target = 10000
            
            time_saved = max(0, 300 - self.frames_since_checkpoint)
            self.speed_bonus += time_saved * 10 
            self.frames_since_checkpoint = 0
            
            if self.checkpoints_passed >= len(CHECKPOINTS) * TARGET_LAPS:
                self.completed = True
                self.alive = False
                self.fitness += 50000 
            else:
                self.checkpoint_index = (self.checkpoint_index + 1) % len(CHECKPOINTS)

    def check_collision(self):
        for p1, p2 in zip(self.get_corners(), self.get_corners()[1:] + [self.get_corners()[0]]):
            for wall in ALL_WALLS:
                if line_intersection(p1, p2, wall[0], wall[1]):
                    self.alive = False
                    # ARABA ÖLDÜĞÜNDE PATLAMA EFEKTİ OLUŞTUR!
                    if not showcase_mode or fast_forward == False: # Hızlı çekimde efektleri kapat (kasmasın)
                        for _ in range(20):
                            particles.append(Particle(self.x, self.y))
                    return

    def update(self):
        if not self.alive: return

        # --- 1. HAREKETSİZLİK (DURMA) CEZASI ---
        if abs(self.velocity) < 0.5: 
            self.slow_frames += 1
        else: 
            self.slow_frames = 0
            
        # Eğer araba 60 kare (1 saniye) boyunca durursa veya çok yavaşlarsa ÖLDÜR!
        if self.slow_frames > 60:
            self.alive = False
            if not showcase_mode or fast_forward == False:
                for _ in range(20):
                    particles.append(Particle(self.x, self.y))
            return
        
        # Sensörleri at ve kararı al
        self.cast_sensors()
        speed_normalized = abs(self.velocity) / self.max_speed
        inputs = self.sensor_readings + [speed_normalized]
        decision = self.genome.forward(inputs)

        # Hız ve direksiyon (Analog)
        self.current_accel = (decision[0]) 
        self.current_turn = (decision[1] - 0.5) * 2  

        if self.current_accel < 0:
            self.current_accel *= 0.3 

        self.velocity += self.acceleration * self.current_accel
        self.velocity = max(0, min(self.max_speed, self.velocity))
        if self.velocity > 0.5:
            self.angle += self.turn_speed * self.current_turn
        self.velocity *= self.friction

        # Hareketi uygula
        self.x += math.cos(math.radians(self.angle)) * self.velocity
        self.y -= math.sin(math.radians(self.angle)) * self.velocity

        self.trail.append((self.x, self.y))
        if len(self.trail) > 40: self.trail.pop(0)

        self.cast_sensors()

        # --- YENİ: ANTİ-SPİN (TAKILMA VE DÖNME ENGELLEYİCİ) ---
        target = CHECKPOINTS[self.checkpoint_index]
        dist_to_target = math.hypot(self.x - target[0], self.y - target[1])

        # Eğer araba hedefe, eski rekorundan daha fazla yaklaştıysa sorun yok
        if dist_to_target < self.min_dist_to_target:
            self.min_dist_to_target = dist_to_target
            self.stuck_frames = 0 # Sayacı sıfırla (İyi yolda gidiyor)
        else:
            self.stuck_frames += 1 # Hedefe yaklaşamıyor (Spin atıyor veya duvara takıldı)

        # Eğer araba 80 kare (yaklaşık 1.3 saniye) boyunca hedefe bir milim bile yaklaşamazsa onu ÖLDÜR!
        if self.stuck_frames > 80:
            self.alive = False
            return

        # --- YENİ: KUSURSUZ ÖDÜL (FITNESS) SİSTEMİ ---
        # Yaklaştıkça artan 0-1 arası bir çarpan (Sadece hedefe yaklaşmak puan getirir)
        proximity_multiplier = max(0.0, (1000 - dist_to_target) / 1000.0) 
        
        # Puan = (Geçtiği her checkpoint için 10 Puan) + (O anki hedefe yakınlığı)
        self.fitness = (self.checkpoints_passed * 10) + proximity_multiplier

        if hasattr(self.genome, "fitness"):
            self.genome.fitness = self.fitness

        self.check_checkpoint()
        self.check_collision()

    def draw(self, is_best=False, is_selected=False):
        if not self.alive and not self.completed: return

        is_highlighted = is_best or is_selected

        if is_highlighted and len(self.trail) > 2:
            trail_color = (100, 255, 100) if is_selected else (255, 100, 100)
            pygame.draw.lines(screen, trail_color, False, self.trail, 3)

        if is_highlighted and not self.completed:
            for (end_x, end_y), hit in self.sensor_lines:
                color = (255, 50, 50) if hit else (0, 255, 100)
                pygame.draw.line(screen, color, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 1)
                if hit: pygame.draw.circle(screen, (255, 255, 0), (int(end_x), int(end_y)), 4)

        img = pygame.Surface((40, 20), pygame.SRCALPHA)
        if is_selected:
            img.fill((50, 255, 50))
        elif is_best:
            img.fill((255, 215, 0))
        else:
            img.fill((0, 200, 255, 60))
            
        pygame.draw.rect(img, (255, 50, 50), (30, 5, 10, 10))

        rotated = pygame.transform.rotate(img, self.angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect.topleft)

# --- CANLI GRAFİK ÇİZİM FONKSİYONU ---
def draw_live_graph(surface, history):
    if len(history) < 2: return
    
    # Grafiğin arka planı (Sağ alt köşe)
    graph_w, graph_h = 220, 100
    graph_x, graph_y = WIDTH - graph_w - 20, HEIGHT - graph_h - 20
    pygame.draw.rect(surface, (20, 25, 30), (graph_x, graph_y, graph_w, graph_h), border_radius=5)
    pygame.draw.rect(surface, (100, 100, 100), (graph_x, graph_y, graph_w, graph_h), 2, border_radius=5)
    
    max_fit = max(h["best"] for h in history)
    if max_fit <= 0: max_fit = 1

    points = []
    gen_count = len(history)
    for i, h in enumerate(history):
        # Puanları X ve Y koordinatlarına dönüştür
        px = graph_x + (i / max(1, gen_count - 1)) * graph_w
        py = (graph_y + graph_h) - (h["best"] / max_fit) * graph_h
        # Grafiğin dışına taşmaması için ufak güvenlik payı
        py = max(graph_y, min(graph_y + graph_h, py)) 
        points.append((px, py))
        
    if len(points) > 1:
        pygame.draw.lines(surface, (0, 255, 200), False, points, 2)
        
    surface.blit(font.render("Öğrenme Eğrisi", True, (200,200,200)), (graph_x + 5, graph_y + 5))

def draw_hud(car, is_selected=False):
    hud_x, hud_y = WIDTH - 160, 20
    bg_color = (20, 50, 20) if is_selected else (30, 30, 30)
    
    pygame.draw.rect(screen, bg_color, (hud_x, hud_y, 140, 115), border_radius=10)
    pygame.draw.rect(screen, (100, 100, 100), (hud_x, hud_y, 140, 115), 2, border_radius=10)
    
    title = "SEÇİLİ ARAÇ" if is_selected else "LİDER ARAÇ"
    screen.blit(font.render(title, True, (255,255,255)), (hud_x + 10, hud_y + 10))
    
    current_lap = min(TARGET_LAPS, (car.checkpoints_passed // len(CHECKPOINTS)) + 1)
    screen.blit(font.render(f"Tur: {current_lap} / {TARGET_LAPS}", True, (255,215,0)), (hud_x + 10, hud_y + 30))
    
    bar_width = int(50 * abs(car.current_accel))
    color = (0, 255, 100) if car.current_accel > 0 else (255, 50, 50)
    pygame.draw.rect(screen, color, (hud_x + 10, hud_y + 55, bar_width, 15))
    pygame.draw.rect(screen, (255,255,255), (hud_x + 10, hud_y + 55, 50, 15), 1)

    center_x = hud_x + 60
    turn_offset = int(40 * car.current_turn)
    pygame.draw.line(screen, (255,255,255), (hud_x+20, hud_y+95), (hud_x+100, hud_y+95), 2)
    pygame.draw.circle(screen, (255, 215, 0), (center_x - turn_offset, hud_y + 95), 6)

# --- Ana Simülasyon ---
INPUT_SIZE = 6 
OUTPUT_SIZE = 2   
POPULATION_SIZE = 30 
MAX_GENERATION_FRAMES = 1800 

population = NEATPopulation(POPULATION_SIZE, INPUT_SIZE, OUTPUT_SIZE)
cars = [Car(START_X, START_Y, g) for g in population.genomes]

font = pygame.font.SysFont("arial", 16, bold=True)
large_font = pygame.font.SysFont("arial", 22, bold=True)

generation_frames = 0
leader_car = None
selected_car = None
fast_forward = False
showcase_mode = False

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: 
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for car in cars:
                    if car.alive and math.hypot(car.x - mouse_x, car.y - mouse_y) < 30:
                        selected_car = car
                        break
            elif event.button == 3: 
                selected_car = None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s and leader_car:
                with open("en_iyi_beyin.pkl", "wb") as f:
                    pickle.dump(leader_car.genome, f)
                print("KAYDEDİLDİ!")
            
            if event.key == pygame.K_l:
                if os.path.exists("en_iyi_beyin.pkl"):
                    with open("en_iyi_beyin.pkl", "rb") as f:
                        master_genome = pickle.load(f)
                    cars = [Car(START_X, START_Y, master_genome)]
                    showcase_mode = True
                    selected_car = None
                    print("USTA BEYİN YÜKLENDİ! ŞOV BAŞLIYOR!")
                else:
                    print("HATA: 'en_iyi_beyin.pkl' bulunamadı.")
                    
            if event.key == pygame.K_f:
                fast_forward = not fast_forward

    if showcase_mode:
        if not cars[0].alive or cars[0].completed:
            cars[0] = Car(START_X, START_Y, master_genome)
            particles.clear()

    simulation_speed = 30 if fast_forward and not showcase_mode else 1
    
    for _ in range(simulation_speed):
        generation_frames += 1
        for car in cars: car.update()

        if not showcase_mode:
            if any(car.completed for car in cars):
                print(f"Harita {current_map_index + 1} Başarıldı!")
                current_map_index += 1
                load_map(current_map_index)
                population.evolve()
                cars = [Car(START_X, START_Y, g) for g in population.genomes]
                generation_frames = 0
                leader_car = None
                selected_car = None
                particles.clear()
                break

            if all(not car.alive for car in cars) or generation_frames > MAX_GENERATION_FRAMES:
                best_fit = max(c.fitness for c in cars)
                fitness_history.append({"generation": population.generation, "best": best_fit})
                
                population.evolve()
                cars = [Car(START_X, START_Y, g) for g in population.genomes]
                generation_frames = 0
                leader_car = None
                selected_car = None
                particles.clear()
                break 

    # --- ÇİZİM ---
    screen.fill((25, 30, 35))

    pygame.draw.polygon(screen, (50, 55, 60), TRACK_OUTER)
    pygame.draw.polygon(screen, (25, 30, 35), TRACK_INNER)
    pygame.draw.lines(screen, (200, 50, 50), True, TRACK_OUTER, 4)
    pygame.draw.lines(screen, (200, 50, 50), True, TRACK_INNER, 4)
    
    # ENGELLERİ (KUTULARI) ÇİZ
    for obs in OBSTACLES:
        pygame.draw.polygon(screen, (200, 150, 0), obs)
        pygame.draw.lines(screen, (255, 200, 0), True, obs, 2)

    pygame.draw.circle(screen, (255, 255, 255), CHECKPOINTS[0], 6)
    pygame.draw.circle(screen, (0, 0, 0), CHECKPOINTS[0], 4)

    # Parçacıkları Çiz (Patlamalar)
    for p in particles[:]:
        p.update()
        p.draw(screen)
        if p.timer <= 0:
            particles.remove(p)

    if selected_car and not selected_car.alive:
        selected_car = None

    if not showcase_mode:
        alive_cars = [c for c in cars if c.alive]
        if alive_cars:
            current_max = max(alive_cars, key=lambda c: c.fitness)
            if leader_car not in alive_cars:
                leader_car = current_max
            elif current_max.fitness > leader_car.fitness + 50:
                leader_car = current_max
            best_car = leader_car
        else:
            best_car = cars[0]
            leader_car = None
    else:
        best_car = cars[0]
        alive_cars = cars

    for car in sorted(cars, key=lambda c: (c is best_car, c is selected_car)): 
        car.draw(is_best=(car is best_car), is_selected=(car is selected_car))

    # Arayüz ve Grafikler
    if not showcase_mode and not fast_forward:
        draw_live_graph(screen, fitness_history)

    if showcase_mode:
        screen.blit(large_font.render("USTA ŞOFÖR ŞOV MODU", True, (255,215,0)), (10, 10))
    else:
        best_score = max(car.fitness for car in cars) if cars else 0
        speed_txt = "HIZ: 50x (HIZLI ÇEKİM)" if fast_forward else "HIZ: 1x (Normal)"
        speed_color = (255, 100, 100) if fast_forward else (100, 255, 100)
        
        screen.blit(large_font.render(f"AI CAR EVOLUTION", True, (0,255,150)), (10, 10))
        screen.blit(font.render(f"Harita: {current_map_index + 1} / {len(MAPS)}", True, (200,200,200)), (10, 40))
        screen.blit(font.render(f"Nesil: {population.generation}", True, (200,200,200)), (10, 60))
        screen.blit(font.render(f"Hayatta: {len(alive_cars)} / {POPULATION_SIZE}", True, (200,200,200)), (10, 80))
        screen.blit(font.render(f"Skor: {best_score:.0f}", True, (255,215,0)), (10, 100))
        screen.blit(font.render(speed_txt, True, speed_color), (10, 120))
        
        screen.blit(font.render("[S] Kaydet | [L] Yükle | [F] Hızlandır | Sağ/Sol Tık Seçim", True, (150,150,150)), (10, HEIGHT - 30))

    target_hud_car = selected_car if selected_car else best_car
    if target_hud_car and target_hud_car.alive and not fast_forward:
        draw_hud(target_hud_car, is_selected=(selected_car is not None))

    pygame.display.flip()
    clock.tick(FPS)
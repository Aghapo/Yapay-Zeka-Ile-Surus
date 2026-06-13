import pygame
import sys
import math
import numpy as np
import os
import pickle
import random 
from neat import NEATPopulation

pygame.init()

# --- HD ÇÖZÜNÜRLÜK VE DİNAMİK ÖLÇEKLEME ---
WIDTH, HEIGHT = 1280, 720 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Cars - Premium Simülatör v12.0 (Gerçekçi Şanzıman & G-Force)")
clock = pygame.time.Clock()
FPS = 60

SCALE_X = WIDTH / 800.0
SCALE_Y = HEIGHT / 600.0
AVG_SCALE = (SCALE_X + SCALE_Y) / 2.0

# --- ARAYÜZ VE SİSTEM KONTROLLERİ ---
is_fullscreen = False
show_brain = False       
show_hud = True          
night_mode = False       
raining = False          
show_ghost = True        

all_time_best_fitness = -1
all_time_best_history = [] 
all_time_best_lap_frames = float('inf') 
ghost_frame = 0

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
        "obstacles": [ [(350, 100), (400, 100), (400, 130), (350, 130)] ] 
    }
]

TARGET_LAPS = 3 
current_map_index = 0
TRACK_OUTER, TRACK_INNER, ALL_WALLS = [], [], []
CHECKPOINTS, CHECKPOINT_GATES = [], []
START_X, START_Y = 0, 0
OBSTACLES = []

particles, exhaust_particles, rain_drops, smoke_particles = [], [], [], []

class Particle: 
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx = random.uniform(-4*AVG_SCALE, 4*AVG_SCALE)
        self.vy = random.uniform(-4*AVG_SCALE, 4*AVG_SCALE)
        self.timer = random.randint(15, 30)
        self.color = (255, random.randint(50, 200), 0)
        self.size = random.randint(2, 6) * AVG_SCALE
    def update(self):
        self.x += self.vx; self.y += self.vy
        self.timer -= 1; self.size = max(0, self.size - 0.2)
    def draw(self, surface):
        if self.timer > 0 and self.size > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

class ExhaustParticle: 
    def __init__(self, x, y, angle):
        self.x, self.y = x, y
        spread = math.radians(random.uniform(-15, 15))
        speed = random.uniform(1, 3) * AVG_SCALE
        self.vx = -math.cos(angle + spread) * speed
        self.vy = math.sin(angle + spread) * speed
        self.timer = random.randint(5, 15)
        self.color = random.choice([(255, 150, 0), (255, 200, 0), (50, 150, 255)]) 
        self.size = random.randint(2, 4) * AVG_SCALE
    def update(self):
        self.x += self.vx; self.y += self.vy
        self.timer -= 1; self.size = max(0, self.size - 0.3)
    def draw(self, surface):
        if self.timer > 0 and self.size > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

class SmokeParticle:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx = random.uniform(-1, 1) * AVG_SCALE
        self.vy = random.uniform(-1, 1) * AVG_SCALE
        self.timer = random.randint(20, 40)
        self.max_timer = self.timer
        self.size = random.uniform(3, 8) * AVG_SCALE
    def update(self):
        self.x += self.vx; self.y += self.vy
        self.timer -= 1; self.size += 0.3 * AVG_SCALE 
    def draw(self, surface):
        if self.timer > 0:
            alpha = int((self.timer / self.max_timer) * 80) 
            surf = pygame.Surface((int(self.size*2), int(self.size*2)), pygame.SRCALPHA)
            pygame.draw.circle(surf, (200, 200, 200, alpha), (int(self.size), int(self.size)), int(self.size))
            surface.blit(surf, (int(self.x - self.size), int(self.y - self.size)))

class RainDrop: 
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(-HEIGHT, 0)
        self.vy = random.uniform(10, 20) * AVG_SCALE
        self.length = random.uniform(10, 20) * AVG_SCALE
    def update(self):
        self.y += self.vy
        if self.y > HEIGHT:
            self.y = random.randint(-100, -10); self.x = random.randint(0, WIDTH)
    def draw(self, surface):
        pygame.draw.line(surface, (150, 180, 220, 150), (self.x, self.y), (self.x, self.y + self.length), 1)

def make_segments(points):
    return [(points[i], points[(i+1) % len(points)]) for i in range(len(points))]

def load_map(index):
    global TRACK_OUTER, TRACK_INNER, ALL_WALLS, CHECKPOINTS, CHECKPOINT_GATES, START_X, START_Y, OBSTACLES, all_time_best_history, all_time_best_lap_frames
    track = MAPS[index % len(MAPS)]
    
    TRACK_OUTER = [(o[0]*SCALE_X, o[1]*SCALE_Y) for o in track["outer"]]
    TRACK_INNER = [(i[0]*SCALE_X, i[1]*SCALE_Y) for i in track["inner"]]
    
    OBSTACLES = []
    for obs in track.get("obstacles", []): OBSTACLES.append([(p[0]*SCALE_X, p[1]*SCALE_Y) for p in obs])
        
    ALL_WALLS = make_segments(TRACK_OUTER) + make_segments(TRACK_INNER)
    for obs in OBSTACLES: ALL_WALLS.extend(make_segments(obs))
        
    CHECKPOINTS = [(int((o[0]+i[0])/2), int((o[1]+i[1])/2)) for o, i in zip(TRACK_OUTER, TRACK_INNER)]
    CHECKPOINT_GATES = list(zip(TRACK_OUTER, TRACK_INNER)) 
    START_X, START_Y = CHECKPOINTS[0]
    all_time_best_history = [] 
    all_time_best_lap_frames = float('inf')

load_map(current_map_index)
fitness_history = []

def line_intersection(p1, p2, p3, p4):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0: return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
    if 0 < t < 1 and 0 < u < 1: return (x1 + t*(x2-x1), y1 + t*(y2-y1))
    return None

class Car:
    def __init__(self, x, y, genome, generation, is_ghost=False):
        self.x, self.y = x, y
        self.angle = 0
        self.dx, self.dy = 0.0, 0.0 
        self.velocity = 0.0 
        self.actual_speed = 0.0 
        
        self.acceleration = 0.35 * AVG_SCALE
        self.max_speed = 11.0 * AVG_SCALE 
        self.base_friction = 0.98 
        self.friction = self.base_friction
        self.base_turn_speed = 5.0 
        self.turn_speed = self.base_turn_speed 

        # --- YENİ: VİTES KUTUSU VE G-FORCE ---
        self.gear = 1
        self.shift_delay = 0 # Vites atarken yaşanan güç kaybı süresi
        self.rpm = 1000
        self.g_x = 0.0 # G-Force Metre X
        self.g_y = 0.0 # G-Force Metre Y

        self.stuck_frames = 0 
        self.min_dist_to_target = 10000 * AVG_SCALE

        self.alive = True
        self.completed = False
        self.fitness = 0
        self.checkpoint_index = 0
        self.checkpoints_passed = 0
        
        self.speed_bonus = 0 
        self.frames_since_checkpoint = 0
        self.slow_frames = 0
        
        self.current_lap_frames = 0
        self.best_lap_frames = float('inf')
        
        self.skid_segments = []
        self.drifting_last_frame = False
        self.in_slipstream = False 
        self.is_braking = False 
        
        self.position_history = [] 
        self.telemetry = [] 
        
        self.sensor_lines = []
        self.genome = genome
        self.generation = generation 
        self.is_ghost = is_ghost 
        
        self.current_accel, self.current_turn = 0, 0
        self.target_angle_diff = 0 

        self.sensor_angles = [-90, -45, 0, 45, 90] 
        self.sensor_length = 250 * AVG_SCALE 
        self.sensor_readings = [1.0] * 5

    def get_car_color(self):
        if self.is_ghost: return (0, 255, 255, 120) 
        r = min(255, max(0, 20 + self.generation * 8))
        b = max(50, 255 - self.generation * 6)
        g = min(200, 100 + self.generation * 2)
        return (r, g, b, 80) 

    def cast_sensors(self):
        self.sensor_lines, self.sensor_readings = [], []
        for s_angle in self.sensor_angles:
            angle_rad = math.radians(self.angle + s_angle)
            end_x = self.x + math.cos(angle_rad) * self.sensor_length
            end_y = self.y - math.sin(angle_rad) * self.sensor_length
            closest_point, closest_dist = None, self.sensor_length

            for wall in ALL_WALLS:
                hit = line_intersection((self.x, self.y), (end_x, end_y), wall[0], wall[1])
                if hit:
                    dist = math.hypot(hit[0] - self.x, hit[1] - self.y)
                    if dist < closest_dist:
                        closest_dist, closest_point = dist, hit

            ratio = closest_dist / self.sensor_length
            self.sensor_readings.append(ratio)
            draw_end = closest_point if closest_point else (end_x, end_y)
            self.sensor_lines.append((draw_end, closest_point is not None, ratio))

    def get_corners(self):
        angle_rad = math.radians(-self.angle)
        w, h = 18 * AVG_SCALE, 9 * AVG_SCALE
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        corners = []
        for dx, dy in [(-w,-h),(w,-h),(w,h),(-w,h)]:
            cx = self.x + dx * cos_a - dy * sin_a
            cy = self.y + dx * sin_a + dy * cos_a
            corners.append((cx, cy))
        return corners

    def check_checkpoint(self):
        target = CHECKPOINTS[self.checkpoint_index]
        dist = math.hypot(self.x - target[0], self.y - target[1])
        if dist < (80 * AVG_SCALE):
            if self.checkpoint_index == 0 and self.checkpoints_passed > 0:
                if self.current_lap_frames < self.best_lap_frames:
                    self.best_lap_frames = self.current_lap_frames
                    global all_time_best_lap_frames
                    if self.best_lap_frames < all_time_best_lap_frames:
                        all_time_best_lap_frames = self.best_lap_frames
                self.current_lap_frames = 0 
                
            self.checkpoints_passed += 1
            self.stuck_frames = 0
            self.min_dist_to_target = 10000 * AVG_SCALE
            
            time_saved = max(0, 300 - self.frames_since_checkpoint)
            self.speed_bonus += time_saved * 10 
            self.frames_since_checkpoint = 0
            
            if self.checkpoints_passed >= len(CHECKPOINTS) * TARGET_LAPS:
                self.completed, self.alive = True, False
                self.fitness += 50000 
            else:
                self.checkpoint_index = (self.checkpoint_index + 1) % len(CHECKPOINTS)

    def check_collision(self):
        for p1, p2 in zip(self.get_corners(), self.get_corners()[1:] + [self.get_corners()[0]]):
            for wall in ALL_WALLS:
                if line_intersection(p1, p2, wall[0], wall[1]):
                    self.alive = False
                    if not showcase_mode and not fast_forward and not self.is_ghost: 
                        for _ in range(20): particles.append(Particle(self.x, self.y))
                    return

    def update(self, all_cars):
        if not self.alive: return
        
        self.position_history.append((self.x, self.y, self.angle, self.drifting_last_frame, self.is_braking))

        if self.actual_speed < 0.5 * AVG_SCALE: self.slow_frames += 1
        else: self.slow_frames = 0
            
        if self.slow_frames > 60:
            self.alive = False
            if not showcase_mode and not fast_forward and not self.is_ghost:
                for _ in range(20): particles.append(Particle(self.x, self.y))
            return
        
        self.cast_sensors()

        target = CHECKPOINTS[self.checkpoint_index]
        dx, dy = target[0] - self.x, target[1] - self.y
        target_angle = math.degrees(math.atan2(-dy, dx)) 
        
        angle_diff = (target_angle - self.angle) % 360
        if angle_diff > 180: angle_diff -= 360
        
        self.target_angle_diff = angle_diff 
        normalized_angle_diff = angle_diff / 180.0 

        speed_normalized = abs(self.actual_speed) / self.max_speed
        inputs = self.sensor_readings + [speed_normalized, normalized_angle_diff]
        decision = self.genome.forward(inputs)

        self.current_accel = (decision[0] - 0.5) * 2   
        self.current_turn = (decision[1] - 0.5) * 2    
        self.is_braking = self.current_accel < -0.1

        self.telemetry.append((self.current_accel, self.current_turn))
        if len(self.telemetry) > 60: self.telemetry.pop(0)

        # SLIPSTREAM
        self.in_slipstream = False
        for other in all_cars:
            if other is not self and other.alive:
                dist = math.hypot(self.x - other.x, self.y - other.y)
                if 20 * AVG_SCALE < dist < 200 * AVG_SCALE:
                    angle_to_other = math.degrees(math.atan2(-(other.y - self.y), other.x - self.x))
                    ang_diff = (angle_to_other - self.angle) % 360
                    if ang_diff > 180: ang_diff -= 360
                    if abs(ang_diff) < 20: 
                        self.in_slipstream = True
                        break
        
        temp_max_speed = self.max_speed * 1.15 if self.in_slipstream else self.max_speed
        speed_ratio = self.actual_speed / temp_max_speed

        # --- YENİ: GERÇEKÇİ ŞANZIMAN (Vites Geçişleri) ---
        if speed_ratio < 0.15: ideal_gear = 1
        elif speed_ratio < 0.35: ideal_gear = 2
        elif speed_ratio < 0.55: ideal_gear = 3
        elif speed_ratio < 0.75: ideal_gear = 4
        elif speed_ratio < 0.90: ideal_gear = 5
        else: ideal_gear = 6

        if self.gear != ideal_gear and self.shift_delay <= 0:
            self.shift_delay = 10 # Vites atarken kısa süreli güç kesintisi
            self.gear = ideal_gear
        
        if self.shift_delay > 0:
            self.shift_delay -= 1
            torque = 0.0 # Vites değişirken motor torku sıfırlanır!
        else:
            torque = max(0.2, 1.0 - (speed_ratio ** 1.5)) 

        if self.current_accel > 0: 
            self.velocity += self.acceleration * self.current_accel * torque
        else: 
            self.velocity += self.acceleration * self.current_accel * 2.5 

        self.velocity = max(0.0, min(temp_max_speed, self.velocity))
        self.velocity *= self.friction 

        turn_penalty = max(0.2, 1.0 - (speed_ratio ** 2))
        self.turn_speed = self.base_turn_speed * turn_penalty

        if self.actual_speed > 0.5 * AVG_SCALE:
            self.angle += self.turn_speed * self.current_turn

        target_dx = math.cos(math.radians(self.angle)) * self.velocity
        target_dy = -math.sin(math.radians(self.angle)) * self.velocity
        
        grip = 0.08 if raining else 0.20 
        self.dx = self.dx * (1 - grip) + target_dx * grip
        self.dy = self.dy * (1 - grip) + target_dy * grip
        
        self.x += self.dx; self.y += self.dy
        self.actual_speed = math.hypot(self.dx, self.dy)

        # --- YENİ: G-FORCE HESAPLAMALARI ---
        lat_g = self.current_turn * speed_ratio * 20
        lon_g = -self.current_accel * 15 if not self.is_braking else 20
        # Yumuşak geçiş
        self.g_x += (lat_g - self.g_x) * 0.2
        self.g_y += (lon_g - self.g_y) * 0.2

        if self.current_accel > 0.6 and not fast_forward and not self.is_ghost:
            angle_rad = math.radians(self.angle)
            rear_x = self.x - math.cos(angle_rad) * (18 * AVG_SCALE)
            rear_y = self.y + math.sin(angle_rad) * (18 * AVG_SCALE)
            exhaust_particles.append(ExhaustParticle(rear_x, rear_y, angle_rad))

        movement_angle = math.degrees(math.atan2(-self.dy, self.dx))
        slip_angle = abs((self.angle - movement_angle + 180) % 360 - 180)

        if self.actual_speed > (4.0 * AVG_SCALE) and slip_angle > 12 and not self.is_ghost:
            angle_rad = math.radians(self.angle)
            rear_x = self.x - math.cos(angle_rad) * (15 * AVG_SCALE)
            rear_y = self.y + math.sin(angle_rad) * (15 * AVG_SCALE)
            left_x = rear_x + math.cos(angle_rad + math.pi/2) * (8 * AVG_SCALE)
            left_y = rear_y - math.sin(angle_rad + math.pi/2) * (8 * AVG_SCALE)
            right_x = rear_x + math.cos(angle_rad - math.pi/2) * (8 * AVG_SCALE)
            right_y = rear_y - math.sin(angle_rad - math.pi/2) * (8 * AVG_SCALE)

            if not self.drifting_last_frame or len(self.skid_segments[-1]['left']) > 50:
                self.skid_segments.append({'left': [], 'right': [], 'timer': 200})

            self.skid_segments[-1]['left'].append((left_x, left_y))
            self.skid_segments[-1]['right'].append((right_x, right_y))
            self.drifting_last_frame = True
            
            if not fast_forward and random.random() < 0.5:
                smoke_particles.append(SmokeParticle(left_x, left_y))
                smoke_particles.append(SmokeParticle(right_x, right_y))
        else:
            self.drifting_last_frame = False

        self.cast_sensors()
        dist_to_target = math.hypot(self.x - target[0], self.y - target[1])

        if dist_to_target < self.min_dist_to_target:
            self.min_dist_to_target = dist_to_target; self.stuck_frames = 0 
        else: self.stuck_frames += 1 

        if self.stuck_frames > 80:
            self.alive = False
            if not showcase_mode and not fast_forward and not self.is_ghost:
                for _ in range(20): particles.append(Particle(self.x, self.y))
            return

        proximity_multiplier = max(0.0, ((1000*AVG_SCALE) - dist_to_target) / (1000*AVG_SCALE)) 
        self.fitness = (self.checkpoints_passed * 1000) + self.speed_bonus + proximity_multiplier

        if hasattr(self.genome, "fitness") and not self.is_ghost:
            self.genome.fitness = self.fitness

        self.check_checkpoint()
        self.check_collision()

    def draw(self, is_best=False, is_selected=False):
        if not self.alive and not self.completed: return

        is_highlighted = is_best or is_selected or self.is_ghost

        if is_highlighted and not self.is_ghost:
            for seg in self.skid_segments[:]:
                seg['timer'] -= 1
                if seg['timer'] <= 0:
                    self.skid_segments.remove(seg)
                else:
                    if len(seg['left']) > 1:
                        c_val = max(30, 255 - seg['timer']) if not night_mode else 10
                        trail_c = (50, 255, 50) if is_selected else (c_val, c_val, c_val)
                        pygame.draw.lines(screen, trail_c, False, seg['left'], max(2, int(3 * AVG_SCALE)))
                        pygame.draw.lines(screen, trail_c, False, seg['right'], max(2, int(3 * AVG_SCALE)))

        if is_highlighted and not self.completed and not night_mode and not self.is_ghost:
            for (end_x, end_y), hit, ratio in self.sensor_lines:
                r, g = int(255 * (1 - ratio)), int(255 * ratio)
                color = (r, g, 0)
                pygame.draw.line(screen, color, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 1)
                if hit: pygame.draw.circle(screen, color, (int(end_x), int(end_y)), int(3 * AVG_SCALE))

        if self.in_slipstream and self.actual_speed > (5.0 * AVG_SCALE) and not fast_forward and not self.is_ghost:
            for _ in range(2):
                wx = self.x + random.randint(int(-20*AVG_SCALE), int(20*AVG_SCALE))
                wy = self.y + random.randint(int(-20*AVG_SCALE), int(20*AVG_SCALE))
                ex = wx - math.cos(math.radians(self.angle)) * 40 * AVG_SCALE
                ey = wy + math.sin(math.radians(self.angle)) * 40 * AVG_SCALE
                pygame.draw.line(screen, (200, 255, 255), (wx, wy), (ex, ey), 1)

        if night_mode and self.alive and not self.completed and not self.is_ghost:
            light_len, light_spread = 350 * AVG_SCALE, math.radians(30) 
            l_x1 = self.x + math.cos(math.radians(self.angle) - light_spread) * light_len
            l_y1 = self.y - math.sin(math.radians(self.angle) - light_spread) * light_len
            l_x2 = self.x + math.cos(math.radians(self.angle) + light_spread) * light_len
            l_y2 = self.y - math.sin(math.radians(self.angle) + light_spread) * light_len
            
            light_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(light_surf, (255, 255, 150, 15), [(self.x, self.y), (l_x1, l_y1), (l_x2, l_y2)])
            screen.blit(light_surf, (0, 0))

        car_w, car_h = int(40 * AVG_SCALE), int(20 * AVG_SCALE)
        img = pygame.Surface((car_w, car_h), pygame.SRCALPHA)
        
        if is_selected: img.fill((50, 255, 50))
        elif is_best and not self.is_ghost: img.fill((255, 215, 0))
        else: img.fill(self.get_car_color())
            
        pygame.draw.rect(img, (15, 15, 15), (car_w*0.1, 0, car_w*0.2, car_h*0.2)) 
        pygame.draw.rect(img, (15, 15, 15), (car_w*0.7, 0, car_w*0.2, car_h*0.2)) 
        pygame.draw.rect(img, (15, 15, 15), (car_w*0.1, car_h*0.8, car_w*0.2, car_h*0.2)) 
        pygame.draw.rect(img, (15, 15, 15), (car_w*0.7, car_h*0.8, car_w*0.2, car_h*0.2)) 
        
        cockpit_color = (255, 150, 150) if night_mode else (255, 50, 50)
        if self.is_ghost: cockpit_color = (200, 255, 255)
        pygame.draw.rect(img, cockpit_color, (car_w*0.6, car_h*0.25, car_w*0.2, car_h*0.5)) 

        if self.is_braking:
            pygame.draw.circle(img, (255, 0, 0), (int(car_w*0.05), int(car_h*0.2)), int(3*AVG_SCALE))
            pygame.draw.circle(img, (255, 0, 0), (int(car_w*0.05), int(car_h*0.8)), int(3*AVG_SCALE))
            pygame.draw.circle(img, (255, 100, 100), (int(car_w*0.05), int(car_h*0.2)), int(1.5*AVG_SCALE))
            pygame.draw.circle(img, (255, 100, 100), (int(car_w*0.05), int(car_h*0.8)), int(1.5*AVG_SCALE))

        rotated = pygame.transform.rotate(img, self.angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect.topleft)
        
        # YENİ: Lider için Altın Taç
        if is_best and not self.is_ghost and not night_mode:
            crown_y = self.y - 30 * AVG_SCALE
            pygame.draw.polygon(screen, (255, 215, 0), [
                (self.x - 10, crown_y), (self.x - 15, crown_y - 15), 
                (self.x - 5, crown_y - 5), (self.x, crown_y - 15), 
                (self.x + 5, crown_y - 5), (self.x + 15, crown_y - 15), 
                (self.x + 10, crown_y)
            ])

def draw_ghost_car(surface, gx, gy, gangle, is_drifting, is_braking):
    if is_drifting and not fast_forward:
        angle_rad = math.radians(gangle)
        rear_x = gx - math.cos(angle_rad) * (15 * AVG_SCALE)
        rear_y = gy + math.sin(angle_rad) * (15 * AVG_SCALE)
        left_x = rear_x + math.cos(angle_rad + math.pi/2) * (8 * AVG_SCALE)
        left_y = rear_y - math.sin(angle_rad + math.pi/2) * (8 * AVG_SCALE)
        right_x = rear_x + math.cos(angle_rad - math.pi/2) * (8 * AVG_SCALE)
        right_y = rear_y - math.sin(angle_rad - math.pi/2) * (8 * AVG_SCALE)
        
        pygame.draw.circle(surface, (0, 200, 255), (int(left_x), int(left_y)), 2)
        pygame.draw.circle(surface, (0, 200, 255), (int(right_x), int(right_y)), 2)
        if random.random() < 0.3:
            smoke_particles.append(SmokeParticle(left_x, left_y))

    car_w, car_h = int(40 * AVG_SCALE), int(20 * AVG_SCALE)
    img = pygame.Surface((car_w, car_h), pygame.SRCALPHA)
    img.fill((0, 255, 255, 120)) 
    pygame.draw.rect(img, (15, 15, 15, 120), (car_w*0.1, 0, car_w*0.2, car_h*0.2)) 
    pygame.draw.rect(img, (15, 15, 15, 120), (car_w*0.7, 0, car_w*0.2, car_h*0.2)) 
    pygame.draw.rect(img, (15, 15, 15, 120), (car_w*0.1, car_h*0.8, car_w*0.2, car_h*0.2)) 
    pygame.draw.rect(img, (15, 15, 15, 120), (car_w*0.7, car_h*0.8, car_w*0.2, car_h*0.2)) 
    pygame.draw.rect(img, (200, 255, 255, 150), (car_w*0.6, car_h*0.25, car_w*0.2, car_h*0.5)) 
    
    if is_braking:
        pygame.draw.circle(img, (255, 0, 0, 150), (int(car_w*0.05), int(car_h*0.2)), int(3*AVG_SCALE))
        pygame.draw.circle(img, (255, 0, 0, 150), (int(car_w*0.05), int(car_h*0.8)), int(3*AVG_SCALE))

    rotated = pygame.transform.rotate(img, gangle)
    rect = rotated.get_rect(center=(int(gx), int(gy)))
    surface.blit(rotated, rect.topleft)
    if not night_mode:
        surface.blit(font.render("GHOST", True, (0, 255, 255)), (gx - 20, gy - 30))

def draw_f1_kerbs(surface, points, color1=(255,30,30), color2=(255,255,255), thickness=4):
    for i in range(len(points)):
        p1, p2 = points[i], points[(i+1)%len(points)]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        dist = math.hypot(dx, dy)
        if dist == 0: continue
        nx, ny = dx/dist, dy/dist
        seg_len = 12 * AVG_SCALE
        count = int(dist / seg_len)
        for j in range(count):
            c = color1 if (i+j)%2 == 0 else color2
            start = (p1[0] + nx*j*seg_len, p1[1] + ny*j*seg_len)
            end = (p1[0] + nx*(j+1)*seg_len, p1[1] + ny*(j+1)*seg_len)
            pygame.draw.line(surface, c, start, end, int(thickness*AVG_SCALE))

def draw_start_line(surface, p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist == 0: return
    nx, ny = dx/dist, dy/dist
    segment_len = 10 * AVG_SCALE
    count = int(dist / segment_len)
    for i in range(count):
        color = (255,255,255) if i%2==0 else (0,0,0)
        start = (p1[0] + nx*i*segment_len, p1[1] + ny*i*segment_len)
        end = (p1[0] + nx*(i+1)*segment_len, p1[1] + ny*(i+1)*segment_len)
        pygame.draw.line(surface, color, start, end, int(6*AVG_SCALE))

def format_time(frames):
    if frames == float('inf') or frames == 0: return "--:--"
    seconds = frames / FPS
    return f"{seconds:.2f}s"

# --- HUD VE GRAFİKLER ---
def draw_live_graph(surface, history):
    if len(history) < 2: return
    graph_w, graph_h = 240, 120
    graph_x, graph_y = WIDTH - graph_w - 20, HEIGHT - graph_h - 20
    pygame.draw.rect(surface, (20, 25, 30, 200), (graph_x, graph_y, graph_w, graph_h), border_radius=5)
    pygame.draw.rect(surface, (100, 100, 100), (graph_x, graph_y, graph_w, graph_h), 2, border_radius=5)
    
    max_fit = max(h["best"] for h in history)
    if max_fit <= 0: max_fit = 1

    points = []
    gen_count = len(history)
    for i, h in enumerate(history):
        px = graph_x + (i / max(1, gen_count - 1)) * graph_w
        py = (graph_y + graph_h) - (h["best"] / max_fit) * graph_h
        py = max(graph_y, min(graph_y + graph_h, py)) 
        points.append((px, py))
        
    if len(points) > 1: pygame.draw.lines(surface, (0, 255, 200), False, points, 2)
    surface.blit(font.render("Öğrenme Eğrisi", True, (200,200,200)), (graph_x + 5, graph_y + 5))

def draw_minimap(surface, cars, best_car, ghost_pos):
    mm_w, mm_h = 200, 150
    mm_x, mm_y = WIDTH - mm_w - 20, 260 
    
    pygame.draw.rect(surface, (20, 25, 30, 200), (mm_x, mm_y, mm_w, mm_h), border_radius=8)
    pygame.draw.rect(surface, (100, 100, 100), (mm_x, mm_y, mm_w, mm_h), 2, border_radius=8)
    
    sc_x, sc_y = mm_w / WIDTH, mm_h / HEIGHT
    outer_mm = [(mm_x + p[0]*sc_x, mm_y + p[1]*sc_y) for p in TRACK_OUTER]
    inner_mm = [(mm_x + p[0]*sc_x, mm_y + p[1]*sc_y) for p in TRACK_INNER]
    pygame.draw.lines(surface, (100, 100, 100), True, outer_mm, 1)
    pygame.draw.lines(surface, (100, 100, 100), True, inner_mm, 1)
    
    for car in cars:
        if car.alive:
            cx, cy = mm_x + car.x * sc_x, mm_y + car.y * sc_y
            color = (255, 215, 0) if car is best_car else (0, 200, 255)
            size = 4 if car is best_car else 2
            pygame.draw.circle(surface, color, (int(cx), int(cy)), size)
            
    if show_ghost and ghost_pos:
        gx, gy = mm_x + ghost_pos[0] * sc_x, mm_y + ghost_pos[1] * sc_y
        pygame.draw.circle(surface, (0, 255, 255), (int(gx), int(gy)), 4)
        
    surface.blit(font.render("Radar", True, (150,150,150)), (mm_x + 5, mm_y + 5))

def draw_brain(surface, genome, x, y, w, h):
    if not genome: return
    bg_surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(bg_surf, (20, 25, 30, 200), (0, 0, w, h), border_radius=8)
    surface.blit(bg_surf, (x, y))
    pygame.draw.rect(surface, (100, 100, 100), (x, y, w, h), 2, border_radius=8)
    surface.blit(font.render("Sinir Ağı", True, (200,200,200)), (x + 10, y + 10))

    inputs = [n for n in genome.nodes.values() if n.node_type == "input"]
    outputs = [n for n in genome.nodes.values() if n.node_type == "output"]
    hidden = [n for n in genome.nodes.values() if n.node_type == "hidden"]

    node_positions = {}
    for i, n in enumerate(inputs): node_positions[n.node_id] = (x + 30, y + 40 + i * ((h - 60) / max(1, len(inputs) - 1)))
    for i, n in enumerate(outputs): node_positions[n.node_id] = (x + w - 30, y + (h/2) - 20 + i * 40)
    for i, n in enumerate(hidden): node_positions[n.node_id] = (x + (w / 2), y + 40 + i * ((h - 60) / max(1, len(hidden) - 1)))

    for conn in genome.connections.values():
        if not conn.enabled: continue
        if conn.from_node in node_positions and conn.to_node in node_positions:
            p1, p2 = node_positions[conn.from_node], node_positions[conn.to_node]
            color = (0, 255, 100) if conn.weight > 0 else (255, 50, 50)
            thickness = max(1, min(4, int(abs(conn.weight) * 1.5)))
            pygame.draw.line(surface, color, p1, p2, thickness)

    for nid, pos in node_positions.items():
        node = genome.nodes[nid]
        color = (200, 200, 200)
        if node.node_type == "input": color = (100, 150, 255)
        if node.node_type == "output": color = (255, 150, 100)
        pygame.draw.circle(surface, color, (int(pos[0]), int(pos[1])), 6)
        pygame.draw.circle(surface, (0,0,0), (int(pos[0]), int(pos[1])), 6, 1)

def draw_hud(car, is_selected=False):
    hud_w, hud_h = 240, 240
    hud_x, hud_y = WIDTH - hud_w - 20, 10
    bg_color = (20, 50, 20, 200) if is_selected else (30, 30, 30, 200)
    
    bg_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
    pygame.draw.rect(bg_surf, bg_color, (0, 0, hud_w, hud_h), border_radius=10)
    screen.blit(bg_surf, (hud_x, hud_y))
    pygame.draw.rect(screen, (100, 100, 100), (hud_x, hud_y, hud_w, hud_h), 2, border_radius=10)
    
    title = "SEÇİLİ ARAÇ" if is_selected else "LİDER ARAÇ"
    screen.blit(font.render(title, True, (255,255,255)), (hud_x + 15, hud_y + 10))
    
    # YENİ: MODERN KADRAN ÇİZİMİ
    center_speed_x, center_speed_y = hud_x + 70, hud_y + 90
    radius = 45
    
    # Kadran Arkaplanı
    pygame.draw.circle(screen, (10, 10, 15), (center_speed_x, center_speed_y), radius)
    pygame.draw.circle(screen, (100, 100, 100), (center_speed_x, center_speed_y), radius, 2)
    
    # Çentikler
    for i in range(11):
        angle = math.pi + (i / 10.0) * math.pi
        x1 = center_speed_x + math.cos(angle) * (radius - 5)
        y1 = center_speed_y - math.sin(angle) * (radius - 5)
        x2 = center_speed_x + math.cos(angle) * radius
        y2 = center_speed_y - math.sin(angle) * radius
        color = (200, 0, 0) if i >= 8 else (200, 200, 200)
        pygame.draw.line(screen, color, (x1, y1), (x2, y2), 2)
    
    # İbre ve Hız
    speed_kph = int((car.actual_speed / car.max_speed) * 320) # 320 km/h simulasyonu
    speed_ratio = min(1.0, speed_kph / 320.0)
    speed_angle = math.pi - (speed_ratio * math.pi) 
    
    needle_x = center_speed_x + math.cos(speed_angle) * (radius - 8)
    needle_y = center_speed_y - math.sin(speed_angle) * (radius - 8)
    pygame.draw.line(screen, (255, 50, 50), (center_speed_x, center_speed_y), (needle_x, needle_y), 3)
    pygame.draw.circle(screen, (255, 255, 255), (center_speed_x, center_speed_y), 4)
    
    # Dijital Yazılar
    screen.blit(large_font.render(f"{speed_kph}", True, (255,255,255)), (center_speed_x - 15, center_speed_y - 20))
    screen.blit(pygame.font.SysFont("arial", 12).render("KM/H", True, (150,150,150)), (center_speed_x - 12, center_speed_y + 5))
    
    # VİTES GÖSTERGESİ
    gear_txt = "N" if car.actual_speed < 0.5 and car.current_accel <= 0 else str(car.gear)
    gear_color = (255, 215, 0) if car.shift_delay > 0 else (200, 200, 200)
    screen.blit(large_font.render(gear_txt, True, gear_color), (center_speed_x + 60, center_speed_y - 25))

    # YENİ: G-FORCE METRE
    gfx, gfy = hud_x + 180, hud_y + 90
    g_radius = 25
    pygame.draw.circle(screen, (30, 30, 30), (gfx, gfy), g_radius)
    pygame.draw.circle(screen, (100, 100, 100), (gfx, gfy), g_radius, 1)
    pygame.draw.line(screen, (50, 50, 50), (gfx-g_radius, gfy), (gfx+g_radius, gfy), 1)
    pygame.draw.line(screen, (50, 50, 50), (gfx, gfy-g_radius), (gfx, gfy+g_radius), 1)
    
    dot_x = gfx + car.g_x
    dot_y = gfy + car.g_y
    # Dot'ın dışarı taşmasını engelle
    dist_from_center = math.hypot(dot_x - gfx, dot_y - gfy)
    if dist_from_center > g_radius - 4:
        angle = math.atan2(dot_y - gfy, dot_x - gfx)
        dot_x = gfx + math.cos(angle) * (g_radius - 4)
        dot_y = gfy + math.sin(angle) * (g_radius - 4)
        
    pygame.draw.circle(screen, (0, 255, 255), (int(dot_x), int(dot_y)), 4)
    screen.blit(pygame.font.SysFont("arial", 12).render("G-Force", True, (150,150,150)), (gfx - 20, gfy + 30))

    # TELEMETRİ
    tel_y = hud_y + 160
    pygame.draw.rect(screen, (10, 10, 15), (hud_x + 10, tel_y, hud_w - 20, 60))
    if len(car.telemetry) > 1:
        accel_pts, turn_pts = [], []
        step_x = (hud_w - 20) / len(car.telemetry)
        for i, (accel, turn) in enumerate(car.telemetry):
            ax = hud_x + 10 + i * step_x
            accel_pts.append((ax, tel_y + 30 - (accel * 25)))
            turn_pts.append((ax, tel_y + 30 - (turn * 25)))
        pygame.draw.lines(screen, (0, 255, 100), False, accel_pts, 2) 
        pygame.draw.lines(screen, (255, 215, 0), False, turn_pts, 2)  
    screen.blit(font.render("Telemetri", True, (100,100,100)), (hud_x + 15, tel_y + 5))

# --- ANA SİMÜLASYON ---
INPUT_SIZE = 7 
OUTPUT_SIZE = 2   
POPULATION_SIZE = 50 
MAX_GENERATION_FRAMES = 1800 

population = NEATPopulation(POPULATION_SIZE, INPUT_SIZE, OUTPUT_SIZE)

cars = []
for i, g in enumerate(population.genomes):
    ox, oy = random.randint(int(-20*AVG_SCALE), int(20*AVG_SCALE)), random.randint(int(-20*AVG_SCALE), int(20*AVG_SCALE))
    cars.append(Car(START_X + ox, START_Y + oy, g, population.generation))

font = pygame.font.SysFont("arial", 16, bold=True)
large_font = pygame.font.SysFont("arial", 22, bold=True)

generation_frames = 0
leader_car = None
selected_car = None
fast_forward = False
showcase_mode = False

for _ in range(100): rain_drops.append(RainDrop())

while True:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: 
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for car in cars:
                    if car.alive and math.hypot(car.x - mouse_x, car.y - mouse_y) < (30*AVG_SCALE):
                        selected_car = car; break
            elif event.button == 3: selected_car = None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s and leader_car:
                with open("en_iyi_beyin.pkl", "wb") as f: pickle.dump(leader_car.genome, f)
                print("KAYDEDİLDİ!")
            if event.key == pygame.K_l:
                if os.path.exists("en_iyi_beyin.pkl"):
                    with open("en_iyi_beyin.pkl", "rb") as f: master_genome = pickle.load(f)
                    cars = [Car(START_X, START_Y, master_genome, population.generation)]
                    showcase_mode, selected_car = True, None
                    print("USTA BEYİN YÜKLENDİ! ŞOV BAŞLIYOR!")
            
            if event.key == pygame.K_f: fast_forward = not fast_forward
            if event.key == pygame.K_n: show_brain = not show_brain
            if event.key == pygame.K_h: show_hud = not show_hud
            if event.key == pygame.K_d: night_mode = not night_mode
            if event.key == pygame.K_w: raining = not raining
            if event.key == pygame.K_g: show_ghost = not show_ghost
                
            if event.key == pygame.K_m:
                is_fullscreen = not is_fullscreen
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN) if is_fullscreen else pygame.display.set_mode((WIDTH, HEIGHT))

    if showcase_mode:
        if not cars[0].alive or cars[0].completed:
            cars[0] = Car(START_X, START_Y, master_genome, population.generation)
            particles.clear(); exhaust_particles.clear(); smoke_particles.clear()

    simulation_speed = 30 if fast_forward and not showcase_mode else 1
    
    for _ in range(simulation_speed):
        generation_frames += 1
        
        if show_ghost and not showcase_mode: ghost_frame += 1
            
        for car in cars: car.update(cars)

        if not showcase_mode:
            if any(car.completed for car in cars):
                current_map_index += 1
                load_map(current_map_index)
                population.evolve()
                cars = [Car(START_X + random.randint(int(-20*AVG_SCALE), int(20*AVG_SCALE)), START_Y + random.randint(int(-20*AVG_SCALE), int(20*AVG_SCALE)), g, population.generation) for g in population.genomes]
                generation_frames, ghost_frame, leader_car, selected_car = 0, 0, None, None
                particles.clear(); exhaust_particles.clear(); smoke_particles.clear()
                break

            if all(not car.alive for car in cars) or generation_frames > MAX_GENERATION_FRAMES:
                best_fit = max(c.fitness for c in cars)
                current_best_car = max(cars, key=lambda c: c.fitness)
                fitness_history.append({"generation": population.generation, "best": best_fit})
                
                if best_fit > all_time_best_fitness:
                    all_time_best_fitness = best_fit
                    all_time_best_history = current_best_car.position_history.copy()
                    
                population.evolve()
                cars = [Car(START_X + random.randint(int(-20*AVG_SCALE), int(20*AVG_SCALE)), START_Y + random.randint(int(-20*AVG_SCALE), int(20*AVG_SCALE)), g, population.generation) for g in population.genomes]
                
                generation_frames, ghost_frame, leader_car, selected_car = 0, 0, None, None
                particles.clear(); exhaust_particles.clear(); smoke_particles.clear()
                break 

    # --- ÇİZİM ---
    if night_mode:
        bg_c = (15, 20, 15) 
        track_out_color, track_in_color, border_color = (30, 35, 40), bg_c, (100, 20, 20)
    else:
        bg_c = (34, 139, 34) if not raining else (24, 100, 24) 
        track_c = (60, 65, 70) if not raining else (45, 50, 55) 
        screen.fill(bg_c)
        track_out_color, track_in_color, border_color = track_c, bg_c, (255, 255, 255)

    screen.fill(bg_c)
    pygame.draw.polygon(screen, track_out_color, TRACK_OUTER)
    pygame.draw.polygon(screen, track_in_color, TRACK_INNER)
    
    draw_f1_kerbs(screen, TRACK_OUTER, color1=(200, 30, 30), color2=(230, 230, 230), thickness=4)
    draw_f1_kerbs(screen, TRACK_INNER, color1=(200, 30, 30), color2=(230, 230, 230), thickness=4)
    
    if len(CHECKPOINT_GATES) > 0:
        draw_start_line(screen, CHECKPOINT_GATES[0][0], CHECKPOINT_GATES[0][1])
    
    for obs in OBSTACLES:
        pygame.draw.polygon(screen, (200, 150, 0), obs)
        pygame.draw.lines(screen, (255, 200, 0), True, obs, max(2, int(2*AVG_SCALE)))

    if raining and not fast_forward:
        for drop in rain_drops: drop.update(); drop.draw(screen)

    if not fast_forward:
        for p in particles[:]:
            p.update(); p.draw(screen)
            if p.timer <= 0: particles.remove(p)
        for ep in exhaust_particles[:]:
            ep.update(); ep.draw(screen)
            if ep.timer <= 0: exhaust_particles.remove(ep)
        for sp in smoke_particles[:]:
            sp.update(); sp.draw(screen)
            if sp.timer <= 0: smoke_particles.remove(sp)

    if selected_car and not selected_car.alive: selected_car = None

    if not showcase_mode:
        alive_cars = [c for c in cars if c.alive]
        if alive_cars:
            current_max = max(alive_cars, key=lambda c: c.fitness)
            if leader_car not in alive_cars: leader_car = current_max
            elif current_max.fitness > leader_car.fitness + 50: leader_car = current_max
            best_car = leader_car
        else:
            best_car, leader_car = cars[0], None
    else:
        best_car, alive_cars = cars[0], cars

    target_hud_car = selected_car if selected_car else best_car
    if target_hud_car and target_hud_car.alive and not fast_forward and not night_mode:
        gate = CHECKPOINT_GATES[target_hud_car.checkpoint_index]
        pygame.draw.line(screen, (0, 255, 100), gate[0], gate[1], max(2, int(3*AVG_SCALE)))

    ghost_current_pos = None
    if show_ghost and all_time_best_history and not showcase_mode:
        idx = min(ghost_frame, len(all_time_best_history) - 1)
        gx, gy, gangle, is_drifting, is_braking = all_time_best_history[idx]
        ghost_current_pos = (gx, gy)
        draw_ghost_car(screen, gx, gy, gangle, is_drifting, is_braking)

    for car in sorted(cars, key=lambda c: (c is best_car, c is selected_car)): 
        car.draw(is_best=(car is best_car), is_selected=(car is selected_car))

    # --- ARAYÜZ (HUD) ÇİZİMLERİ ---
    if show_hud:
        if not showcase_mode:
            board_y = 190
            screen.blit(font.render("CANLI SIRALAMA", True, (0,255,200)), (10, board_y))
            top_cars = sorted(cars, key=lambda c: c.fitness, reverse=True)[:5]
            for i, c in enumerate(top_cars):
                color = (255, 215, 0) if i==0 else ((192, 192, 192) if i==1 else (205, 127, 50) if i==2 else (200,200,200))
                txt = f"{i+1}. Puan: {c.fitness:.0f}"
                screen.blit(font.render(txt, True, color), (10, board_y + 25 + i*25))

        if not showcase_mode and not fast_forward:
            draw_live_graph(screen, fitness_history)
            draw_minimap(screen, cars, best_car, ghost_current_pos)
            
            if show_brain and target_hud_car and hasattr(target_hud_car, "genome"):
                draw_brain(screen, target_hud_car.genome, 20, HEIGHT - 300, 260, 240)

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
            
            if target_hud_car:
                cur_time_str = format_time(target_hud_car.current_lap_frames)
                best_time_str = format_time(all_time_best_lap_frames)
                screen.blit(font.render(f"Tur Süresi: {cur_time_str}", True, (200, 200, 255)), (10, 100))
                screen.blit(font.render(f"En İyi Tur: {best_time_str}", True, (255, 215, 0)), (10, 120))
            
            screen.blit(font.render(f"Skor: {best_score:.0f}", True, (255,215,0)), (10, 140))
            screen.blit(font.render(speed_txt, True, speed_color), (10, 160))
            
            info_txt_1 = "[S] Kaydet | [L] Yükle | [F] Hızlandır | [D] Gece Modu | [W] Yağmur"
            info_txt_2 = "[G] Hayalet Araç | [N] Beyin | [H] HUD Gizle | [M] Tam Ekran"
            screen.blit(font.render(info_txt_1, True, (255,255,255)), (10, HEIGHT - 50))
            screen.blit(font.render(info_txt_2, True, (255,255,255)), (10, HEIGHT - 25))

        if target_hud_car and target_hud_car.alive and not fast_forward:
            draw_hud(target_hud_car, is_selected=(selected_car is not None))

    pygame.display.flip()
    clock.tick(FPS)
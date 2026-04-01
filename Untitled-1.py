import pygame
import sys

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pist Tasarımı - Dış duvar için tıkla, ENTER ile bitir")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

def draw_track(outer, inner, current):
    screen.fill((30, 30, 30))
    
    # Mevcut noktaları çiz
    if len(outer) > 1:
        pygame.draw.lines(screen, (255, 100, 0), True, outer, 2)
    if len(inner) > 1:
        pygame.draw.lines(screen, (0, 200, 255), True, inner, 2)
    
    for p in outer:
        pygame.draw.circle(screen, (255, 100, 0), p, 4)
    for p in inner:
        pygame.draw.circle(screen, (0, 200, 255), p, 4)
    for p in current:
        pygame.draw.circle(screen, (255, 255, 255), p, 4)
    if len(current) > 1:
        pygame.draw.lines(screen, (255, 255, 255), False, current, 1)
    
    pygame.display.flip()

outer = []
inner = []
phase = "outer"  # önce dış, sonra iç

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if phase == "outer":
                outer.append(event.pos)
            else:
                inner.append(event.pos)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if phase == "outer" and len(outer) > 2:
                    phase = "inner"
                    pygame.display.set_caption("İç duvar için tıkla, ENTER ile bitir")
                elif phase == "inner" and len(inner) > 2:
                    # Koordinatları yazdır
                    print("TRACK_OUTER =", outer)
                    print("TRACK_INNER =", inner)
                    pygame.quit()
                    sys.exit()

            if event.key == pygame.K_z:  # geri al
                if phase == "outer" and outer:
                    outer.pop()
                elif phase == "inner" and inner:
                    inner.pop()

    current = outer if phase == "outer" else inner
    
    hint = "DIŞ DUVAR: Tıkla nokta ekle | Z: geri al | ENTER: bitir" if phase == "outer" \
           else "İÇ DUVAR: Tıkla nokta ekle | Z: geri al | ENTER: bitir"
    
    draw_track(outer, inner, current)
    text = font.render(hint, True, (200, 200, 200))
    screen.blit(text, (10, 10))
    pygame.display.flip()
    clock.tick(60)
import pygame
from   physical_elements import charge_distribution
from event_manager import Notifier


screen_x=1280
screen_y=1200


pygame.init()
screen = pygame.display.set_mode((screen_x, screen_y))
pygame.display.set_caption("Draw field lines")
clock = pygame.time.Clock()

running=True

notifier=Notifier()

def quit_simulation():
    pygame.quit()
    quit()

notifier.subscribe(pygame.QUIT,quit_simulation)

distribution=charge_distribution([])

notifier.subscribe(pygame.MOUSEBUTTONDOWN,distribution)
notifier.subscribe(pygame.KEYUP,distribution)
notifier.subscribe(pygame.MOUSEWHEEL,distribution)
notifier.subscribe(pygame.MOUSEMOTION,distribution)
notifier.subscribe(pygame.MOUSEBUTTONUP,distribution)


while running:
    screen.fill("white")
    notifier.dispatch(pygame.event.get())
    for q in distribution.charges:
        q.draw()
    
   
    
    distribution.E.draw_field_lines()
    
    pygame.display.flip()    
    clock.tick(20)

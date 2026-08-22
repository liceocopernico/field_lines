import pygame
import math
import numpy as np
from scipy.integrate import solve_ivp

box_x=1.0
box_y=1.0
k=1



def to_screen(x,y,screen_size,box_size):
    return (screen_size[0]/2+x/box_size[0]*screen_size[0],screen_size[1]/2-y/box_size[1]*screen_size[1])


def to_box(x,y,screen_size,box_size):
    return(x*box_size[0]/screen_size[0]-box_size[0]/2,box_size[1]/2-y*box_size[1]/screen_size[1])


class Field:
    def __init__(self,charges):
        self.screen=pygame.display.get_surface()
        self.screen_size=pygame.display.get_surface().get_size()
        self.charges=charges
        self.probes=[]
        self.field_lines=[]
        self.scale=0.2
        

    def _draw_arrow_head(self,size,angle,color):
        
        surface=pygame.Surface((size,size),pygame.SRCALPHA, 32 )    
        pygame.gfxdraw.filled_polygon( surface, ((size,0.5*size),(0,0),(0,size) ),color)
        surface=pygame.transform.rotate(surface,angle)
        
        return surface
  
    
    def draw(self,x,y):
        x_versor=pygame.Vector2(1,0)
        m_f=10
        field=self.compute_field(x,y)

        start=to_screen(x,y,self.screen_size,(box_x,box_y))
        end=to_screen(x+0.5*field.x/m_f*box_x*self.scale,   y+field.y/m_f*box_y*self.scale,self.screen_size,(box_x,box_y))
        screen_field=pygame.Vector2(end[0]-start[0],end[1]-start[1])
        pygame.draw.line(self.screen,'orange',start,end,2)
        angle=screen_field.angle_to(x_versor)
        arrow=self._draw_arrow_head(20,angle,(255,0,0))
        arrow_size=arrow.get_size()
        self.screen.blit(arrow,(end[0]-0.5*arrow_size[0],end[1]-0.5*arrow_size[1]))
        
        
    

    
    def draw_probes(self):
        for probe in self.probes:
            self.draw(probe[0],probe[1])
        
    def draw_field_lines(self):
        
        
        for field_line in self.field_lines:
            i=0
            samples=len(field_line.y[0])
            while i < samples-1:
                start=to_screen(field_line.y[0][i],field_line.y[1][i],self.screen_size,(box_x,box_y))
                end=to_screen(field_line.y[0][i+1],field_line.y[1][i+1],self.screen_size,(box_x,box_y))
                pygame.draw.aaline(self.screen,(255*(samples-i)/samples,0,255*i/samples),start,end,4)
                i+=1
                  
    def compute_field2(self,t,r):
            x,y=r
            field=[0.0,0.0]
            for charge in self.charges:
                 r_vector_x=x-charge.r.x
                 r_vector_y=y-charge.r.y
                 dist=math.sqrt(r_vector_x**2+r_vector_y**2)
                 vers_r_x=r_vector_x/dist
                 vers_r_y=r_vector_y/dist
                 field[0]+=k*charge.q/dist**2*vers_r_x
                 field[1]+=k*charge.q/dist**2*vers_r_y
            field_m=math.sqrt(field[0]**2+field[1]**2)
            field[0]=field[0]/field_m
            field[1]=field[1]/field_m
            
            return field

    def refresh_field_lines(self):
        self.field_lines=[]
        for probe in self.probes:
            self.get_field_line(probe[0],probe[1])
               
    def get_field_line(self,x,y):
        t_max=2
        t_span=(0,t_max)
        r_0=[x,y]
        field_line=solve_ivp(self.compute_field2,t_span,r_0,t_eval=np.linspace(0,t_max,150),method='RK23')
        self.field_lines.append(field_line)
        return field_line
    
    def compute_field(self,x,y):
        f_pos=pygame.Vector2(x,y)
        field=pygame.Vector2(0,0)

        for charge in self.charges:
            r_vector=f_pos-charge.r
            r=r_vector.length()
            vers_r=r_vector/r
            field+=k*charge.q/r**2*vers_r
        return field

class charge_distribution:
    def __init__(self,charges):
        self.charges=charges
        self.screen=pygame.display.get_surface()
        self.screen_size=pygame.display.get_surface().get_size()
        self.E=Field(self.charges)
        self.anchor=None
     
    
    def handle_event(self,event):
        def add_charge(magnitude):
            screen_position=pygame.mouse.get_pos()
            box_position=to_box(screen_position[0],screen_position[1],self.screen_size,(box_x,box_y))
            self.charges.append(charge(box_position[0],box_position[1],magnitude))
            self.E.charges=self.charges
            self.E.refresh_field_lines()
        match event.type:
            case pygame.KEYUP:
                match event.key:
                    case pygame.K_c:
                        add_charge(0.1)
                    case pygame.K_m:
                        add_charge(-0.1)
                    case pygame.K_t:
                        screen_position=pygame.mouse.get_pos()
                        for q in self.charges:
                         if q.get_collision(screen_position):
                              q.q=-q.q
                              q.resize_charge(0)
                              self.E.refresh_field_lines()
                    case pygame.K_r:
                        self.charges=[]
                        self.E.probes=[]
                        self.E.charges=self.charges
                        self.E.field_lines=[]
                                        
            case pygame.MOUSEWHEEL:
                    screen_position=pygame.mouse.get_pos()
                    for q in self.charges:
                         if q.get_collision(screen_position):
                              print(event.y)
                              q.resize_charge(event.y)
                              self.E.refresh_field_lines()

            case pygame.MOUSEBUTTONDOWN:
                if event.button==1:
                   for q in self.charges:
                        if q.get_collision(event.pos):
                            print("Too close to a charge")
                            self.anchor=q
                            break
                        else: 
                            probe_position=to_box(event.pos[0],event.pos[1],self.screen_size,(box_x,box_y))
                            self.E.probes.append(probe_position)
                            self.E.get_field_line(probe_position[0],probe_position[1])
            
            case pygame.MOUSEMOTION:    
                if self.anchor:
                    self.anchor.move_screen(event.pos)
                    
            
            
            case pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if self.anchor:
                        self.E.probes=[]
                        self.E.field_lines=[]
                        print(f"Anchor released")
                    self.anchor=None       

    

class charge:
    def __init__(self,x,y,q):
        
        self.r=pygame.Vector2(x,y)
        self.q=q
        
        if (q>0):
            self.color=(255,0,0)
        else:
            self.color=(0,0,255)
        self.rectangle=pygame.Rect(0,0,0,0)    
        self.screen=pygame.display.get_surface()
        self.screen_size=pygame.display.get_surface().get_size()
        
        self.max_q=10.0
        self.q_radius=abs(q)/self.max_q*500
    
    
    def move_screen(self,r):
        x,y=r
        new_position=to_box(x,y,self.screen_size,(box_x,box_y))
        self.r.x=new_position[0]
        self.r.y=new_position[1]
        
    def resize_charge(self,step):
        
        self.q+=0.03*step
        
        
        
        if (self.q>0):
            self.color=(255,0,0)
        else:
            self.color=(0,0,255)
        self.q_radius=max(10,abs(self.q)/self.max_q*500)
        print(self.q_radius)
        
    def get_collision(self,pos):
        if self.rectangle.collidepoint(pos):
           return self
        return None            
    
    def draw(self):
        screen_position=to_screen(self.r.x,self.r.y,self.screen_size,(box_x,box_y))
        self.rectangle=pygame.Rect(screen_position[0]-self.q_radius,screen_position[1]-self.q_radius,2*self.q_radius,2*self.q_radius)              
        pygame.draw.circle(self.screen, self.color, screen_position, self.q_radius)
        
    

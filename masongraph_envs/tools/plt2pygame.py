import pygame
import numpy as np
from hexsimulator.blocks.blocks_norot import Grid,DiscreteBlockNorot as Block, side2corner, outer_sides,get_cm
from hexsimulator import graphics as gr
import matplotlib.pyplot as plt
ground_color = (41, 0, 12)
block_color = (88, 14, 56)
robots_color = [(139, 95, 191),
                (109, 163, 77)]

def open_window(gridsize,scale):
    size = ((gridsize[0]+gridsize[1])*scale,(gridsize[1]+1)*scale*np.sqrt(3)/2)
    pygame.init()
    pygame.display.init()
    window = pygame.display.set_mode(size)
    canvas = pygame.Surface(size)
    factor = (scale,(gridsize[1]/2,np.sqrt(3)/2))
    return window,canvas
def new_caneva(gridsize,scale):
    return pygame.Surface(((gridsize[0]+gridsize[1])*scale,(gridsize[1]+1)*scale*np.sqrt(3)/2))
def fig2surface(fig):
    fig.canvas.draw()
    #image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image =  pygame.image.fromstring(fig.canvas.tostring_rgb())
    #image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)
    return image
def draw_struct(surface,grid,scale,linewidth=0.1):
    shift = (grid.shape[1]/2,np.sqrt(3)/2)
    for gid in range(grid.nreg):
        parts = np.array(np.nonzero((grid.occ==0)&(grid.connection==gid))).T
        sides = outer_sides(parts,ordered=True)
        corners = (side2corner(sides)+shift)*scale
        corners[:,1]=surface.get_size()[1]-corners[:,1]
        pygame.draw.polygon(surface,pygame.Color(ground_color),corners)
        pygame.draw.polygon(surface,pygame.Color(255,255,255),corners,
                                                 width =int(linewidth*scale))
        for c in corners:
             pygame.draw.circle(surface,pygame.Color(255,255,255),center=c,
                                                 radius =int(linewidth*scale)//2)
    for bid in range(1,np.max(grid.occ)+1):
        parts = np.array(np.nonzero(grid.occ==bid)).T
        sides = outer_sides(parts,ordered=True)
        corners = (side2corner(sides)+shift)*scale
        corners[:,1]=surface.get_size()[1]-corners[:,1]
        pygame.draw.polygon(surface,pygame.Color(block_color),corners,
                                                 width =0)
        pygame.draw.polygon(surface,pygame.Color(255,255,255),corners,
                                                 width =int(linewidth*scale))
        for c in corners:
             pygame.draw.circle(surface,pygame.Color(255,255,255),center=c,
                                                 radius =int(linewidth*scale)//2)
    for rid in np.unique(grid.hold):
        if rid==-1:
            continue
        parts = np.array(np.nonzero(grid.hold==rid)).T
        center = (get_cm(parts)+shift)*scale
        center[1]=surface.get_size()[1]-center[1]
        pygame.draw.circle(surface,pygame.Color(robots_color[rid]),center=center,radius=scale/3)

if __name__ =='__main__':
    grid = Grid([10,10])
    
    t = Block([[0,0,1],[0,0,0]])
    #ground = Block([[0,0,0],[2,0,0],[6,0,0],[8,0,0]]+[[i,0,1] for i in range(0,maxs[0])])
    ground = Block([[0,0,1]])
    #ground = Block([[0,0,0],[maxs[0]-1,0,0]]+[[i,0,1] for i in range(0,maxs[0])],muc=0.7)
    hinge = Block([[1,0,0],[0,1,1],[1,1,1],[1,1,0],[0,2,1],[0,1,0]])
    
    linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]])
    linkl = Block([[1,0,1],[1,0,0],[0,1,1],[1,1,0],[1,1,1],[2,0,0]])
    linkh = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[1,2,1],[2,0,0]])
    _,_,i,_=grid.put(ground,[1,0],0,floating=True)
    grid.put(ground,[7,0],0,floating=True)
    grid.put(hinge,[1,0],1,holder_rid=1)
    grid.put(linkr,[0,2],2)
    grid.put(hinge,[1,3],3)
    grid.put(linkh,[2,3],4)
    grid.put(hinge,[4,3],5,holder_rid=0)
    grid.put(linkl,[5,2],6)
    grid.put(hinge,[7,0],7)
    fig,ax = gr.draw_grid([10,10])
    gr.fill_grid(ax,grid)
    w,c,f = open_window([10,10],scale=100)
    draw_struct(c,grid,f)
    w.blit(c, c.get_rect())
    pygame.event.pump()
    pygame.display.update()
    print("End test")
    pass
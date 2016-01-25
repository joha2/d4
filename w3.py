import pygame
import sys
import numpy as np
from math import radians, cos, sin, sqrt



def main():

    blocksize = 100.0
    sizexblocks = 20
    sizeyblocks = 20

    mapsizex = (sizexblocks-1)*blocksize
    mapsizey = (sizeyblocks-1)*blocksize

    wallhight = 1024.0


    fov = 45


    mapbitcode = np.random.random_integers(0, 1, (sizexblocks, sizeyblocks))


    def display_map(surf, bitcode, blsize):
        for indx, row in enumerate(list(bitcode)):
            for indy, col in enumerate(row):
                if col == 1:
                    fillcolor = (255, 0, 0)
                    pygame.draw.rect(surf, fillcolor, (indx*blsize, indy*blsize, blsize, blsize), 0)

    def display_player_onmap(surf, x, y, angledeg, blsize):
        posinmap = (int(x/blocksize*blsize), int(y/blocksize*blsize))
        angl = radians(angledeg)
        posinmapdir = (int((x + blocksize*cos(angl))/blocksize*blsize), int((y + blocksize*sin(angl))/blocksize*blsize))

        pygame.draw.line(surf, np.random.random_integers(0, 255, (3,)), posinmap, posinmapdir)

    def pointtoindex((xp, yp)):
        indexpair = [int(xp/blocksize), int(yp/blocksize)]
        if indexpair[0] > sizexblocks-1:
            indexpair[0] -= sizexblocks
        if indexpair[0] < 0:
            indexpair[0] += sizexblocks
        if indexpair[1] > sizeyblocks-1:
            indexpair[1] -= sizeyblocks
        if indexpair[1] < 0:
            indexpair[1] += sizeyblocks
        return tuple(indexpair)


    def indextopoint((xi, yi)):
        return (xi*blocksize, yi*blocksize)

    def rasterscreen(surf, x, y, angledeg):
        angleslist = [radians(ang) for ang in np.linspace(angledeg - fov, angledeg + fov, display[0], True)]
        scalinglist = [radians(ang) for ang in np.linspace(- fov, + fov, display[0], True)]

        for ind, a in enumerate(angleslist):
            scale_angle = scalinglist[ind]
            blockhit = False
            xend = x #+ blocksize*cos(a) # starte einen block weiter
            yend = y #+ blocksize*sin(a)
            while not blockhit:
                xold = xend
                yold = yend
                xend += blocksize*cos(a)
                yend += blocksize*sin(a)

                mapindexpair = list(pointtoindex((xend, yend)))

                if mapbitcode[tuple(mapindexpair)] == 1:
                    # eine blockgroesse zurueck und dann schnittpunkt zwischen den zwei bloecken bestimmen
                    list(mapindexpair) # eckpunkt aktueller block
                    -(cos(a), sin(a)) # rueckwaertige richtung strahl
                    blockhit = True



            distance = sqrt((xend - x)**2 + (yend - y)**2)
            wallh = wallhight*blocksize/((blocksize + distance)*cos(scale_angle))

            pygame.draw.line(surf, (120, 120, 120), (ind, display[1]/2 + wallh/2), (ind, display[1]/2 - wallh/2))


    xp = 0
    yp = 0
    angle = 0
    hitblock = True

    while hitblock:
        xp = mapsizex*np.random.random()
        yp = mapsizey*np.random.random()
        if mapbitcode[pointtoindex((xp, yp))] == 0:
            hitblock = False




    pygame.init()
    display = (800,600)
    screen = pygame.display.set_mode(display, pygame.DOUBLEBUF)




    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN: # one time key events
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        keys_pressed = pygame.key.get_pressed() # real time key events (keys may be pressed continously)

        speed = 2.0

        if keys_pressed[pygame.K_UP]:
            xp += speed*cos(radians(angle))
            yp += speed*sin(radians(angle))
        if keys_pressed[pygame.K_DOWN]:
            xp -= speed*cos(radians(angle))
            yp -= speed*sin(radians(angle))

        if keys_pressed[pygame.K_LEFT]:
            angle -= 1
        if keys_pressed[pygame.K_RIGHT]:
            angle += 1




        #screen.set_at(tuple(np.random.random_integers(0, high=480, size=(2,))),
        #              tuple(np.random.random_integers(0, high=255, size=(3,))))

        screen.fill((0,0,0))

        rasterscreen(screen, xp, yp, angle)

        display_map(screen, mapbitcode, 10)
        display_player_onmap(screen, xp, yp, angle, 10)

        pygame.display.flip()
        #pygame.time.wait(10)


if __name__=="__main__":
    main()


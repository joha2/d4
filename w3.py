import pygame
import sys
import numpy as np
from math import radians, cos, sin, sqrt



def main():

    blocksize = 100.0
    sizexblocks = 20
    sizeyblocks = 20
    sizezblocks = 20

    mapsizex = (sizexblocks-1)*blocksize
    mapsizey = (sizeyblocks-1)*blocksize
    mapsizez = (sizezblocks-1)*blocksize

    wallhight = 100.0


    fov = 45


    mapbitcode = np.random.random_integers(0, 1, (sizexblocks, sizeyblocks, sizezblocks))


    def rotvec(vec, ay, ax):
        return np.dot(np.array([[cos(ay), sin(ax)*sin(ay), cos(ax)*sin(ay)],
                         [0, cos(ax), -sin(ax)], [-sin(ay), cos(ay)*sin(ax), cos(ax)*cos(ay)]]), vec)

    def cubepoints(tindex):
        if mapbitcode[tindex] == 1:
            return [0]
        else:
            return []
        return []

    print(cubepoints((0, 0, 0)))

    xp = 0
    yp = 0
    zp = 0
    anglexy = 0
    anglez = 0

    ey = np.array([0, 1, 0])


    pygame.init()
    display = (1024, 768)
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
            ey = rotvec(ey, 0.0, 0.05)
        if keys_pressed[pygame.K_DOWN]:
            ey = rotvec(ey, 0.0, -0.05)

        if keys_pressed[pygame.K_LEFT]:
            ey = rotvec(ey, -0.05, 0.0)
        if keys_pressed[pygame.K_RIGHT]:
            ey = rotvec(ey, 0.05, 0.0)


        screen.set_at(tuple(np.random.random_integers(0, high=display[0], size=(2,))),
                      tuple(np.random.random_integers(0, high=255, size=(3,))))

        #screen.fill((0,0,0))

        #rasterscreen(screen, xp, yp, angle)

        #display_map(screen, mapbitcode, 10)
        #display_player_onmap(screen, xp, yp, angle, 10)

        pygame.display.flip()
        pygame.time.wait(10)


if __name__=="__main__":
    main()


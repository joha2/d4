import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import sys, math, pygame
from operator import itemgetter
import numpy as np
from scipy.spatial import ConvexHull


class Point4D:
    def __init__(self, x = 0, y = 0, z = 0, w = 0):
        self.x, self.y, self.z, self.w = float(x), float(y), float(z), float(w)

    def setnp(self, nppt):
        self.x, self.y, self.z, self.w = nppt[0], nppt[1], nppt[2], nppt[3]


    def translate(self, tx, ty, tz, tw):
        x = self.x + tx
        y = self.y + ty
        z = self.z + tz
        w = self.w + tw
        return Point4D(x, y, z, w)

    def rotateXY(self, angle):
        rad = angle * math.pi / 180
        cosa = math.cos(rad);
        sina = math.sin(rad);
        x = cosa*self.x - sina*self.y
        y = sina*self.x + cosa*self.y
        return Point4D(x, y, self.z, self.w)


    def rotateYZ(self, angle):
        rad = angle * math.pi / 180
        cosa = math.cos(rad);
        sina = math.sin(rad);
        y = cosa*self.y - sina*self.z
        z = sina*self.y + cosa*self.z
        return Point4D(self.x, y, z, self.w)


    def rotateZX(self, angle):
        rad = angle * math.pi / 180
        cosa = math.cos(rad);
        sina = math.sin(rad);
        z = cosa*self.z - sina*self.x
        x = sina*self.z + cosa*self.x
        return Point4D(x, self.y, z, self.w)

    def rotateXW(self, angle):
        rad = angle * math.pi / 180
        cosa = math.cos(rad);
        sina = math.sin(rad);
        x = cosa*self.x + sina*self.w
        w = -sina*self.x + cosa*self.w
        return Point4D(x, self.y, self.z, w)

    def rotateYW(self, angle):
        rad = angle * math.pi / 180
        cosa = math.cos(rad);
        sina = math.sin(rad);
        y = cosa*self.y + sina*self.w
        w = -sina*self.y + cosa*self.w
        return Point4D(self.x, y, self.z, w)

    def rotateZW(self, angle):
        rad = angle * math.pi / 180
        cosa = math.cos(rad);
        sina = math.sin(rad);
        z = cosa*self.z + sina*self.w
        w = -sina*self.z + cosa*self.w
        return Point4D(self.x, self.y, z, w)

    def project(self):
        """ Transforms this 4D point to 3D using a cutoff in w. """
        return Point3D(self.x, self.y, self.z)



class Point3D:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def rotateX(self, angle):
        """ Rotates the point around the X axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        y = self.y * cosa - self.z * sina
        z = self.y * sina + self.z * cosa
        return Point3D(self.x, y, z)

    def rotateY(self, angle):
        """ Rotates the point around the Y axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        z = self.z * cosa - self.x * sina
        x = self.z * sina + self.x * cosa
        return Point3D(x, self.y, z)

    def rotateZ(self, angle):
        """ Rotates the point around the Z axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        x = self.x * cosa - self.y * sina
        y = self.x * sina + self.y * cosa
        return Point3D(x, y, self.z)

    def project(self, win_width, win_height, fov, viewer_distance):
        """ Transforms this 3D point to 2D using a perspective projection. """
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + win_width / 2
        y = -self.y * factor + win_height / 2
        return Point3D(x, y, self.z)



verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
    )

colors = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (0,1,0),
    (1,1,1),
    (0,1,1),
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,0,0),
    (1,1,1),
    (0,1,1),
    )



verticies4d = (
    (1, -1, -1, -1),
    (1, 1, -1, -1),
    (-1, 1, -1, -1),
    (-1, -1, -1, -1),
    (1, -1, 1, -1),
    (1, 1, 1, -1),
    (-1, -1, 1, -1),
    (-1, 1, 1, -1),
    (1, -1, -1, 1),
    (1, 1, -1, 1),
    (-1, 1, -1, 1),
    (-1, -1, -1, 1),
    (1, -1, 1, 1),
    (1, 1, 1, 1),
    (-1, -1, 1, 1),
    (-1, 1, 1, 1)
    )

edges4d = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7),
    (8,9), # 01 OK
    (8,11), # 03 OK
    (8,12), # 04 OK
    (10,9), # 21 OK
    (10,11), # 23 OK
    (10,15), # 27 OK
    (14,11), # 63 OK
    (14,12), # 64 OK
    (14,15), # 67 OK
    (13,9), # 51 OK
    (13,12), # 54 OK
    (13,15), # 57 OK
    (0, 8),
    (1, 9),
    (2, 10),
    (3, 11),
    (4, 12),
    (5, 13),
    (6, 14),
    (7, 15)
    )

def drawText(position, textString):
    font = pygame.font.Font (None, 64)
    textSurface = font.render(textString, True, (255,255,255,255), (0,0,0,255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)

    glWindowPos3f(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def Cube4d(screen, tx, ty, tz, tw, axy, ayz, azx, axw, ayw, azw, camposition, camangleXZ, camangleYZ, camangleZW, col = (1, 1, 0)):

    com4d = np.array((tx, ty, tz, tw))

    #print com4d[3]

    transformverticies4d = []
    transformverticies4dproj3d = []


    ostring = ''
    todrawv = []
    for e in edges4d:
        p1 = Point4D()
        p2 = Point4D()
        p1.setnp(np.array(verticies4d[e[0]]))
        p2.setnp(np.array(verticies4d[e[1]]))

        p1r = p1.rotateXY(axy).rotateYZ(ayz).rotateZX(azx).rotateXW(axw).rotateYW(ayw).rotateZW(azw)
        p1r = p1r.translate(com4d[0] - camposition[0], com4d[1] - camposition[1], com4d[2] - camposition[2], com4d[3] - camposition[3])
        p1r = p1r.rotateZX(camangleXZ).rotateYZ(camangleYZ).rotateZW(camangleZW)

        # TODO: ersetze letzte rotation durch cameravektor der sich drehen laesst und der wieder auf (0,0,1,0) gedreht wird
        # camvector wird gemaesz rotationsmatrizen gedreht, aber nicht die punkte im raum sonst hat man bewegungsprobleme an den polen
        # der kugelkoordinaten, d.h. aus dem camvector werden die richtungscosinusse berechnet und dann die punkte entsprechend dieser hingedreht


        p2r = p2.rotateXY(axy).rotateYZ(ayz).rotateZX(azx).rotateXW(axw).rotateYW(ayw).rotateZW(azw)
        p2r = p2r.translate(com4d[0] - camposition[0], com4d[1] - camposition[1], com4d[2] - camposition[2], com4d[3] - camposition[3])
        p2r = p2r.rotateZX(camangleXZ).rotateYZ(camangleYZ).rotateZW(camangleZW)

        if p1r.w != p2r.w:
            lp = -p1r.w/(p2r.w-p1r.w) # parameter has to be 0 <= lp <= 1
            if lp >= 0 and lp <= 1:
                ostring += '* '
                todrawv.append((p1r.x + lp*(p2r.x - p1r.x), p1r.y + lp*(p2r.y - p1r.y), p1r.z + lp*(p2r.z - p1r.z), p1r.w + lp*(p2r.w - p1r.w)))
        else:
            if abs(p1r.w) <= 1e-16:
                ostring += '*'
                todrawv.append((p1r.x, p1r.y, p1r.z, p1r.w))
            if abs(p2r.w) <= 1e-16:
                ostring += '*'
                todrawv.append((p2r.x, p2r.y, p2r.z, p2r.w))
            ostring += ' '

    #print ostring

    for i in todrawv:
        v4d = np.array(i)
        transformv4d = v4d
#        p4d = Point4D(v4d[0], v4d[1], v4d[2], v4d[3])
#        p4d = p4d.rotateXY(axy).rotateYZ(ayz).rotateZX(azx).rotateXW(axw).rotateYW(ayw).rotateZW(azw)
#        rotatedcom = np.array((p4d.x,p4d.y,p4d.z,p4d.w))
#        transformv4d = rotatedcom + com4d
        transformv4dcam = Point4D(transformv4d[0], transformv4d[1], transformv4d[2], transformv4d[3])
#        transformv4dcam = transformv4dcam.translate(-camposition[0], -camposition[1], -camposition[2], -camposition[3])
#        transformv4dcam = transformv4dcam.rotateZX(camangleXZ).rotateYZ(camangleYZ).rotateZW(camangleZW)
        transformcamv4d = np.array((transformv4dcam.x, transformv4dcam.y, transformv4dcam.z, transformv4dcam.w))#-camposition

        transformverticies4d.append(transformcamv4d)

        projtransformv = (transformcamv4d[0], transformcamv4d[1], transformcamv4d[2])
        transformverticies4dproj3d.append(projtransformv)

    if len(todrawv) > 3: # needs at least 4 points
        hull3d = ConvexHull(transformverticies4dproj3d)

        glBegin(GL_TRIANGLES)
        #trianglecolors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        glMaterialfv(GL_FRONT,GL_AMBIENT,(0.1,0.1,0.1,1)) # no light emission
        glMaterialfv(GL_FRONT,GL_SPECULAR,(0.1,0.1,0.1,1))
        trianglecolors = (col, col, col)

        for simplex in hull3d.simplices: # simplicies vorher nach umlaufsinn ordnen
            x = 0

            v1 = np.array(transformverticies4dproj3d[simplex[1]]) - np.array(transformverticies4dproj3d[simplex[0]])
            v2 = np.array(transformverticies4dproj3d[simplex[2]]) - np.array(transformverticies4dproj3d[simplex[0]])

            nv = np.cross(v1, v2)
            nv = nv/np.linalg.norm(nv)

            if np.dot(np.array(transformverticies4dproj3d[simplex[0]]) - com4d[0:3], nv) < 0:
                nv = -nv

#             simplexgvector = np.array((0,0,0))
#             for vnum in simplex: # schwerpunktberechnung
#                 simplexgvector += transformverticies4dproj3d[vnum]
#             simplexgvector = simplexgvector/3.0
#
#             vdifferences = [] # winkelberechnung
#             for vnum in simplex:
#                 vdiff = transformverticies4dproj3d[vnum] - simplexgvector
#                 vdiff = vdiff/np.linalg.norm(vdiff)
#                 vdifferences.append([vdiff, vnum])
#
#             vangles = [] # winkelberechnung
#             for vn in vdifferences:
#                 vangles.append([np.dot(vdifferences[0][0], vn[0]), vn[1]])
#
#             sortedsimplex = []
#             for tmp in sorted(vangles,key=itemgetter(1),reverse=False):
#                 sortedsimplex.append(tmp[1])
            sortedsimplex = simplex

            glNormal3dv(nv)

            for vnum in sortedsimplex:
                glColor3fv(trianglecolors[x])
                x+=1
                glVertex3fv(transformverticies4dproj3d[vnum])
        glEnd()



#     glBegin(GL_LINES)
#     for edge in edges:
#         for vertex in edge:
#             glColor3fv((0,0,1))
#             glVertex3fv(transformverticies4dproj3d[vertex])
#     glEnd()



def Cube():

    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x+=1
            glColor3fv(colors[x])
            glVertex3fv(verticies[vertex])
    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv((0,0,1))
            glVertex3fv(verticies[vertex])
    glEnd()



def main():
    pygame.init()
    display = (800,600)
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(90, (display[0]/display[1]), 0.1, 50.0)


    #glTranslatef(0.0,0.0, -20.0) # -5


    glEnable( GL_DEPTH_TEST )
    #glEnable( GL_CULL_FACE )

    #// enable color tracking
    glEnable(GL_COLOR_MATERIAL)
    #// set material properties which will be assigned by glColor
    glColorMaterial(GL_FRONT, GL_AMBIENT)
    glColorMaterial(GL_FRONT, GL_DIFFUSE)
    #glColorMaterial(GL_FRONT, GL_SPECULAR)


    #// Create light components

    ambientLight = ( 1.0, 1.0, 1.0, 1.0 )
    diffuseLight = ( 0.8, 0.8, 0.8, 1.0 )
    specularLight = ( 0.5, 0.5, 0.5, 1.0 )


#    ambientLight = ( 0.2, 0.2, 0.2, 1.0 )
#    diffuseLight = ( 0.8, 0.8, 0.8, 1.0 )
#    specularLight = ( 0.5, 0.5, 0.5, 1.0 )
#     position = ( -1.5, 1.0, 10.0, 1.0 )

#     ambientLight = ( 0.0, 0.0, 0.0, 1.0 )
#     diffuseLight = ( 1.0, 1.0, 1.0, 1.0 )
#     specularLight = ( 0.5, 0.5, 0.5, 1.0 )

    lx = -1.5
    ly = 1.0
    lz = -5

    #// Assign created components to GL_LIGHT0


    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)


    angle = 0
    angleXY = 0
    angleYZ = 0
    angleZX = 0
    angleXW = 0
    angleYW = 0
    angleZW = 0

    xyzangle = 0

    camangleYZ = 0
    camangleXZ = 0
    camangleZW = 0

    rotatecube = False
    printlines = False
    lighton = False
    rotatelight = True

    cx = 0
    cy = 0
    cz = 0
    cw = 0

    camx = 0
    camy = 0
    camz = 10
    camw = 0


    cubepoints = (10.0*(np.ones((20*4)) - 2.0*np.random.random(20*4))).reshape((20, 4))

    camdirection = Point4D(0, 0, -1, 0)


    while True:

        angle += 1.0
        xyzangle += 2.0*math.pi/180.0


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN: # one time key events
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_k:
                    rotatecube = 1 - rotatecube
                if event.key == pygame.K_l:
                    printlines = 1 - printlines
                if event.key == pygame.K_o:
                    rotatelight = 1 - rotatelight
                if event.key == pygame.K_p:
                    lighton = 1 - lighton

        keys_pressed = pygame.key.get_pressed() # real time key events (keys may be pressed continously)

#######################################################
# old controls for one single cube
#######################################################

#         if keys_pressed[pygame.K_q]:
#             cx += 0.1
#         if keys_pressed[pygame.K_w]:
#             cx -= 0.1
#         if keys_pressed[pygame.K_e]:
#             cy += 0.1
#         if keys_pressed[pygame.K_r]:
#             cy -= 0.1
#         if keys_pressed[pygame.K_t]:
#             cz += 0.1
#         if keys_pressed[pygame.K_y]:
#             cz -= 0.1
#         if keys_pressed[pygame.K_u]:
#             cw += 0.1
#         if keys_pressed[pygame.K_i]:
#             cw -= 0.1
#
#
#         if keys_pressed[pygame.K_a]:
#             angleXY += 1.0
#         if keys_pressed[pygame.K_z]:
#             angleXY -= 1.0
#         if keys_pressed[pygame.K_s]:
#             angleYZ += 1.0
#         if keys_pressed[pygame.K_x]:
#             angleYZ -= 1.0
#         if keys_pressed[pygame.K_d]:
#             angleZX += 1.0
#         if keys_pressed[pygame.K_c]:
#             angleZX -= 1.0
#
#         if keys_pressed[pygame.K_f]:
#             angleXW += 1.0
#         if keys_pressed[pygame.K_v]:
#             angleXW -= 1.0
#         if keys_pressed[pygame.K_g]:
#             angleYW += 1.0
#         if keys_pressed[pygame.K_b]:
#             angleYW -= 1.0
#         if keys_pressed[pygame.K_h]:
#             angleZW += 1.0
#         if keys_pressed[pygame.K_n]:
#             angleZW -= 1.0

#         if keys_pressed[pygame.K_UP]:
#             if camangleYZ < 360:
#                 camangleYZ += 5.0
#             else:
#                 camangleYZ = camangleYZ - 360
#         if keys_pressed[pygame.K_DOWN]:
#             if camangleYZ > 0:
#                 camangleYZ -= 5.0
#             else:
#                 camangleYZ = 360 + camangleYZ
#
#         if keys_pressed[pygame.K_PAGEUP]:
#             #if camangleZW < 90:
#             camangleZW += 0.1
#             #else:
#             #    camangleZW = 90
#         if keys_pressed[pygame.K_PAGEDOWN]:
#             #if camangleZW > -90:
#             camangleZW -= 0.1
#             #else:
#             #    camangleZW = -90
#
#
#         if keys_pressed[pygame.K_LEFT]:
#             camangleXZ -= 5.0
#         if keys_pressed[pygame.K_RIGHT]:
#             camangleXZ += 5.0


#######################################################
# new controls for descent like behaviour
#######################################################


        if keys_pressed[pygame.K_UP]:
            camdirection = camdirection.rotateYZ(5.0)
            if camangleYZ < 360:
                camangleYZ += 5.0
            else:
                camangleYZ = camangleYZ - 360

        if keys_pressed[pygame.K_DOWN]:
            camdirection = camdirection.rotateYZ(-5.0)
            if camangleYZ > 0:
                camangleYZ -= 5.0
            else:
                camangleYZ = 360 + camangleYZ

        if keys_pressed[pygame.K_PAGEUP]:
            #if camangleZW < 90:
            camdirection = camdirection.rotateZW(0.1)

            camangleZW += 0.1
            #else:
            #    camangleZW = 90
        if keys_pressed[pygame.K_PAGEDOWN]:
            #if camangleZW > -90:
            camdirection = camdirection.rotateZW(-0.1)

            camangleZW -= 0.1
            #else:
            #    camangleZW = -90


        if keys_pressed[pygame.K_LEFT]:
            camdirection = camdirection.rotateZX(-5.0)
            camangleXZ -= 5.0
        if keys_pressed[pygame.K_RIGHT]:
            camdirection = camdirection.rotateZX(5.0)
            camangleXZ += 5.0



        speed = 0.1
        if keys_pressed[pygame.K_z]:
            camx += -speed*camdirection.x#math.cos(camangleXZ*math.pi/180.0 - math.pi*0.5)*math.cos(-camangleYZ*math.pi/180.0)*math.cos(camangleZW*math.pi/180.0)
            camz += -speed*camdirection.z#math.sin(camangleXZ*math.pi/180.0 - math.pi*0.5)*math.cos(-camangleYZ*math.pi/180.0)*math.cos(camangleZW*math.pi/180.0)
            camy += -speed*camdirection.y#math.sin(-camangleYZ*math.pi/180.0)
            camw += -speed*camdirection.w#math.sin(-camangleYZ*math.pi/180.0)*math.sin(camangleZW*math.pi/180.0)

        if keys_pressed[pygame.K_a]:
            camx += speed*camdirection.x#math.cos(camangleXZ*math.pi/180.0 - math.pi*0.5)*math.cos(-camangleYZ*math.pi/180.0)*math.cos(camangleZW*math.pi/180.0)
            camz += speed*camdirection.z#math.sin(camangleXZ*math.pi/180.0 - math.pi*0.5)*math.cos(-camangleYZ*math.pi/180.0)*math.cos(camangleZW*math.pi/180.0)
            camy += speed*camdirection.y#math.sin(-camangleYZ*math.pi/180.0)
            camw += speed*camdirection.w#math.sin(-camangleYZ*math.pi/180.0)*math.sin(camangleZW*math.pi/180.0)

        if rotatecube:
            angleXY += 1
            angleYZ += 1
            angleZX += 1
            angleXW += 1
            angleYW += 1
            angleZW += 1

        #print camx, " ", camy, " ", camz, " ", cx, " ", cy, " ", cz, " ", camangleXZ, " ", camangleYZ


        #glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        #glClearColor(math.exp(-1.0/(0.001+abs(camw))), 0.0, 0.0, 1.0)


        drawText((0,0,0), "xy,yz,zw: %.2f %.2f %.2f" % (camangleXZ, camangleYZ, camangleZW))
        drawText((0,50,0), "cam: %.2f %.2f %.2f %.2f" % (camdirection.x, camdirection.y, camdirection.z, camdirection.w))
        drawText((0,100,0), "xyzw: %.2f %.2f %.2f %.2f" % (camx, camy, camz, camw))

        position = ( lx, ly, lz, 1.0 )

        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight)
        glLightfv(GL_LIGHT0, GL_POSITION, position)

        #glPushMatrix()

        for i in range(4):
            pt = cubepoints[i]
            Cube4d(screen, cx + pt[0], cy + pt[1], cz + pt[2], cw, angleXY, angleYZ, angleZX, angleXW, angleYW, angleZW, np.array((camx,camy,camz,camw)), camangleXZ, camangleYZ, camangleZW, (1,0,0))
        #Cube4d(screen, cx+2,cy+2, cz, cw, angleXY, angleYZ, angleZX, angleXW, angleYW, angleZW, np.array((camx,camy,camz,camw)), camangleXZ, camangleYZ, 0.0, (1, 0, 0))
        #Cube4d(screen, cx+2,cy, cz, cw, angleXY, angleYZ, angleZX, angleXW, angleYW, angleZW, np.array((camx,camy,camz,camw)), camangleXZ, camangleYZ, 0.0, (1,0,1))
        #Cube4d(screen, cx,cy+2, cz, cw, angleXY, angleYZ, angleZX, angleXW, angleYW, angleZW, np.array((camx,camy,camz,camw)), camangleXZ, camangleYZ, 0.0, (1, 0, 0))




        pygame.display.flip()
        pygame.time.wait(10)


main()

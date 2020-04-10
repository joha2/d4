# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:19:26 2015

@author: Johannes Hartung
"""

import numpy as np
import math
import pygame
import sys

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

# the n-d routines are implemented based on the stuff published at https://ef.gy/
# and explained by Magnus Deininger in his ccc talk


def calc_basis(dim, n):
    res = np.zeros(dim)
    res[n] = 1.0
    return res

def calc_normals(dim, *args):
    res = np.zeros(dim)
    if len(args) == dim - 1:
        # calculate normal of dim - 1 vectors
        tupleofargs = tuple(v.reshape(dim, 1) for v in args) # for later concatenation
        coefficientmatrix = np.concatenate(tupleofargs, axis=1)
        mone = 1.0

        for i in range(dim):
            res += mone*np.linalg.det(np.delete(coefficientmatrix, (i), axis=0))*calc_basis(dim, i)
            mone *= -1.0

    return res

def calc_lookat(dim, to, frm, *args):
    res = np.zeros((dim, dim))
    if len(args) == dim - 2:
        matrixcolumns = []
        matrixcolumns.append(to-frm)

        for i in range(dim-1):
            necessarybasisvectors = [calc_basis(dim, j) for j in range(i, dim-2)]
            #print("basis: ", necessarybasisvectors)
            #print(matrixcolumns)
            newcol = calc_normals(dim, *(necessarybasisvectors + matrixcolumns))
            #print(newcol)
            matrixcolumns = [newcol] + matrixcolumns
            #print(matrixcolumns)

        matrixcolumns = tuple(m.reshape(dim, 1) for m in matrixcolumns)

        res = np.concatenate(matrixcolumns, axis=1)
        #print(res)

    return res

def calc_perspective(dim, eyeangle):
    matdiag = np.concatenate((1.0/math.tan(eyeangle/2.)*np.ones(dim-1), np.array([1,1])))
    return np.diag(matdiag)

def gen_homogen_matrix(dim, matrix):
    temp = np.concatenate((matrix, np.zeros((dim, 1))), axis=1)
    vec = calc_basis(dim+1, dim).reshape((1, dim+1))
    res = np.concatenate((temp, vec))
    return res

def gen_translation_matrix(dim, trans):
    res =  np.diag(np.ones(dim+1))
    res[:dim,dim] = trans
    return res

def gen_viewmatrix_nd(dim, to, frm, eyeangle, *args):
    res = np.dot(gen_translation_matrix(dim, -frm), np.dot(gen_homogen_matrix(dim, calc_lookat(dim, to, frm, *args)),calc_perspective(dim, eyeangle)))
    return res


def gen_viewmatrix_proj(dim, to, frm, *args):
    res = np.dot(gen_translation_matrix(dim, -frm), gen_homogen_matrix(dim, calc_lookat(dim, to, frm, *args)))
    return res



def gen_rotation_matrix(dim, angle, u, v):
    mat = np.diag(np.ones(dim))
    mat += math.sin(angle)*(np.kron(v, u) - np.kron(u, v)).reshape((dim, dim))
    mat += (math.cos(angle) - 1.0)*(np.kron(u, u) + np.kron(v, v)).reshape((dim, dim))
    return gen_homogen_matrix(dim, mat)


class Camera(object):
    """
    Camera class. Has one center of mass which can be translated.
    And has different rotation angles which are put into a matrix.
    All together with some mirror matrix this constitutes the
    camera transform. Some projection projects the points in 4D
    into screen coordinates by removing the 4th component and
    projecting onto screen coordinates.
    """
    pass

class PointCloud(object):
    """
    Class for managing nD point transformations.
    Geometric calculations are also part of this class.
    Projection into screen coordinates should not be missing.
    Calculations are done via (n+1)D homogenious coordinates.
    All changes like translation and rotation are incremental.
    """
    def __init__(self):
        self.edges = []

    def setPoints(self, points):
        self.points = points #np.vstack((points.T, np.ones(len(points)).T)).T
        

    def addEdge(self, edges = (0,1)):
        if type(edges) == tuple and len(edges) == 2:
            self.edges.append(edges)
        elif type(edges) == list:
            self.edges.extend(edges)
            
    def getEdges(self):
        return self.edges

    def getEdgeProperties(self, edge):
        """
        Returns direction vector, startpoint
        and endpoint of an edge.

        :param edge (tuple)
        :returns (direction, startpoint, endpoint)
        """
        
        startpoint = self.points[edge[0]]
        endpoint = self.points[edge[1]]

        return (endpoint - startpoint, startpoint, endpoint)
    
    def getEdgeIntersectionAfterProj(self, dim, edge, proj):

        eps = 1e-10
        
        projstart = np.dot(self.points[edge[0]], proj)
        projend = np.dot(self.points[edge[1]], proj)
        
        print(projstart)
        print(projend)

        diffproj = projend[dim-1] - projstart[dim-1]
        a = projstart[dim-1]

        if abs(diffproj) < eps:
            if abs(a) < eps:
                t = 0.0 # t arbitrary and in particular t=0.0 (eigentlich muesste man a und b zurueckgeben)
            else:
                t = None # no solution
        else:
            t = - a/diffproj
            if t < 0 or t > 1:
                t = None # outside of edge

        return t        
        

class Tet(PointCloud):
    def __init__(self, dim, points):
        super(Tet, self).__init__()
        
        shp = points.shape

        print(shp)        


        numpoints = shp[0]
        dimpoints = shp[1]
        
        
        if numpoints != dim+1 or dimpoints != dim:
            pts = np.zeros((dim+1, dim+1))
            for n in range(dim+1):
                pts[n][dim] = 1.0
                pts[n][n] = 1.0
        else:
            pts = points
            print(points)
            print(np.ones((numpoints,1)))
            pts = np.concatenate((pts, np.ones((numpoints,1))), axis=1)

        for n in range(dim+1):
            for m in range(n):
                self.addEdge((m, n))

            
        self.setPoints(pts)
            
        
class PygameApp(object):
    def __init__(self):
        pygame.init()
        self.display = (800,600)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF|OPENGL)

        gluPerspective(90, (self.display[0]/self.display[1]), 0.1, 50.0)


        glTranslatef(0.0,0.0, -20.0) # -5


        glEnable( GL_DEPTH_TEST )
        #glEnable( GL_CULL_FACE )

        #// enable color tracking
        glEnable(GL_COLOR_MATERIAL)
        #// set material properties which will be assigned by glColor
    
        glColorMaterial(GL_FRONT, GL_AMBIENT)
        glColorMaterial(GL_FRONT, GL_DIFFUSE)
        glColorMaterial(GL_FRONT, GL_SPECULAR)


        #// Create light components

        self.ambientLight = ( 1.0, 1.0, 1.0, 1.0 )
        self.diffuseLight = ( 0.8, 0.8, 0.8, 1.0 )
        self.specularLight = ( 0.5, 0.5, 0.5, 1.0 )

        #// Assign created components to GL_LIGHT0


        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
    def drawText(self, position, fcolor, bcolor, textString):
        font = pygame.font.Font (None, 64)
        textSurface = font.render(textString, True, fcolor, bcolor)
        textData = pygame.image.tostring(textSurface, "RGBA", True)

        glWindowPos3f(*position)
        glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)
        

    def run(self):
        
        counter = 0        
        
        while True:

            counter += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN: # one time key events
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_k:
                        pass
                    if event.key == pygame.K_l:
                        pass
                    if event.key == pygame.K_o:
                        pass
                    if event.key == pygame.K_p:
                        pass

            keys_pressed = pygame.key.get_pressed() 

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)


            glLightfv(GL_LIGHT0, GL_AMBIENT, self.ambientLight)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, self.diffuseLight)
            glLightfv(GL_LIGHT0, GL_SPECULAR, self.specularLight)
            glLightfv(GL_LIGHT0, GL_POSITION, (0., 0. -100.))

            self.drawText((400, 200, -100), (255, 0, 0, 255), (0, 0, 0, 0), str(counter))
            self.drawText((350, 180, 0), (0, 255, 0, 255), (0, 0, 0, 0), str(counter))

            pygame.display.flip()
            pygame.time.wait(10)


    def close(self):        
        pass
        


def main():
    pts = np.array((
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
    ))

    p = PointCloud()
    p.setPoints(pts)
    p.addEdge(edges =
    [
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
    ]
    )


    pm = gen_viewmatrix_proj(4, np.array([0,0,-10,0]), np.array([1/2,1/2,1/2,1/2]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]))

    t = Tet(4, np.array([[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]))
    print(t.points)
    print(t.getEdges())
    print(t.getEdgeProperties(t.edges[9]))
    for a in t.edges:
        print(t.getEdgeIntersectionAfterProj(4, a, pm))
    print("blub")

    pg = PygameApp()
    pg.run()
    pg.close()
    

if __name__ == "__main__":
    main()

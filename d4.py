# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 19:19:26 2015

@author: Johannes Hartung
"""

import numpy as np
import math


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

def gen_viewmatrix(dim, to, frm, eyeangle, *args):
    res = np.dot(gen_translation_matrix(dim, -frm), np.dot(gen_homogen_matrix(3, calc_lookat(dim, to, frm, *args)),calc_perspective(dim, eyeangle)))
    return res

#def gen_rotation_matrix(angle, u, v):
#    res = np.diag([1.,1.,1.,1.,1.])
#    u = np.append(u, [0.])
#    v = np.append(v, [0.])
#    res += math.sin(angle)*(np.kron(v, u) - np.kron(u, v)).reshape((5,5))
#    + (math.cos(angle) - 1.0)*(np.kron(u, u) + np.kron(v, v)).reshape((5,5))
#    return res


class Camera4D(object):
    """
    Camera class. Has one center of mass which can be translated.
    And has different rotation angles which are put into a matrix.
    All together with some mirror matrix this constitutes the
    camera transform. Some projection projects the points in 4D
    into screen coordinates by removing the 4th component and
    projecting onto screen coordinates.
    """
    pass

class PointCloud4D(object):
    """
    Class for managing 4D point transformations.
    Geometric calculations are also part of this class.
    Projection into screen coordinates should not be missing.
    Calculations are done via 5D homogenious coordinates.
    All changes like translation and rotation are incremental.
    """
    def __init__(self, points, com=np.array([0.,0.,0.,0.])):
        self.points = np.vstack((points.T, np.ones(len(points)).T)).T
        self.com = np.append(com, [1.])
        self.edges = []
        self.faces = []

    def addEdge(self, edges = (0,1)):
        if type(edges) == tuple and len(edges) == 2:
            self.edges.append(edges)
        elif type(edges) == list:
            self.edges.extend(edges)

    def getEdgeProperties(self, edgenum=0):
        """
        Returns direction vector, startpoint
        and endpoint of an edge.

        :param edgenum (int)
        :returns (direction, startpoint, endpoint)
        """

        edge = self.edges[edgenum]
        startpoint = self.points[edge[0]]
        endpoint = self.points[edge[1]]

        return (endpoint - startpoint, startpoint, endpoint)



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

    p = PointCloud4D(pts)
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

    #print(p.points)
    #print(p.com)
    #print(p.edges, len(p.edges))
    print(calc_normals(4, np.array([1,0,0,0]), np.array([0,1,0,0]), np.array([0,0,1,0])))

    bla = calc_lookat(3, np.array([1,1,1]), np.array([0,0,0]), np.array([0, -1, 1]))
    print(bla)
    print(gen_homogen_matrix(3, bla))
    print(calc_perspective(4, 45*math.pi/180.0))
    print(gen_viewmatrix(3, np.array([1,1,1]), np.array([0,0,0]), 45*math.pi/180.0, np.array([0, -1, 1])))
    #print(gen_translation_matrix(4, np.array([1,2,3,4])))
    #print(gen_rotation_matrix(0.1, np.array([1,0,0,0]),np.array([0,1,0,0])))


if __name__ == "__main__":
    main()

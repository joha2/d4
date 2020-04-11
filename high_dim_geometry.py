from scipy.spatial import (ConvexHull, Delaunay,
                           delaunay_plot_2d, convex_hull_plot_2d)
import numpy as np
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d as plt3d

import pygame
import pygame.locals as pgl

import OpenGL.GL as ogl
import OpenGL.GLU as oglu


class ConvexHullND:

    def __init__(self, defining_points):
        (self.num_pts, self.dimension) = np.shape(defining_points)
        self.points = defining_points
        self.position = np.zeros(self.dimension)
        self.rotation_matrix = np.eye(self.dimension)
        self.convex_hull = ConvexHull(self.points)
        # print("dimension %d" % (self.dimension,))
        # print("shape of point array")
        # print(np.shape(self.convex_hull.points))
        # print("how much vertices in convex hull?")
        # print(np.shape(self.convex_hull.vertices))
        # print("maximal vertex number?")
        # print(np.max(self.convex_hull.vertices))
        (self.num_convex_hull_pts, _) = np.shape(
            self.convex_hull.points[self.convex_hull.vertices])
        self.generate_edges_from_conv()

    def transform_points(self, points):
        return np.dot(points - self.position, self.rotation_matrix)

    def update_conv_hull(self):
        self.convex_hull = ConvexHull(self.points)
        self.generate_edges_from_conv()

    def generate_edges_from_conv(self):
        all_edges_deep = np.dstack((self.convex_hull.simplices,
                                    np.roll(self.convex_hull.simplices,
                                            1, axis=1)))
        (num_simps, len_simps, len_edge) = np.shape(all_edges_deep)
        all_edges = all_edges_deep.reshape(1, num_simps*len_simps, len_edge)[0]
        all_edges.sort(axis=1)

        # remove double entries
        # notice orientation is lost due to sorting before
        all_edges_set = set([tuple(e) for e in all_edges.tolist()])
        all_edges_removed_doubles = np.array(list(all_edges_set))

        self.edges = all_edges_removed_doubles

    def cut_edges(self, n_vector, x0_point):
        """
        cut all edges of the convex hull body by a plane defined by
        n_vector (x - x0_point) = 0.
        """
        cut_line = np.hstack((n_vector, -np.dot(x0_point, n_vector)))
        # cut_line*(x1, ..., xn, 1) < oder > 0
        pts1 = np.hstack((self.convex_hull.points,
                          np.ones((self.num_pts, 1))))
        cut_line_rep = np.repeat(cut_line[np.newaxis, :],
                                 self.num_pts, axis=0)

        notzero = np.sum(pts1 * cut_line_rep, axis=1)

        diffsigns = np.sign(notzero[self.edges])
        to_be_cut = np.abs(diffsigns[:, 0] - diffsigns[:, 1]) > 0

        two_points_edges = self.convex_hull.points[self.edges[to_be_cut]]

        diff = two_points_edges[:, 1, :] - two_points_edges[:, 0, :]
        startp = two_points_edges[:, 0, :]

        t = np.sum(n_vector*(x0_point - startp), axis=1) / \
            np.sum(n_vector*diff, axis=1)

        cutting_points = diff*t[:, np.newaxis] + startp

        return cutting_points

    def cut_edges_convex_hull(self, n_vector, x0_point, *arg):
        """
        Collects points from cut of ND shape through by plane
        n_vector (x - x0_point) = 0.
        Projects them perpendicular to n_vector and gives components
        in rest of the basis, provided by arg
        """
        (dimension, ) = n_vector.shape
        plane_cut_points = self.cut_edges(n_vector, x0_point)
        projector_perp_n = np.eye(dimension) - np.outer(n_vector, n_vector)

        proj_pts = np.dot(projector_perp_n, plane_cut_points.T)
        remaining_basis = np.array(arg)
        (num_basis_vectors, dimension_arg) = remaining_basis.shape
        if num_basis_vectors != dimension - 1 or dimension_arg != dimension:
            return None
        valueNm1D = np.dot(remaining_basis, proj_pts)
        (num_dimensions, num_points) = valueNm1D.shape
        print(valueNm1D.shape)
        if num_points == 0:
            return None
        return ConvexHull(valueNm1D.T)  # ConvexHull()

    def draw_3d(self):
        """
        Draws only the first three dimensions
        """
        ogl.glBegin(ogl.GL_LINES)
        for edge in self.edges:
            for vertex in edge:
                ogl.glVertex3fv(self.transform_points(
                    self.points[vertex, :])[:3])
        ogl.glEnd()

    def draw_conv(self, n_vector, x0_point, *arg):
        convhull = self.cut_edges_convex_hull(n_vector, x0_point, *arg)

        ogl.glBegin(ogl.GL_LINES)
        for edge in convhull.simplices:
            for vertex in edge:
                ogl.glVertex3fv(self.transform_points(
                                convhull.points[vertex, :])[:3])
        ogl.glEnd()



class SimplexND(ConvexHullND):
    def __init__(self, dimension, points):
        # TODO: consistency check
        super(SimplexND, self).__init__(points)


class SimplexNDStandard:
    def __init__(self, dimension, points):
        if points.shape != (dimension + 1, dimension):
            print("error in points definition")
            print("in ", dimension, " dimensions a simplex needs ",
                  dimension + 1, "points of dimension ", dimension)
            points = None
        self.points = points
        (self.transform_matrix, self.transform_vector) =\
            self.calculate_transform()
        # may rearrange points
        self.transform_matrix_inv = np.linalg.inv(self.transform_matrix)

        self.canonical_points = self.transform_to_canonical(self.points)

    def calculate_transform(self):
        org = self.points[0, :]
        matrix = self.points[1:, :] - org
        if np.linalg.det(matrix) < 0:
            self.points[[1, 2]] = self.points[[2, 1]]
            matrix = self.points[1:, :] - org
        return (matrix, org)

    def transform_to_canonical(self, points):
        v = points - self.transform_vector
        return np.dot(v, self.transform_matrix_inv)

    def transform_to_canonical_dir(self, directions):
        return np.dot(directions,
                      self.transform_matrix_inv)

    def transform_from_canonical(self, canonical_points):
        return np.dot(canonical_points,
                      self.transform_matrix) +\
            self.transform_vector

    def transform_from_canonical_dir(self, directions):
        return np.dot(directions, self.transform_matrix)

    def cut_plane(self, nvector, x0):
        """
        cut simplex with plane n (x - x0) = 0
        """
        nv_can = self.transform_to_canonical_dir(nvector)[0, :]
        nv_can = nv_can/np.linalg.norm(nv_can)
        x0_can = self.transform_to_canonical(x0)[0, :]
        (num_points, _) = self.canonical_points.shape

        # generate edges

        cut_points_can = []
        for i in range(num_points):
            for j in range(i):
                p1 = self.canonical_points[i]
                p2 = self.canonical_points[j]
                diff = p2 - p1
                denom = np.sum(nv_can*diff)
                if np.abs(denom) > 1e-10:
                    t = np.sum(x0_can - p1)/denom
                    print(t)
                    if t >= 0 and t <= 1:
                        cut_points_can.append(p1 + t*diff)
        cut_points_can = np.array(cut_points_can)

        return self.transform_from_canonical(cut_points_can)


class CubeND(ConvexHullND):
    def __init__(self, dimension, width=1):
        # generation via meshgrid
        zero_one = width*np.array([-0.5, 0.5])
        ls_zero_one = [zero_one for i in range(dimension)]
        bits = np.meshgrid(*ls_zero_one)  # spits out tuple
        points_unitcube = np.vstack([b.flatten() for b in bits]).T
        super(CubeND, self).__init__(points_unitcube)


class SphereND(ConvexHullND):
    def __init__(self, dimension, radius=1.,
                 phitype_nr=10, thetatype_nr=5):

        def x(nindex, radius, angles):
            (num_dimsm1, num_pts) = np.shape(angles)
            res = np.ones_like(angles[0, :])
            sinprod = np.ones_like(res)
            if nindex < num_dimsm1:
                res = np.cos(angles[nindex, :])
            if nindex > 0:
                sinprod = np.prod(np.sin(angles[:nindex, :]), axis=0)
            res = radius*res*sinprod
            return res
        # x(1) = r cos(phi1)
        # x(2) = r sin(phi1) cos(phi2)
        # x(3) = r sin(phi1) sin(phi2) cos(phi3)
        # ...
        # x(n-1) = r sin(phi1) .. sin(phi(n-2))*cos(phi(n-1))
        # x(n) = r sin(phi1) ... sin(phi(n-2))*sin(phi(n-1))
        # 0<= phi1...phi(n-2) <= pi; 0 <= phi(n-1) <= 2pi

        thetaangles = np.linspace(0, np.pi, thetatype_nr)
        phiangles = np.linspace(0, 2*np.pi, phitype_nr, endpoint=False)

        lsangles = [thetaangles for i in range(dimension - 2)] + [phiangles]

        angles = np.meshgrid(*lsangles)

        all_angles = np.vstack([a.flatten() for a in angles])

        points_sphere = np.vstack([x(i, radius, all_angles)
                                   for i in range(dimension)]).T
        # print(points_sphere)
        super(SphereND, self).__init__(points_sphere)


def main():

    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, pgl.DOUBLEBUF | pgl.OPENGL)

    # c = CubeND(3)
    s = SphereND(4, radius=0.5, phitype_nr=5, thetatype_nr=5)
    # t = SimplexND(3, np.random.random((4, 3)) - 0.5)

    oglu.gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    ogl.glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # ogl.glRotatef(1, 3, 1, 1)
        ogl.glClear(ogl.GL_COLOR_BUFFER_BIT | ogl.GL_DEPTH_BUFFER_BIT)
        s.position -= np.array([0., 0., 0.01, 0.0])
        s.rotation_matrix = np.dot(np.array([[np.cos(0.01), -np.sin(0.01), 0, 0],
                                             [np.sin(0.01), np.cos(0.01), 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]]), s.rotation_matrix)
        s.draw_3d()
        #s.draw_conv(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]),
        #            np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]),
        #            np.array([0, 0, 1, 0]))
        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == "__main__":

    main()
    c = CubeND(3)
    s = SphereND(3, radius=0.5, phitype_nr=10, thetatype_nr=10)
    t = SimplexND(3, np.random.random((4, 3)) - 0.5)

    nv = np.random.random(3)
    nv = nv/np.linalg.norm(nv)

    x0 = np.random.random(3) - 0.5
    cut_points = t.cut_edges(nv, x0)

    cv = t.cut_edges_convex_hull(nv, x0, (1, 0, 0), (0, 1, 0))


    #sp_pts = s.pts_com

    #phi = 1./3.*np.pi
    #theta = 2.*np.pi/7.

    #n = np.array([np.cos(phi)*np.sin(theta),
    #              np.sin(phi)*np.sin(theta),
    #              np.cos(theta)])
    #ephi = np.array([-np.sin(phi)*np.sin(theta),
    #                 np.cos(phi)*np.sin(theta), 0])
    #etheta = np.array([np.cos(phi)*np.cos(theta),
    #                   np.sin(phi)*np.cos(theta),
    #                   -np.sin(theta)])

    #cNm1D = c.cut_edges_convex_hull(
    #        n,
    #        np.array([0, 0, 0.5]), ephi, etheta)

    #cpts = cNm1D.points[cNm1D.vertices]

    fig = plt.figure()
    ax = plt3d.Axes3D(fig)
    #convex_hull_plot_2d(cNm1D)

    for (p1_nr, p2_nr) in t.edges:
        drawpoints = [t.points[p1_nr, :], t.points[p2_nr, :]]
        drawpoints = np.array(drawpoints)

        ax.plot(drawpoints[:, 0],
                drawpoints[:, 1],
                drawpoints[:, 2])

    ax.scatter(cut_points[:, 0], cut_points[:, 1], cut_points[:, 2], c="r")
    ax.scatter(x0[0], x0[1], x0[2], c="b")

    arrowpoints = np.vstack((x0, x0 + 0.1*nv))

    ax.plot(arrowpoints[:, 0], arrowpoints[:, 1], arrowpoints[:, 2])

    xl = np.linspace(-0.5, 0.5, 20)
    (X, Y) = np.meshgrid(xl, xl)
    Z = (np.sum(nv*x0) - nv[0]*X - nv[1]*Y)/nv[2]
    Z[np.logical_or(Z < -0.5, Z > 0.5)] = np.nan
    ax.plot_surface(X, Y, Z, alpha=0.2)

    # ax.plot_surface(x,y,z) #seems to be a bug in matplotlib
    # ax.scatter(np.append(X.flatten(), cpts[:,0]),
    #            np.append(Y.flatten(), cpts[:,1]),
    #            np.append(Z.flatten(), cpts[:,2]), "r+")
    # plt.scatter(cpts[:,0], cpts[:,1])
    (xmin, xmax) = ax.get_xlim()
    (ymin, ymax) = ax.get_ylim()
    (zmin, zmax) = ax.get_zlim()
    print(xmin, xmax)
    print(ymin, ymax)
    print(zmin, zmax)

    plt.show()

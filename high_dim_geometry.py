from scipy.spatial import ConvexHull, Delaunay, delaunay_plot_2d, convex_hull_plot_2d
import numpy as np
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d as plt3d



class ConvexHullND(object):
    
    def __init__(self, pts_com):
        (self.num_pts, self.dim) = np.shape(pts_com)
        self.pts_com = pts_com
        self.conv = ConvexHull(pts_com)
        print("dimension %d" % (self.dim,))
        print("shape of point array")
        print(np.shape(self.conv.points))
        print("how much vertices in convex hull?")
        print(np.shape(self.conv.vertices))
        print("maximal vertex number?")
        print(np.max(self.conv.vertices))
        (self.num_conv_pts, tmp) = np.shape(self.conv.points[self.conv.vertices])
        self.generate_edges_from_conv()

    def generate_edges_from_conv(self):
        all_edges_deep = np.dstack((self.conv.simplices, np.roll(self.conv.simplices, 1, axis=1)))
        (num_simps, len_simps, len_edge) = np.shape(all_edges_deep)
        all_edges = all_edges_deep.reshape(1, num_simps*len_simps, len_edge)[0]        
        all_edges.sort(axis=1)

        # remove double entries
        # notice orientation is lost due to sorting before        
        all_edges_set = set([tuple(e) for e  in all_edges.tolist()])
        all_edges_removed_doubles = np.array(list(all_edges_set))
    
        self.edges = all_edges_removed_doubles

    def cut_edges(self, n, x0):
        cut_line = np.hstack((n, -np.dot(x0, n))) # cut_line*(x1, ..., xn, 1) < oder > 0
        pts1 = np.hstack((self.conv.points, np.ones((self.num_pts, 1))))        
        cut_line_rep = np.repeat(cut_line[np.newaxis, :], self.num_pts, axis=0)
                
        notzero = np.sum(pts1*cut_line_rep,axis=1)

        diffsigns = np.sign(notzero[self.edges])
        to_be_cut = np.abs(diffsigns[:,0] - diffsigns[:,1])>0

        two_points_edges = self.conv.points[self.edges[to_be_cut]]
        
        diff = two_points_edges[:,1,:] - two_points_edges[:,0,:]
        startp = two_points_edges[:,0,:]
        
        t = np.sum(n*(x0 - startp),axis=1)/np.sum(n*diff,axis=1)
        
        cutting_points = diff*t[:,np.newaxis] + startp

        return cutting_points
        
    def cut_edges_convex_hull(self, n, x0, *arg):
        cut_pts = self.cut_edges(n, x0)
        proj = np.eye(3) - np.outer(n, n)
        proj_pts = np.dot(proj, cut_pts.T)
        rest_basis = np.array(arg)
        valueNm1D = np.dot(rest_basis, proj_pts)
        
        return ConvexHull(valueNm1D.T) #ConvexHull()

class CubeND(ConvexHullND):
    def __init__(self, n, w=1):
        # generation via meshgrid
        zero_one = w*np.array([-0.5, 0.5])
        ls_zero_one = [zero_one for i in range(n)]
        bits = np.meshgrid(*ls_zero_one) # spits out tuple
        points_unitcube = np.vstack([b.flatten() for b in bits]).T
        super(CubeND, self).__init__(points_unitcube)


class SphereND(ConvexHullND):
    def __init__(self, n, r=1, phitype_nr=10, thetatype_nr=5):
        
        def x(n, angles):
            (num_dimsm1, num_pts) = np.shape(angles)
            res = np.ones_like(angles[0, :])                        
            sinprod = np.ones_like(res)
            if n < num_dimsm1:
                res = np.cos(angles[n, :])
            if n > 0:
                sinprod = np.prod(np.sin(angles[:n, :]), axis=0)
            res = res*sinprod
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
                
        lsangles = [thetaangles for i in range(n-2)] + [phiangles]
        
        angles = np.meshgrid(*lsangles)
        
        all_angles = np.vstack([a.flatten() for a in angles])
        
        points_sphere = np.vstack([x(i, all_angles) for i in range(n)]).T
        #print(points_sphere)
        super(SphereND, self).__init__(points_sphere)


c = CubeND(3)
s = SphereND(3, phitype_nr=10, thetatype_nr=5)

sp_pts = s.pts_com


phi = 1./3.*np.pi
theta = 2.*np.pi/7.

n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
ephi = np.array([-np.sin(phi)*np.sin(theta), np.cos(phi)*np.sin(theta), 0])
etheta = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), -np.sin(theta)])

cNm1D = c.cut_edges_convex_hull(
        n, 
        np.array([0, 0, 0.5]), ephi, etheta)

cpts = cNm1D.points[cNm1D.vertices]


fig = plt.figure()
ax = plt3d.Axes3D(fig)
convex_hull_plot_2d(cNm1D)

ax.scatter(sp_pts[:,0],sp_pts[:,1],sp_pts[:,2])
    #ax.plot_surface(x,y,z) #seems to be a bug in matplotlib
#ax.scatter(np.append(X.flatten(), cpts[:,0]),
#           np.append(Y.flatten(), cpts[:,1]), 
#            np.append(Z.flatten(), cpts[:,2]), "r+")
#plt.scatter(cpts[:,0], cpts[:,1])


plt.show()






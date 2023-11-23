# This tutorial will take you through the basics of Scientific Python.
from scipy import constants

print(constants.liter)  # How many cubic meters are in one liter?

# Constants in SciPy is a module that provide scientific and mathematical constants.
from scipy import constants

print(constants.pi)  # pi is a scientific constant
print(constants.kilo)
print(constants.tera)  # these and many others are the SI Prefixes

# Binary Prefixes
print(constants.kibi)  # kb
print(constants.mebi)  # mb
print(constants.gibi)  # gb
print(constants.tebi)  # tb

# Mass
print(constants.gram)  # g
print(constants.metric_ton)  # mt
print(constants.lb)  # lb
print(constants.oz)  # oz

# Angle
print(constants.degree)
print(constants.arcmin)
print(constants.arcminute)
print(constants.arcsec)
print(constants.arcsecond)

# Time
print(constants.minute)
print(constants.hour)
print(constants.day)
print(constants.week)
print(constants.year)
print(constants.Julian_year)

# Length
print(constants.inch)
print(constants.foot)
print(constants.mile)
print(constants.survey_foot)
print(constants.angstrom)

# Finding the root of the equation x + cos(x) through optimization
from scipy.optimize import root
from math import cos


def eqn(x):
    return x + cos(x)


myroot = root(eqn, 0)

print(myroot)

# SciPy Sparse Data: Data with unused information (elements that don't carry information like zero)
import numpy as np
from scipy.sparse import csr_matrix

arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])

print(csr_matrix(arr))

# Adjucency Matrix: The elements are connected as follows:
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)

print(connected_components(newarr))

# From here we can use Dijkstra method thr find the shortest path in the graph from one element to another.
# The shortest path from element 1 to 2 is;
import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse import csr_matrix

arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)

print(dijkstra(newarr, return_predecessors=True, indices=0))

# Floyd Warshall method of the shortest path between all pairs of elements.
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix

arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)

print(floyd_warshall(newarr, return_predecessors=True))

# Bellman Ford method.
import numpy as np
from scipy.sparse.csgraph import bellman_ford
from scipy.sparse import csr_matrix

arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

newarr = csr_matrix(arr)

print(bellman_ford(newarr, return_predecessors=True))  # The two methods give the same answer

# SciPy Spatial Data: Data represented in a geometric space.
# Spatial Triangulation
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

points = np.array([
    [2, 4],
    [3, 4],
    [3, 0],
    [2, 2],
    [4, 1]
])

simplices = Delaunay(points).simplices
plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')

plt.show()

# Convex Hull: The smallest polygon that covers all the given points.
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

points = np.array([
    [2, 4],
    [3, 4],
    [3, 0],
    [2, 2],
    [4, 1],
    [1, 2],
    [5, 0],
    [3, 1],
    [1, 2],
    [0, 2]
])
hull = ConvexHull(points)
hull_points = hull.simplices
plt.scatter(points[:, 0], points[:, 1])
for simplex in hull_points:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.show()

# KDTrees for the nearest neighbour. Find the nearest neighbour to point (1, 1)
from scipy.spatial import KDTree

points = [(1, -1), (2, 3), (-2, 3), (2, -3)]

kdtree = KDTree(points)

res = kdtree.query((1, 1))

print(res)  # The code returns (2.0, 0) as the nearest neighbour to point (1, 1)

# The Distance Matrices. The Euclidean distance between p1 and p2 is as follows:
from scipy.spatial.distance import euclidean

p1 = (1, 0)
p2 = (10, 2)

res = euclidean(p1, p2)

print(res)

# Cytiblock Distance/ Manhattan Distance; Computed using 4 degrees of movement: Up, Down, Left, Right.
from scipy.spatial.distance import cityblock

p1 = (1, 0)
p2 = (10, 2)

res = cityblock(p1, p2)

print(res)

# Cosine Distance: Value of cosine angle between given points.
from scipy.spatial.distance import cosine

p1 = (1, 0)
p2 = (10, 2)

res = cosine(p1, p2)

print(res)

# Hamming Distance: A way of measuring distance for binary sequence.
from scipy.spatial.distance import hamming

p1 = (True, False, True)
p2 = (False, True, True)

res = hamming(p1, p2)

print(res)

# SciPy Hypothesis Testing: Statistical Significance Tests.
import numpy as np
from scipy.stats import ttest_ind

v1 = np.random.normal(size=100)
v2 = np.random.normal(size=100)

res = ttest_ind(v1, v2)

print(res)

# K-S test: Test if given values follow a given distribution.
import numpy as np
from scipy.stats import kstest, describe, normaltest

v = np.random.normal(size=100)

res = kstest(v, 'norm')

print(res)

res = describe(v)  # Shows the descriptive statistics
print(res)
print(normaltest(v))  # Tests whether the data is normally distributed.
# This marks the end of basic Scientific Python.

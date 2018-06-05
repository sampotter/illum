# Shoot a ray from the center of the octree in a random
# direction. Compare the direction of the centroid of the triangle
# that it hits with the direction of the ray to see if we're hitting
# something that makes sense.

ph, th = np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)
n = np.array([
    np.cos(ph)*np.sin(th),
    np.sin(ph)*np.sin(th),
    np.cos(th)])
p = octree.center

face = next(octree.ray_tri_intersections(p, n))

x = np.mean(v[face], 0) - p
x /= np.linalg.norm(x)

print(x)
print(n)



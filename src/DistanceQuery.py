#this query is needed to find the closest distant to the border
#the paper speaks about the BVH method to reduce the complexity 
#it should be reduced to 0(log(N)) 

#we will use the triangles of the mesh as the primitives shapes.
#we will need a list a vertices then a list of faces consisting in a 3-ple of indices of the list of vertices.
#each triangle will be a primitive then encased into a boundind box(AABB) , defined by two points; max and min.
#the centroid of the triangle provides a position reference of the primitive
#the tree structure will be as follows: a boundind box node will have its two defining points
#and 2 childs boxex (left and right): if it is an internal node of the tree
#if it is a leaf it will have a primitive list instead of childs. the tree will be built recursively by 
#splitting the primitives into two groups using the biggest axis distance of the centroids.

#then using a stack we will explore the tree to find the closest point on the mesh to a query point q.

import numpy as np


#helper function needed later, having a triangle(2D) what would be it's closest point from a (3D) point
#instead of just projecting there is an algo using different cases:
#next to vertex, edge or face. i did not bother fully understanding it i just copied it.

def closest_point_on_triangle(q: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    ab = b - a
    ac = c - a
    ap = q - a

    #Check Vertex Region A
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()

    # 2. Check Vertex Region B
    bp = q - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()

    # 3. Check Edge Region AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    # 4. Check Vertex Region C
    cp = q - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()

    # 5. Check Edge Region AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    # 6. Check Edge Region BC
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    # 7. Interior Face Region
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w




#____________________________________________________________________
#now we start creating our subclasses for the BHV:

#boundind box object ; 
class AABB:
    def __init__(self, min_pt: np.ndarray, max_pt: np.ndarray):
        self.min_pt = min_pt 
        self.max_pt = max_pt  


#the primitives we will be working with
class Primitive:
    def __init__(self, index: int, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, aabb: AABB, centroid: np.ndarray):
        self.index = index          #original index
        self.v0 = v0                
        self.v1 = v1                
        self.v2 = v2                
        self.aabb = aabb            # Bounding box of the triangle
        self.centroid = centroid    

    def closest_point(self, q: np.ndarray) -> np.ndarray:
        return closest_point_on_triangle(q, self.v0, self.v1, self.v2)

#node of our tree
class BVHNode:
    def __init__(self, aabb: AABB):
        self.aabb = aabb
        self.left = None
        self.right = None
        self.primitives = []  # Non-empty = leaf nodes


#main class we will work with
class BVH:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, max_leaf_size: int = 4):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)

        #this is a hyper-parameter to control how the tree goes, probably chosen by feeling more than anything
        self.max_leaf_size = max_leaf_size

        #we parse the data into usable primitives
        self.primitives = self.parse_mesh()

        #create the tree and get the root
        self.root = self.build_recursive(self.primitives)


    #transform our mesh data into primitive data
    def parse_mesh(self):
        #verticise is (N,3)
        #faces is (M,3)
        tri_vertices = self.vertices[self.faces]  # is then (M,3,3); 0 dim= face, 1dim=index of coord, 2dim= 3 coords of the point.

        #we separate for each corner of the triangle
        v0 = tri_vertices[:, 0, :]
        v1 = tri_vertices[:, 1, :]
        v2 = tri_vertices[:, 2, :]

        #we search for the extreme points of the triangles for the aabb
        min_pts = np.min(tri_vertices, axis=1)  #(M, 3)
        max_pts = np.max(tri_vertices, axis=1)  #(M, 3)
        centroids = (v0 + v1 + v2) / 3.0  #(M,1)


        #we build the primitives
        primitives = []
        for i in range(len(self.faces)):
            box = AABB(min_pt=min_pts[i], max_pt=max_pts[i])

            prim = Primitive(index=i, v0=v0[i], v1=v1[i], v2=v2[i] ,aabb=box, centroid=centroids[i])
            
            primitives.append(prim)
            
        return primitives


    #now we build the tree recursively from a list of primitives
    def build_recursive(self, primitives):

        #end of recursion if no primitives left
        if len(primitives) == 0 or primitives is None:
            return None

        #first we find the AABB for the current primitive list
        # Find global min and max 
        min_pt = np.min([p.aabb.min_pt for p in primitives], axis=0)
        max_pt = np.max([p.aabb.max_pt for p in primitives], axis=0)
        #create the new node with the aabb
        node_aabb = AABB(min_pt,max_pt)
        node = BVHNode(node_aabb)

        # leaf criteria reached (arbitrary chosen when creating the class object)
        if len(primitives) <= self.max_leaf_size:
            node.primitives = primitives
            return node

        #Choose split axis (finding the biggest axis diff)
        centroids = np.array([p.centroid for p in primitives])
        centroid_min = np.min(centroids, axis=0)
        centroid_max = np.max(centroids, axis=0)
        extent = centroid_max - centroid_min
        axis = int(np.argmax(extent))

        #AI found this case, if they are too close to each other just create a leaf
        if extent[axis] < 1e-12:
            node.primitives = primitives
            return node

        #Median split along chosen axis
        #AI found a much cleaner way to write it, using lambda functions.
        primitives.sort(key=lambda p: p.centroid[axis])
        mid = len(primitives) // 2
        left_prims = primitives[:mid]
        right_prims = primitives[mid:]

        #tree building the internal node. (.primitives stays empty)
        node.left = self.build_recursive(left_prims)
        node.right = self.build_recursive(right_prims)

        return node




    #_______________________________________________
    #now for the final query, the algo usually uses a stack, so we will do that
    #first we need the distance from a point to an AABB so we can explore the tree
    #in a good order. this distance must be set to 0 if inside or touching the box

    def distance_point_aabb(self, q: np.ndarray, aabb):

        # d_min > 0 only when q is to the left/below the box
        d_min = np.maximum(0.0, aabb.min_pt - q)

        # d_max > 0 only when q is to the right/above the box
        d_max = np.maximum(0.0, q - aabb.max_pt)
    
        distance = d_min + d_max
        distance = np.dot(distance, distance)
        return distance



    #closest point query, returns the closest point on the mesh to the query point q, 
    #the distance and the index of the triangle
    def closest_point(self, q: np.ndarray):

        # initialize the variables
        best_pt = None
        best_sq_dist = float("inf") #-infinity 
        best_tri_idx = -1
        stack = [self.root]

        while stack:
            node = stack.pop()

            #we can skip this node and all children if the distance is bigger than best
            if self.distance_point_aabb(q, node.aabb) >= best_sq_dist: 
                continue

            if node.primitives:  # if Leaf node
                for p in node.primitives:
                    pt = p.closest_point(q)
                    d2 = np.sum((q - pt) ** 2)
                    if d2 < best_sq_dist:
                        best_sq_dist = d2
                        best_pt = pt
                        best_tri_idx = p.index
            else:
                d_left = self.distance_point_aabb(q, node.left.aabb)
                d_right = self.distance_point_aabb(q, node.right.aabb)


                #order is important, we want to explore the closest box first
                if d_left < d_right:
                    if d_right < best_sq_dist: stack.append(node.right)
                    if d_left < best_sq_dist:  stack.append(node.left)
                else:
                    if d_left < best_sq_dist:  stack.append(node.left)
                    if d_right < best_sq_dist: stack.append(node.right)

        return best_pt, np.sqrt(best_sq_dist), best_tri_idx




#AI made test on standar mesh found on matplotlib or also this repo : https://github.com/NicolasDouillet/3D_test_shapes_repository

if __name__ == "__main__":
    import trimesh
    import polyscope as ps
    import os

    # 1. Path to your downloaded mesh (supports .obj, .ply, .stl, etc.)
    # Change "bunny.obj" to your actual filename inside assets/
    mesh_path = os.path.join(os.path.dirname(__file__), "assets", "Gargoyle_5k.ply")
    
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Could not find mesh file at: {mesh_path}")

    # force='mesh' guarantees trimesh attempts to construct faces or raises an explicit warning
    loaded = trimesh.load(mesh_path, force='mesh')

    # If the file contains multiple sub-meshes or a Scene, extract the primary geometry
    if isinstance(loaded, trimesh.Scene):
        # Concatenate all sub-meshes into a single Trimesh
        mesh = trimesh.util.concatenate([geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)])
    else:
        mesh = loaded

    # Verify that the loaded object is a surface mesh with faces
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError(
            f"The file '{mesh_path}' loaded as a {type(mesh).__name__} without faces. "
            "Please download a triangulated mesh (e.g. .obj with 'f' lines) rather than a point cloud."
        )

    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)

    # 2. Build BVH
    print(f"Building BVH for {len(faces)} triangles...")
    bvh = BVH(vertices, faces, max_leaf_size=4)
    print("BVH construction complete.")

    # 3. Create a line of query points positioned far outside the mesh
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    max_span = float(np.max(extent))

    # A line passing above and diagonally outside the model
    # Span reaches from -2x to +2x the mesh extent
    num_query_points = 100
    t = np.linspace(-1.5, 1.5, num_query_points)

    # Line trajectory: offset significantly along Y and X
    line_start = center + np.array([-1.5 * max_span, 1.8 * max_span, -0.8 * max_span])
    line_end = center + np.array([1.5 * max_span, 1.8 * max_span, 0.8 * max_span])
    query_points = np.array([line_start + s * (line_end - line_start) for s in np.linspace(0.0, 1.0, num_query_points)])

    # 4. Evaluate closest point queries
    print(f"Querying {num_query_points} points against BVH...")
    projected_points = []
    face_hit_counts = np.zeros(len(faces), dtype=np.float64)

    for q in query_points:
        p_surf, dist, tri_idx = bvh.closest_point(q)
        projected_points.append(p_surf)
        face_hit_counts[tri_idx] += 1.0

    projected_points = np.array(projected_points)

    # 5. Visualize in Polyscope
    ps.init()
    ps.set_up_dir("y_up")

    # Gargoyle mesh with painted hit regions
    ps_mesh = ps.register_surface_mesh("Gargoyle", vertices, faces)
    ps_mesh.add_scalar_quantity("Closest Face Hits", face_hit_counts, defined_on="faces", enabled=True)

    marker_radius = max_span * 0.007

    # Line of query points (orange)
    ps_queries = ps.register_point_cloud("Query Line Points", query_points, radius=marker_radius)
    ps_queries.set_color((1.0, 0.5, 0.0))

    # Projected surface points (cyan)
    ps_proj = ps.register_point_cloud("Projected Surface Points", projected_points, radius=marker_radius)
    ps_proj.set_color((0.0, 0.8, 1.0))

    # Lines connecting the query line to the gargoyle surface (magenta)
    edge_nodes = np.vstack([query_points, projected_points])
    edges = np.array([[i, i + num_query_points] for i in range(num_query_points)])
    ps_lines = ps.register_curve_network("Projection Rays", edge_nodes, edges)
    ps_lines.set_color((0.9, 0.1, 0.4))
    ps_lines.set_radius(marker_radius * 0.3)

    print("Launching Polyscope viewer...")
    ps.show()
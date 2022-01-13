import cairo
import random
import math
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiPolygon, MultiLineString
from shapely.ops import unary_union, split
import colorsys
from sklearn.neighbors import KNeighborsClassifier

import texture
from params import *
from tqdm import tqdm
import trimesh
import pyrender

size_half = (size[0] / 2, size[1] / 2)

surface = cairo.ImageSurface(cairo.FORMAT_RGB24, *size)
ctx = cairo.Context(surface)
ctx.translate(*size_half)
ctx.scale(*size_half)

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def branch(position, angle, type, depth, hsv, max_depth, length_range, thickness, angle_range, paths):

    angle_rad = deg_to_rad(angle)
    length = random.uniform(length_range[0], length_range[1])
    destination = (position[0] + math.cos( angle_rad ) * length,
                   position[1] + math.sin( angle_rad ) * length)
    ctx.move_to(*position)
    ctx.line_to(*destination)
    rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1] / np.sqrt(depth), hsv[2] / np.sqrt(depth))
    ctx.set_source_rgb(rgb[0], rgb[1], rgb[2])
    ctx.set_line_width(thickness)
    ctx.stroke()

    # add the line segment to our paths
    paths.append( LineString( [Point(position), Point(destination)]))

    if depth < max_depth:
        angle_left = random.uniform(-angle_range[1], angle_range[0])
        angle_right = random.uniform(angle_range[0], angle_range[1])

        branch(destination, angle + angle_left, type, depth + 1, hsv, max_depth, length_range,
                   thickness - thickness_branch_reduction , angle_range, paths)

        branch(destination, angle + angle_right, type, depth + 1, hsv, max_depth, length_range,
                   thickness - thickness_branch_reduction , angle_range, paths)


def perform_union(paths):
    # now we union the lines
    print('performing union')
    union = unary_union(paths)
    print(' done')
    # grab the end points to determine the intersections
    nodes = [c for l in union.geoms for c in l.coords]
    nodes = np.array(nodes)
    # now we remove the duplicate points and retrieve the counts
    # the counts are the degrees of each node
    nodes, counts = np.unique(nodes, axis=0, return_counts=True)
    return nodes, counts

def create_quad(coords):
    faces = [(0,1,3), (1,2,3)]
    return trimesh.Trimesh(vertices = coords, faces = faces)

def render_building(x, y, max_radius, height_offset, building_walls, building_roofs):
    dist = np.sqrt(x ** 2 + y ** 2)
    angle = math.atan2(x,y)
    xn, yn = x / dist, y / dist
    # grab the orthogonal direction
    xo, yo = yn, -xn
    random_rate = dist * 0.004 + 0.002
    random_rate_half = random_rate / 2
    forw_random = random.random() * random_rate - random_rate_half
    perp_random = random.random() * random_rate - random_rate_half
    forw_distance = 0.005 + forw_random
    perp_distance = 0.005 + perp_random

    x1 = x - xn * forw_distance - xo * perp_distance
    y1 = y - yn * forw_distance - yo * perp_distance
    x2 = x - xn * forw_distance + xo * perp_distance
    y2 = y - yn * forw_distance + yo * perp_distance
    x3 = x + xn * forw_distance + xo * perp_distance
    y3 = y + yn * forw_distance + yo * perp_distance
    x4 = x + xn * forw_distance - xo * perp_distance
    y4 = y + yn * forw_distance - yo * perp_distance

    ctx.move_to(x1, y1)
    ctx.line_to(x2, y2)
    ctx.set_line_width(0.001)

    hue = random.random() / 7.0 + 0.5 * angle / np.pi
    sat = 1.0 - (random.random() / 7.0 + 1.1 * dist)
    val = random.random() / 2.0 + 0.5
    rgb = colorsys.hsv_to_rgb(hue, sat, val)

    ctx.set_source_rgb(*rgb)
    ctx.stroke()
    ctx.move_to(x2, y2)
    ctx.line_to(x3, y3)
    ctx.stroke()
    ctx.move_to(x3, y3)
    ctx.line_to(x4, y4)
    ctx.stroke()
    ctx.move_to(x4, y4)
    ctx.line_to(x1, y1)
    ctx.stroke()

    base_height = terrain_get_base_height(max_radius, height_offset) + terrain_get_extrude_height(max_radius, dist) - 0.004
    building_height = (1.0 - dist) * building_height_rate + random.random() * 0.02 - 0.01
    building_height = max(building_height, 0.006)
    top_height = base_height + building_height


    building_walls.append(
        create_quad([(x1, y1, base_height), (x2, y2, base_height), (x2, y2, top_height), (x1, y1, top_height)]))
    building_walls.append(
        create_quad([(x2, y2, base_height), (x3, y3, base_height), (x3, y3, top_height), (x2, y2, top_height)]))
    building_walls.append(
        create_quad([(x3, y3, base_height), (x4, y4, base_height), (x4, y4, top_height), (x3, y3, top_height)]))
    building_walls.append(
        create_quad([(x4, y4, base_height), (x1, y1, base_height), (x1, y1, top_height), (x4, y4, top_height)]))

    building_roofs.append(
        create_quad([(x1, y1, top_height), (x2, y2, top_height), (x3, y3, top_height), (x4, y4, top_height)]))

def terrain_get_base_height(island_max_radius, island_height_offset):
    return 0.8 * (1.0 - island_max_radius) * terrain_height_rate + island_height_offset

def terrain_get_extrude_height(island_max_radius, radius):
    radius_difference = island_max_radius - radius
    return radius_difference * terrain_height_rate + 0.001

def get_path_quads(source, destination, source_island, destination_island, island_max_radius, island_height_offsets,
                   floating = False):

    fstx, fsty = source
    source_radius = math.sqrt(fstx ** 2 + fsty ** 2)

    fdex, fdey = destination
    destination_radius = math.sqrt(fdex ** 2 + fdey ** 2)
    dx, dy = fdex - fstx, fdey - fsty
    fdis = np.sqrt(dx ** 2 + dy ** 2)

    path_quads = []
    path_uvs = []

    source_height = terrain_get_base_height(island_max_radius[source_island],
                                                island_height_offsets[source_island]) + \
                        terrain_get_extrude_height(island_max_radius[source_island], source_radius) + 0.01

    destination_height = terrain_get_base_height(island_max_radius[destination_island],
                                                     island_height_offsets[destination_island]) + \
                        terrain_get_extrude_height(island_max_radius[destination_island], destination_radius) + 0.01


    kstep = min(0.005, fdis)
    for inter in np.arange(kstep, fdis + 0.00001, kstep):
        step = inter / fdis

        ostep = (inter - kstep)/fdis

        stx = fstx * (1 - ostep) + fdex * ostep
        sty = fsty * (1 - ostep) + fdey * ostep
        dex = fstx * (1 - step) + fdex * step
        dey = fsty * (1 - step) + fdey * step

        source_radius = math.sqrt(stx ** 2 + sty ** 2)
        destination_radius = math.sqrt(dex ** 2 + dey ** 2)

        sourceh = source_height * (1 - ostep) + destination_height * ostep
        desth = source_height * (1 - step) + destination_height * step

        dx, dy = dex - stx, dey - sty
        dis = np.sqrt(dx ** 2 + dy ** 2)
        fx, fy = dx / dis, dy / dis
        sx, sy = fy, -fx
        average_radius = (source_radius + destination_radius) / 2.0
        thickness = (1 - average_radius) ** 2 / 300.0
        thickness = max(thickness, 0.0004)
        tsx, tsy = sx * thickness, sy * thickness


        path_quad = create_quad([(stx - tsx, sty - tsy, sourceh),
                                       (stx + tsx, sty + tsy, sourceh),
                                       (dex + tsx, dey + tsy, desth),
                                       (dex - tsx, dey - tsy, desth)])

        end_uv = 0.5 * dis/thickness
        path_uv = [(0, 0), (0, end_uv), (1, 0), (1, end_uv)]

        path_quads.append(path_quad)
        path_uvs.extend(path_uv)

    if not floating:
        return path_quads, path_uvs

    middle_height = (source_height + destination_height)/2 - 0.002
    tsx*=3.5
    tsy*=3.5
    fsx = fx * thickness * 1.5
    fsy = fy * thickness * 1.5
    mx, my = (stx+dex)/2, (sty+dey)/2
    vertices = [(mx - tsx - fsx, my - tsy - fsy),
                             (mx + tsx - fsx, my + tsy - fsy),
                             (mx + tsx + fsx, my + tsy + fsy),
                             (mx - tsx + fsx, my - tsy + fsy)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    poly = trimesh.path.polygons.edges_to_polygons(edges, np.array(vertices))[0]
    base_extruded = trimesh.creation.extrude_polygon(poly, 0.002)
    base_extruded.vertices[:, 2] += middle_height - 0.002

    return path_quads, path_uvs, base_extruded

# render the branch starting at the root
# our path list
paths = []
path_features = []
for angle in range(0, 359, branch_angle_increment):
    hue = angle / 360.0
    saturation = 1.0
    value = 1.0
    max_depth = random.randint(3, 9)
    length = random.random()/20.0 + 0.05
    thickness = 0.002
    branch((0.0, 0.0), angle + random.random() * 10.0 - 5.0, 1, 1, (hue, saturation, value),
           max_depth, [length - 0.02, length + 0.06], thickness, [15.0, 25.0], paths)

nodes, counts = perform_union(paths)

circles = []
for node in nodes:
    x, y = node
    distance = np.sqrt(x ** 2 + y ** 2) + 0.000001
    radius = influence_radius_multiplier / distance
    radius = min(radius, influence_radius_max)
    radius = max(radius, influence_radius_min)
    point = Point(x,y)
    circle = point.buffer(radius, resolution=influence_circle_resolution)
    coords = circle.exterior.coords[:]
    noise_factor = (inner_circle_noise_factor) * distance
    coords2 = coords + np.random.random( (len(coords), 2)) * noise_factor
    coords = list(map(tuple, coords2))
    circle = Polygon( coords ).buffer(0)
    circles.append(circle)

print ('create window texture')
window_texture = texture.create_window_texture()
path_texture = texture.create_path_texture()

print ('performing circle union')
circle_union = unary_union(circles)
circle_union = circle_union.intersection( Point(0,0).buffer(inner_circle_max, resolution=8))
print ('done')

# let's determine the number of islands, and for each island several properties
print ('generating island heights and min/max radiuses')
islands = list(circle_union.geoms)
island_height_offsets = []
island_min_exact_radius = []
island_max_exact_radius = []
center_point = Point(0,0)
for island in islands:
    island_height_offset = random.random() * 0.1
    island_height_offsets.append(island_height_offset)
    # determine the min and max radiuses
    x,y = island.exterior.coords.xy
    x = np.array(x)
    y = np.array(y)
    rad = np.sqrt(x**2 + y**2)
    if island.contains(center_point):
        minrad = 0.0
    else:
        minrad = np.min(rad)
    island_min_exact_radius.append(minrad)
    island_max_exact_radius.append(np.max(rad))

print ('done')


# generate the inner circles
inner_circles = []
meshes = []
island_max_radius = np.zeros(len(islands))

splitter = LineString([Point(0, -1), Point(0, 1)])

path_geoms = []
path_uvs = []
path_ring_geoms = []
point = Point(0, 0)
for path_iter in range(0,2):
    if path_iter == 0:
        print ('generating inner ring geometries')
    else:
        print ('generating path ring geometries')

    for radius in tqdm(np.arange(inner_circle_max, inner_circle_min, -inner_circle_increment)):
        circle = point.buffer(radius, resolution = 8)
        if path_iter == 0:
            circle_inner = point.buffer(radius - inner_circle_increment, resolution = 8)
        else:
            circle_inner = point.buffer(radius - inner_circle_increment/8, resolution = 8)
        ring = circle - circle_inner

        inner_circle = circle_union.intersection(ring)
        if path_iter == 0:
            inner_circles.append(inner_circle)

        if not inner_circle.is_empty:


            lines = []
            if path_iter == 1:
                splitted_ring = split(inner_circle, splitter)
                for geom in splitted_ring.geoms:
                    boundary = geom.boundary
                    if boundary.type == 'MultiLineString':
                        for line in boundary.geoms:
                            lines.append(line)
                    else:
                        lines.append(boundary)
            else:
                boundary = inner_circle.boundary

                if boundary.type == 'MultiLineString':
                    for line in boundary.geoms:
                        lines.append(line)
                else:
                    lines.append(boundary)

            for line in lines:
                coords = line.coords
                for i in range(len(coords) - 1):
                    ctx.move_to(coords[i][0], coords[i][1])
                    ctx.line_to(coords[i + 1][0], coords[i + 1][1])
                    ctx.set_line_width(0.001)
                    ctx.set_source_rgb(radius, radius, 1.0)
                    ctx.stroke()

                # intersect the geometry with the relevant islands to get the correct height
                # this could be more efficient
                # use the precomputed min and max radiuses to speed this up a bit
                for island_index in range(len(islands)):
                    if radius > island_min_exact_radius[island_index] and \
                        radius < island_max_exact_radius[island_index] + inner_circle_increment:
                        if islands[island_index].intersects(line):
                            break

                if island_max_radius[island_index] < radius:
                    island_max_radius[island_index] = radius
                # create the ring geometries
                edges = np.array([np.arange(0, len(coords)-1), np.arange(1, len(coords))]).T
                edges[-1, 1] = 0
                poly = trimesh.path.polygons.edges_to_polygons(edges, np.array(coords))[0]

                if len(coords) > 3:
                    try:
                        base_height = terrain_get_base_height(island_max_radius[island_index],
                                                              island_height_offsets[island_index])
                        extrude_height = terrain_get_extrude_height(island_max_radius[island_index], radius)
                        if path_iter == 0:
                            mesh = trimesh.creation.extrude_polygon(poly, extrude_height)
                            mesh.vertices[:, 2] += base_height
                            meshes.append(mesh)
                        else:
                            base_height+=extrude_height
                            mesh = trimesh.creation.extrude_polygon(poly, 0.0005)
                            mesh.vertices[:, 2] += base_height
                            path_ring_geoms.append(mesh)

                    except:
                        pass


# generate the building candidates
print ('\ndone')
print ('generating building candidates')
building_points = []
for inner_radius in np.arange(inner_circle_max, inner_circle_min, -inner_circle_increment):

    number_of_buildings = int(inner_radius * building_rate)
    number_of_buildings += random.randint(0, int(number_of_buildings * building_rate_upper_factor))
    angle_offset = random.random() * angle_offset_factor
    angle_inc = 2.0 * np.pi / number_of_buildings

    for angle_original in np.arange(0.0, np.pi * 2.0 - 0.0001, angle_inc):
        angle = angle_original + angle_offset

        if random.random() > 0.65 + inner_radius:
            outer_radius = inner_radius + 0.02
            x1 = np.cos(angle) * inner_radius
            y1 = np.sin(angle) * inner_radius
            x2 = np.cos(angle) * outer_radius
            y2 = np.sin(angle) * outer_radius
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.set_line_width(0.001)
            ctx.set_source_rgb(1.0,1.0,1.0)
            ctx.stroke()

        elif random.random() > inner_radius * 0.6:
            x = np.cos(angle) * (inner_radius + 0.0065)
            y = np.sin(angle) * (inner_radius + 0.0065)
            point = Point(x, y)
            building_points.append(point)

print('\ndone')

# intersect the building candidates with the islands
building_walls = []
building_roofs = []
building_points = MultiPoint(building_points)
e = 0
print ('intersecting islands with building candidates')
for island in tqdm(islands):
    intersection = island.intersection(building_points)
    if intersection:
        max_radius = island_max_radius[e]
        height_offset = island_height_offsets[e]
        # loop through the intersected buildings and add them to our geometries
        if isinstance(intersection, MultiPoint):
            point_list = intersection.geoms
        else:
            point_list = [intersection]

        for i in point_list:
            x,y = i.coords[0]
            render_building(x,y, max_radius, height_offset, building_walls, building_roofs)
    e+=1

# generate the island bases
island_base_meshes = []
e = 0
print ('generating island bases')
for island in tqdm(islands):
    x, y = island.centroid.coords[0]
    area = island.area
    ctx.arc(x,y, 0.007, 0.0, math.pi*2.0)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.set_line_width(0.003)
    ctx.stroke()

    base_height = terrain_get_base_height(island_max_radius[e], island_height_offsets[e])
    base_coords = [(lx,ly,base_height) for lx,ly in island.exterior.coords]
    center_index = len(base_coords)
    depth = area
    depth = max(area, 0.02)
    depth = min(depth, 0.05)
    bottom_height = base_height - depth

    r1 = 0.5 + area ** .35
    r1 = min(0.95, r1)

    bottom_coords = []
    ncoords = len(island.exterior.coords)
    offset = random.random()
    for ee,(lx,ly) in enumerate(island.exterior.coords):
        qr1 = r1 + random.random() * .02
        qr2 = 1 - qr1
        tx = lx * qr1 + x * qr2
        ty = ly * qr1 + y * qr2
        tz = bottom_height - np.sin(offset + 2.0 * np.pi * ee / ncoords) * depth / 4.0
        bottom_coords.append((tx, ty, tz))

    num_coords = len(base_coords)
    base_coords.extend(bottom_coords)

    faces1 = [ (i, i+1, i+num_coords) for i in range(num_coords-1)]
    faces1.append( (num_coords-1, 0, num_coords))
    faces2 = [ (i+1, i+num_coords+1, i+num_coords) for i in range(num_coords-1)]
    faces2.append( (0, num_coords, num_coords*2-1))
    faces1.extend(faces2)

    island_base_mesh = trimesh.Trimesh(vertices=base_coords,faces=faces1)

    # clip the bottom of the mesh using a translated bounding box
    #island_base_mesh.visual.mesh.visual.face_colors = [212, 212, 212, 255]
    island_base_meshes.append(island_base_mesh)
    e+=1

print ('intersecting paths with islands')
print ('generating path union')
path_union = unary_union(paths)

# here we store the computed heights for later use by the
# nearest neighbor classifier


path_points = []
path_point_islands = []

print ('generating path geometries')
e = 0
for island in tqdm(islands):
    path_cut = island.intersection(path_union)

    if isinstance(path_cut, MultiLineString):
        iterator = path_cut.geoms
    else:
        iterator = [path_cut]

    for line_segment in iterator:
        if len(line_segment.coords)>1:
            source = line_segment.coords[0]
            destination = line_segment.coords[1]
            path_geom, path_uv = get_path_quads(source, destination, e, e, island_max_radius, island_height_offsets)
            path_geoms.extend(path_geom)
            path_uvs.extend(path_uv)

            path_points.append(source)
            path_point_islands.append(e)
            path_points.append(destination)
            path_point_islands.append(e)
    e+=1

print ('\n done')

print ('performing nearest neighbor island lookups')

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(path_points, path_point_islands)

print ('performing path difference to determine floating areas')
path_floating = path_union.difference(circle_union)
path_floating = path_floating.intersection( Point(0,0).buffer(inner_circle_max, resolution=8))
print (' done')

print ('generating floating path geometries')
for floating_path in path_floating.geoms:
    if len(floating_path.coords) > 1:
        source = floating_path.coords[0]
        destination = floating_path.coords[1]

        source_island = int(neigh.predict([source])+0.4)
        destination_island = int(neigh.predict([destination]) + 0.4)


        path_geom, path_uv, base_geom = get_path_quads(source, destination, source_island, destination_island, island_max_radius,
                                                       island_height_offsets, floating=True)

        path_geoms.extend(path_geom)
        path_uvs.extend(path_uv)
        island_base_meshes.append(base_geom)

print (' done')






#for node in nodes:
#    x, y = node
#    distance = np.sqrt(x ** 2 + y ** 2) + 0.000001
#    radius = influence_radius_multiplier / distance
#    radius = min(radius, influence_radius_max)
#    radius = max(radius, influence_radius_min)
#    ctx.arc(x,y, radius, 0.0, math.pi * 2.0)
#    ctx.set_source_rgb(0.9, 0.4, 0.4)
#    ctx.set_line_width(0.0001)
#    ctx.stroke()




#surface.write_to_png(out_image)
#os.system(out_image)
terrain_concat = trimesh.util.concatenate(meshes)
island_base_concat = trimesh.util.concatenate(island_base_meshes)
building_walls_concat = trimesh.util.concatenate(building_walls)
building_roofs_concat = trimesh.util.concatenate(building_roofs)
path_concat = trimesh.util.concatenate(path_geoms)
path_ring_concat = trimesh.util.concatenate(path_ring_geoms)
#building_concats = []
#for i in range(0, len(building_geoms), 400):
#    building_concats.append( trimesh.util.concatenate(building_geoms[i:i+400] ))
#geometry = [terrain_concat, island_base_concat]
#geometry.append(building_geoms)
#scene = trimesh.scene.Scene(geometry=geometry)


scene = trimesh.scene.Scene()
scene.add_geometry(terrain_concat, node_name = 'terrain')
scene.add_geometry(island_base_concat, node_name = 'island_base', parent_node_name = 'terrain')
scene.add_geometry(building_walls_concat, node_name = 'building_walls', parent_node_name = 'terrain')
scene.add_geometry(building_roofs_concat, node_name = 'building_roofs', parent_node_name = 'terrain')
scene.add_geometry(path_concat, node_name = 'paths', parent_node_name = 'terrain')
scene.add_geometry(path_ring_concat, node_name = 'path_rings', parent_node_name = 'terrain')

uvs_wall = [(0, 0), (0, 1), (1, 0), (1, 1)]*len(building_walls)
uvs_roof = [(0, 0), (0, 1), (1, 0), (1, 1)]*len(building_roofs)


material_wall = trimesh.visual.material.PBRMaterial(name='building', baseColorFactor=[255, 100, 100], metallicFactor=0.7,
                                               roughnessFactor=0.1, emissiveFactor=[1.0,0.6,1.0],
                                               emissiveTexture=window_texture)
building_walls_concat.visual = trimesh.visual.TextureVisuals(material=material_wall, uv=uvs_wall)

material_roof = trimesh.visual.material.PBRMaterial(name='roof', baseColorFactor=[180,180,180], metallicFactor=0.9,
                                                    roughnessFactor=0.35)

building_roofs_concat.visual = trimesh.visual.TextureVisuals(material=material_roof, uv=uvs_roof)

material_path = trimesh.visual.material.PBRMaterial(name='path', baseColorFactor=[180,180,180,250], metallicFactor=0.9,
                                                    roughnessFactor=0.1, emissiveFactor=[1.0, 0.0, 1.0],
                                                    emissiveTexture=window_texture)
path_concat.visual = trimesh.visual.TextureVisuals(material=material_path, uv=path_uvs)

path_ring_concat.visual.mesh.visual.face_colors = [80,129,255,200]





#for e, building in enumerate(building_geoms[:2], 1):
#    scene.add_geometry(building, node_name = f'building{e}', parent_node_name = 'terrain')


trimesh.exchange.export.export_scene(scene, 'output/out.glb', file_type='glb')


#print ('spawning keyshot')
#subprocess.Popen([r"c:\program files\KeyShot10\bin\keyshot.exe","-script","keyshot.py"])
#print (' done!')

pyrender_terrain = pyrender.Mesh.from_trimesh(terrain_concat, smooth=False)
pyrender_building_walls = pyrender.Mesh.from_trimesh(building_walls_concat, smooth=False)
pyrender_building_roofs = pyrender.Mesh.from_trimesh(building_roofs_concat, smooth=False)
pyrender_island_bases = pyrender.Mesh.from_trimesh(island_base_concat, smooth=False)
pyrender_paths = pyrender.Mesh.from_trimesh(path_concat, smooth=False)
pyrender_path_rings = pyrender.Mesh.from_trimesh(path_ring_concat, smooth=False)

scene = pyrender.Scene()
scene.add(pyrender_terrain)
scene.add(pyrender_building_walls)
scene.add(pyrender_building_roofs)
scene.add(pyrender_island_bases)
scene.add(pyrender_paths)
scene.add(pyrender_path_rings)
pyrender.Viewer(scene, viewport_size=(2400, 1000),
                use_raymond_lighting=True,
                shadows=False,
                window_title='City Art')



import json
import networkx as nx
from math import sqrt
from shapely.geometry import Point, shape
from const import NODES_PATH, BUILDING_EDGE_PATH, ROOMS_PATH, ROUTING_PATH


def build_graph():
    """
    Build and return a graph G based on the data from the specified GeoJSON file.
    Nodes and edges are added to the graph with Euclidean distances as edge weights.
    """
    nodes = json.load(open(NODES_PATH, 'r', encoding='utf-8'))
    G = nx.Graph()

    # Add nodes to the graph
    for feature in nodes['features']:
        node_id = feature['properties']['id']
        label = feature['properties']['label']
        coordinates = feature['geometry']['coordinates']
        G.add_node(node_id, label=label, coordinates=coordinates)

    # Add edges between nodes based on neighbors and calculate edge weights
    for feature in nodes['features']:
        node_id = feature['properties']['id']
        neighbors = feature['properties']['neighbors']
        if neighbors:
            neighbors = [int(n.strip()) for n in neighbors.split(',')]
            for neighbor in neighbors:
                if not G.has_edge(node_id, neighbor):
                    coord1 = G.nodes[node_id]['coordinates']
                    coord2 = G.nodes[neighbor]['coordinates']
                    distance = sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
                    G.add_edge(node_id, neighbor, weight=distance)

    return G


# Heuristic function for A*
def heuristic(a, b, G):
    coord_a = G.nodes[a]['coordinates']
    coord_b = G.nodes[b]['coordinates']
    return sqrt((coord_a[0] - coord_b[0]) ** 2 + (coord_a[1] - coord_b[1]) ** 2)


# Find the closest node to a given coordinate
def find_closest_node(coord, G):
    closest_node = None
    min_distance = float('inf')
    for node_id, data in G.nodes(data=True):
        node_coord = data['coordinates']
        distance = sqrt((node_coord[0] - coord[0]) ** 2 + (node_coord[1] - coord[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_node = node_id
    return closest_node


# Function to check if a point is inside the boundary
def is_within_boundary(coord, boundary_geojson):
    """
    Checks if a given coordinate is within the boundary defined by the GeoJSON.

    :param coord: A tuple of (x, y) coordinates.
    :param boundary_geojson: GeoJSON object defining the boundary.
    :return: True if the coordinate is inside the boundary, False otherwise.
    """
    point = Point(coord)

    # Extract the MultiPolygon from the GeoJSON features
    for feature in boundary_geojson['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(point):
            return True
    return False


def find_closest_node_or_room(coord, G, rooms_geojson):
    """
    Find the closest node to a given coordinate, or return the node associated with a room
    if the coordinate is inside a room polygon.

    :param coord: (x, y) tuple of coordinates.
    :param G: Graph of nodes and edges.
    :param rooms_geojson: GeoJSON containing room information.
    :return: Closest node or room node.
    """
    point = Point(coord)

    # Check if the point is inside any room
    for feature in rooms_geojson['features']:
        room_polygon = shape(feature['geometry'])
        if room_polygon.contains(point):
            # Return the node associated with the room
            room_node = int(feature['properties']['nodes'])
            room_name = feature['properties']['Name']
            print(f"Coordinate is inside room: {room_name}")
            return room_node

    # If not inside any room, find the closest node as usual
    closest_node = find_closest_node(coord, G)
    return closest_node


# Routing function: find the path between two coordinates using A* algorithm
def routing(start_coord, end_coord, boundary_geojson, rooms_geojson):
    # Check if both coordinates are within the boundary
    if not is_within_boundary(start_coord, boundary_geojson):
        return "Start coordinate is not within the specified boundary.", None
    if not is_within_boundary(end_coord, boundary_geojson):
        return "End coordinate is not within the specified boundary.", None

    G = build_graph()

    start_node = find_closest_node_or_room(start_coord, G, rooms_geojson)
    end_node = find_closest_node_or_room(end_coord, G, rooms_geojson)

    # start_node = find_closest_node(start_coord, G)
    # end_node = find_closest_node(end_coord, G)

    try:
        path = nx.astar_path(G, start_node, end_node, heuristic=lambda a, b: heuristic(a, b, G), weight='weight')
        return path, G
    except nx.NetworkXNoPath:
        return None, "No path found."


def main():
    default_start_coord = (85192.9137, 446861.4036)
    default_end_coord = (85257.853, 446971.300)

    # User input for start and end coordinates
    start_input = input("Please enter the start coordinates (format: x,y) or press Enter to use default: ")
    end_input = input("Please enter the end coordinates (format: x,y) or press Enter to use default: ")

    # Use default coordinates if input is empty
    start_coord = default_start_coord if start_input.strip() == "" else tuple(map(float, start_input.split(',')))
    end_coord = default_end_coord if end_input.strip() == "" else tuple(map(float, end_input.split(',')))

    # Loading the geojson files
    boundary_geojson = json.load(open(BUILDING_EDGE_PATH, 'r', encoding='utf-8'))
    rooms_geojson = json.load(open(ROOMS_PATH, 'r', encoding='utf-8'))

    path, G = routing(start_coord, end_coord, boundary_geojson, rooms_geojson)

    if isinstance(path, list):
        print("Path found:")
        coordinates = []
        for node in path:
            print(f"ID: {node}, Label: {G.nodes[node]['label']}")
            # Access the coordinates directly
            node_coordinates = G.nodes[node]['coordinates']
            x = node_coordinates[0]
            y = node_coordinates[1]
            coordinates.append([x, y])

        # Create a GeoJSON LineString from the coordinates
        linestring_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {},
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::28992"
                }
            }
        }

        # Output the GeoJSON as a string
        linestring_str = json.dumps(linestring_geojson, indent=2)
        print("GeoJSON LineString:")
        print(linestring_str)

        # Save the GeoJSON to a file
        with open(f"{ROUTING_PATH}", "w", encoding='utf-8') as f:
            json.dump(linestring_geojson, f, ensure_ascii=False, indent=2)

        print(f"GeoJSON LineString saved to '{ROUTING_PATH}'.")
        return linestring_geojson
    else:
        print(path)


if __name__ == "__main__":
    main()

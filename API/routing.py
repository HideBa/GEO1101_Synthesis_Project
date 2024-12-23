import json
import os
from math import sqrt
from shapely.geometry import shape, Point
import networkx as nx
import geopandas as gpd


def build_graph(nodes_json_path):
    """
    Build and return a graph G based on the data from the specified GeoJSON file.
    Nodes and edges are added to the graph with Euclidean distances as edge weights.
    """
    nodes = json.load(open(nodes_json_path, "r", encoding="utf-8"))
    G = nx.Graph()

    # Add nodes to the graph
    for feature in nodes["features"]:
        node_id = feature["properties"]["id"]
        label = feature["properties"]["label"]
        coordinates = feature["geometry"]["coordinates"]
        G.add_node(node_id, label=label, coordinates=coordinates)

    # Add edges between nodes based on neighbors and calculate edge weights
    for feature in nodes["features"]:
        node_id = feature["properties"]["id"]
        neighbors = feature["properties"]["neighbors"]
        if neighbors:
            neighbors = [int(n.strip()) for n in neighbors.split(",")]
            for neighbor in neighbors:
                if not G.has_edge(node_id, neighbor):
                    coord1 = G.nodes[node_id]["coordinates"]
                    coord2 = G.nodes[neighbor]["coordinates"]
                    distance = sqrt(
                        (coord1[0] - coord2[0]) ** 2
                        + (coord1[1] - coord2[1]) ** 2
                    )
                    G.add_edge(node_id, neighbor, weight=distance)

    return G


# Heuristic function for A*
def heuristic(a, b, G):
    coord_a = G.nodes[a]["coordinates"]
    coord_b = G.nodes[b]["coordinates"]
    return sqrt(
        (coord_a[0] - coord_b[0]) ** 2 + (coord_a[1] - coord_b[1]) ** 2
    )


def get_room_id(room_name: str, G):
    for node_id, data in G.nodes(data=True):
        if data["label"] == room_name:
            return node_id
    return None


def path_to_linestring(path, G):

    coordinates = []
    for node in path:
        # print(f"ID: {node}, Label: {G.nodes[node]['label']}")
        # Access the coordinates directly
        node_coordinates = G.nodes[node]["coordinates"]
        x = node_coordinates[0]
        y = node_coordinates[1]
        coordinates.append([x, y])

    # Create a GeoJSON LineString from the coordinates
    linestring_geojson = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": {},
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
        },
    }

    return linestring_geojson

# Helper function to update edge weights
def update_edge_weights_for_restricted_rooms(G, floorplan, restricted_rooms, new_weight=float('inf')):
    
    # Filter to get only the polygons for restricted rooms using geopandas
    floorplan_gdf = gpd.GeoDataFrame.from_features(floorplan["features"])
    restricted_gdf = floorplan_gdf[floorplan_gdf['room'].isin(restricted_rooms)]

    print('-' * 60)
    print(f'Setting weights of edges within restricted rooms: {restricted_rooms}')
    # Iterate over nodes and update weights of their connected edges if the node is within a restricted room
    for node_id, node_data in G.nodes(data=True):
        node_point = Point(node_data['coordinates'])  # Assuming 'pos' is (x, y) for each node
        if any(restricted_poly.contains(node_point) for restricted_poly in restricted_gdf.geometry):
            # Update weights of all edges connected to this node
            
            # print(f'\tFound node {node_id}')
            for neighbor in G.neighbors(node_id):
                G[node_id][neighbor]['weight'] = new_weight
                # print(f'\t\tRaise edge weight with node {neighbor} to {new_weight}')


def navigation(
    start: str,
    end: str,
    floorplan_json_path: str,
    nodes_json_path: str,
    route_output_path: str,
    restricted_rooms: list = []
):

    # Loading the floorplan containing the rooms
    floorplan = json.load(open(floorplan_json_path, "r", encoding="utf-8"))


    # build the graph using the nodes json
    Graph = build_graph(nodes_json_path)

    if restricted_rooms:
        update_edge_weights_for_restricted_rooms(Graph, floorplan, restricted_rooms)


    start_node_id = get_room_id(start, Graph)
    end_node_id = get_room_id(end, Graph)

    try:
        path = nx.astar_path(
            Graph,
            start_node_id,
            end_node_id,
            heuristic=lambda a, b: heuristic(a, b, Graph),
            weight="weight",
        )

    except nx.NetworkXNoPath:
        return None, "No path found."

    linestring_str = path_to_linestring(path, Graph)

    # Save the GeoJSON to a file
    with open(f"{route_output_path}", "w", encoding="utf-8") as f:
        json.dump(linestring_str, f, ensure_ascii=False, indent=2)

    folder_and_filename = os.path.join(
        os.path.basename(os.path.dirname(route_output_path)),
        os.path.basename(route_output_path),
    )

    print(f"GeoJSON LineString saved to '{folder_and_filename}'.")





if __name__ == "__main__":

    # building_edge_path = os.path.join("data", "routing", "boundary.geojson")
    nodes_json_path = os.path.join("API", "data", "nodes.geojson")
    floorplan_json_path = os.path.join("API", "data", "floorplan.geojson")
    route_output_path = os.path.join("API", "user_data_cache", "routing.geojson")

    start = "geolab"
    end = "main_entrance"

    #name of polygons that should be restricted while getting routing
    restricted_rooms = []

    navigation(start, end, floorplan_json_path, nodes_json_path, route_output_path, restricted_rooms)

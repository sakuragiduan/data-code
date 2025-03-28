from typing import List
import pandas as pd
import os
import folium
import webbrowser
from data.map import Poly as Poly
from math import radians, cos, sin, asin, sqrt
import numpy as np

from data import data_manager as dm

MAX_LONGITUDE = 104.15
MAX_LATITUDE = 30.7
MIN_LONGITUDE = 104
MIN_LATITUDE = 30.6
CITY_DIVISION = 15

RECTANGLE = [[MIN_LONGITUDE, MIN_LATITUDE], [MIN_LONGITUDE, MAX_LATITUDE], [MAX_LONGITUDE, MAX_LATITUDE],
             [MAX_LONGITUDE, MIN_LATITUDE], [MIN_LONGITUDE, MIN_LATITUDE]]


def is_ray_intersects_segment(poi: List, s_poi: List, e_poi: List) -> bool:
    """

    Args:
        poi: the given point [long, lat]
        s_poi: the start point of an edge
        e_poi: the end point of an edge

    Returns:
        True if there is intersection, else False
    """

    if s_poi[1] == e_poi[1]:
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:
        return False
    if s_poi[0] < poi[0] and e_poi[1] < poi[1]:
        return False

    sec = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])

    if sec < poi[0]:
        return False
    return True


def is_poi_within_poly(poi: List, poly: List) -> bool:
    """

    Args:
        poi: point, [long., lat.]
        poly: polygon: [[x1,y1],[x2,y2],......,[xn,yn],[x1,y1]]

    Returns:
        True if the point is inside the polygon, otherwise False

    Author: Peibo Duan (based on existing work)

    Date: 07/01/2021

    Fun: to justify whether a point is inside a polygon or not

    """
    num_intersections = 0  # the number of intersections
    for i in range(len(poly) - 1):  # [0,len-1]
        s_poi = poly[i]
        e_poi = poly[i + 1]
        if is_ray_intersects_segment(poi, s_poi, e_poi):
            num_intersections += 1

    return True if num_intersections % 2 == 1 else False


def read_map(file_path: str) -> pd.DataFrame:
    """

    Args:
        file_path: file path

    Returns: None

    Author: Peibo Duan

    Date: 07/01/2021

    Fun: get map

    """
    return pd.read_csv(file_path)


def get_study_site_divided_by_hexagon():
    """

    Args: None

    Returns: None

    Author: Peibo Duan

    Date: 07/01/2021

    Fun: get study site based on the given maximal (minimal) lat. and long. (Rectangle).
         The study site is divided into a series of hexagons (based on the original city map)

    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map', 'hexagon_grid_table.csv')
    org_map = read_map(file_path)  # the origial map
    col_names = ['grid_id', 'v_1_long', 'v_1_lat', 'v_2_long', 'v_2_lat', 'v_3_long', 'v_3_lat', 'v_4_long', 'v_4_lat',
                 'v_5_long', 'v_5_lat', 'v_6_long', 'v_6_lat']
    study_site = list()
    for index, row in org_map.iterrows():
        poly = list()
        poly.append([row['v_1_long'], row['v_1_lat']])
        poly.append([row['v_2_long'], row['v_2_lat']])
        poly.append([row['v_3_long'], row['v_3_lat']])
        poly.append([row['v_4_long'], row['v_4_lat']])
        poly.append([row['v_5_long'], row['v_5_lat']])
        poly.append([row['v_6_long'], row['v_6_lat']])
        for point in poly:
            if is_poi_within_poly(point, RECTANGLE):
                study_site.append(row)
                break
    df = pd.DataFrame(study_site, columns=col_names)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map', 'study_site.csv')
    dm.save_data(file_path, df, True)


def study_site_visualization_in_hexagon():
    """

    Args: None

    Returns: None

    Author: Peibo Duan

    Date: 07/01/2021

    Fun: visualize the study site which is divided into a set of hexagons

    """

    # map of chengdu, GCJ-02 system
    chengdu_map = folium.Map(  # based on Chengdu map
        location=[(MAX_LATITUDE + MIN_LATITUDE) / 2, (MAX_LONGITUDE + MIN_LONGITUDE) / 2],
        zoom_start=12,  # 12
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',  # GaoDe
        attr='default')

    # get study site
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map', 'study_site.csv')
    study_site = read_map(file_path)
    for index, row in study_site.iterrows():
        poly = list()
        poly.append([row['v_1_lat'], row['v_1_long']])
        poly.append([row['v_2_lat'], row['v_2_long']])
        poly.append([row['v_3_lat'], row['v_3_long']])
        poly.append([row['v_4_lat'], row['v_4_long']])
        poly.append([row['v_5_lat'], row['v_5_long']])
        poly.append([row['v_6_lat'], row['v_6_long']])
        poly.append([row['v_1_lat'], row['v_1_long']])
        folium.PolyLine(locations=poly).add_to(chengdu_map)

    # visualization
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map', 'study_site.html')
    chengdu_map.save(file_path)
    webbrowser.open(f'{file_path}', 2)


def study_site_visualization_in_rectangle():
    """

    Args: None

    Returns: None

    Author: Peibo Duan

    Date: 07/01/2021

    Fun: visualize the study site which is divided into a set of rectangles

    """
    # map of chengdu, GCJ-02 system
    chengdu_map = folium.Map(
        location=[(MAX_LATITUDE + MIN_LATITUDE) / 2, (MAX_LONGITUDE + MIN_LONGITUDE) / 2],
        zoom_start=12,  # 12
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',  # GaoDe
        attr='default')

    study_site = get_study_site_divided_by_rectangle()
    for row in range(CITY_DIVISION):
        for col in range(CITY_DIVISION):
            region = study_site[row][col]
            poly = list()
            poly.append([region.points['left_top'].lat, region.points['left_top'].long])
            poly.append([region.points['right_top'].lat, region.points['right_top'].long])
            poly.append([region.points['right_bottom'].lat, region.points['right_bottom'].long])
            poly.append([region.points['left_bottom'].lat, region.points['left_bottom'].long])
            poly.append([region.points['left_top'].lat, region.points['left_top'].long])
            folium.PolyLine(locations=poly).add_to(chengdu_map)

    # visualization
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map', 'study_site.html')
    chengdu_map.save(file_path)
    webbrowser.open(f'{file_path}', 2)


def get_study_site_divided_by_rectangle() -> List[List[Poly.Poly]]:
    """

    Args: None

    Returns: the list of rectangles after division

    Author: Peibo Duan

    Date: 07/01/2021

    Fun: divide a study site into a set of rectangles

    """
    lat_step = (MAX_LATITUDE - MIN_LATITUDE) / CITY_DIVISION
    long_step = (MAX_LONGITUDE - MIN_LONGITUDE) / CITY_DIVISION
    study_site = list()
    df_study_site = list()
    grid_id = 0
    col_names = ['grid_id', 'v_1_long', 'v_1_lat', 'v_2_long', 'v_2_lat', 'v_3_long', 'v_3_lat', 'v_4_long', 'v_4_lat']
    for row in range(CITY_DIVISION):
        bottom_lat = MIN_LATITUDE + lat_step * row
        top_lat = MIN_LATITUDE + lat_step * (row + 1)
        row_study_site = list()
        for col in range(CITY_DIVISION):
            left_long = MIN_LONGITUDE + long_step * col
            right_long = MIN_LONGITUDE + (col + 1) * long_step
            left_bottom = Poly.Point(bottom_lat, left_long)
            right_bottom = Poly.Point(bottom_lat, right_long)
            right_top = Poly.Point(top_lat, right_long)
            left_top = Poly.Point(top_lat, left_long)
            left_edge = Poly.Edge(left_top, left_bottom)
            bottom_edge = Poly.Edge(left_bottom, right_bottom)
            right_edge = Poly.Edge(right_bottom, right_top)
            top_edge = Poly.Edge(right_top, left_top)
            grid = Poly.Poly(str(grid_id),
                             {'left_top': left_top, 'left_bottom': left_bottom, 'right_bottom': right_bottom,
                              'right_top': right_top}, [left_edge, bottom_edge, right_edge, top_edge])
            row_study_site.append(grid)
            df_study_site.append(
                [grid_id, left_long, bottom_lat, right_long, bottom_lat, right_long, top_lat, left_long, top_lat])
            grid_id = grid_id + 1
        study_site.append(row_study_site)
    df = pd.DataFrame(df_study_site, columns=col_names)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'map', 'study_site.csv')
    dm.save_data(file_path, df, True)
    return study_site


def get_grid_id_for_poly(poi: List) -> str:
    """

    Args:
        poi: point, [long, lat]

    Returns:
        grid id

    Author: Peibo Duan

    Date: 08/01/2021

    Fun: get grid id where the given point is in

    """
    lat_step = (MAX_LATITUDE - MIN_LATITUDE) / CITY_DIVISION
    long_step = (MAX_LONGITUDE - MIN_LONGITUDE) / CITY_DIVISION
    for row in range(CITY_DIVISION):
        bottom_lat = MIN_LATITUDE + lat_step * row
        top_lat = MIN_LATITUDE + lat_step * (row + 1)
        if poi[1] < bottom_lat or poi[1] > top_lat:
            continue
        for col in range(CITY_DIVISION):
            left_long = MIN_LONGITUDE + long_step * col
            right_long = MIN_LONGITUDE + (col + 1) * long_step
            if poi[0] < left_long or poi[0] > right_long:
                continue
            else:
                return str(row * CITY_DIVISION + col)


def get_adjacent_matrix() -> np:
    """

    Args: None

    Returns: adjacent matrix

    Author: Peibo Duan

    Date: 08/01/2021

    Fun: get adjacent matrix

    """
    num_grids = CITY_DIVISION * CITY_DIVISION
    adjacent_matrix = np.zeros([num_grids, num_grids], dtype=int)

    for row in range(CITY_DIVISION):
        for col in range(CITY_DIVISION):
            grid_id = row * CITY_DIVISION + col
            for sur_row in range(row - 1, row + 2):
                if sur_row < 0 or sur_row > CITY_DIVISION - 1:
                    continue
                for sur_col in range(col - 1, col + 2):
                    if sur_col < 0 or sur_col > CITY_DIVISION - 1:
                        continue
                    sur_grid_id = sur_row * CITY_DIVISION + sur_col
                    if sur_grid_id != grid_id:
                        adjacent_matrix[grid_id][sur_grid_id] = 1

    return adjacent_matrix

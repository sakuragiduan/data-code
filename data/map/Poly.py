from typing import List, Dict


class Point(object):
    def __init__(self, lat: float, long: float):
        self.lat = lat
        self.long = long


class Edge(object):
    def __init__(self, s_point: Point, e_point: Point):
        self.s_point = s_point
        self.e_point = e_point


class Poly(object):
    def __init__(self, poly_id: str, points: Dict[str, Point], edges: List[Edge]):
        self.poly_id = poly_id
        self.points = points
        self.edges = edges

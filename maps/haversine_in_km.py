#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Haversine distance 

@author: Christian Ruf

"""
import math
from dataclasses import dataclass


@dataclass
class Loc:
    lat: float
    lon: float


def distance_haversine(loc1: Loc, loc2: Loc) -> float:
    """ Return the distance in km. """
    if loc1 == loc2:
        return 0.0

    v = (math.sin(loc1.lat * math.pi / 180) * math.sin(loc2.lat * math.pi / 180)
         + math.cos(loc1.lat * math.pi / 180) * math.cos(loc2.lat * math.pi / 180)
         * math.cos(loc2.lon * math.pi / 180 - loc1.lon * math.pi / 180))

    # take care of floating point imprecision
    if 1.0 < v < 1.01:
        v = 1.0
    elif -1.01 < v < -1.0:
        v = -1.0

    if v < -1 or v > 1:
        raise Exception('Error in distance for %s and %s' % (loc1, loc2))

    return 1.852001 * 3443.8985 * math.acos(v)


def distance_haversine_miles(loc1: Loc, loc2: Loc) -> float:
    return distance_haversine(loc1, loc2) * 0.6213712

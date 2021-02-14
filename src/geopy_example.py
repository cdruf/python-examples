#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Christian Ruf
"""

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

geolocator = Nominatim(user_agent="my_app")

munich = geolocator.geocode("Munich")
print(munich.address)
print(munich.latitude, munich.longitude)
#print(munich.raw)

paris = geolocator.geocode("Paris") 
print(paris.address)
print(paris.latitude, paris.longitude)


print(geodesic((munich.latitude, munich.longitude), 
               (paris.latitude, paris.longitude)).km)

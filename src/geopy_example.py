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
print(munich.raw)
print('\n\n')

paris = geolocator.geocode("Paris")
print(paris.address)
print(paris.latitude, paris.longitude)
print('\n\n')

print(geodesic((munich.latitude, munich.longitude),
               (paris.latitude, paris.longitude)).km)

chicago = geolocator.geocode("Chicago")
print(chicago.address)
print(chicago.latitude, chicago.longitude)
print('\n\n')

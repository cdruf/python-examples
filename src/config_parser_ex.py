# -*- coding: utf-8 -*-
"""
This example writes and loads a config file.
"""

import configparser

# %%
# Write

config = configparser.ConfigParser()
config['DEFAULT'] = {'Attribute 1': '42',
                     'Attribute 2': '43'}
config['A'] = {}
config['A']['User'] = 'xyz'

config['C'] = {}
topsecret = config['C']
topsecret['C.1'] = '22'
topsecret['C.2'] = '23'

config['DEFAULT']['Attribute 3'] = 'yes'

with open('config.ini', 'w') as configfile:
    config.write(configfile)

# %%
# Load

conf = configparser.ConfigParser()
conf.read('config.ini')
conf.sections()
print(config['DEFAULT']['Attribute 1'])


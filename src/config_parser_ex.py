# -*- coding: utf-8 -*-
"""
This example writes and loads a config file.
"""

import configparser


def create_and_write_example_config():
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'Attribute 1': '42',
                         'Attribute 2': '43'}
    config['A'] = {}
    config['A']['User'] = 'xyz'

    config['C'] = {}
    top_secret = config['C']
    top_secret['C.1'] = '22'
    top_secret['C.2'] = '23'

    config['DEFAULT']['Attribute 3'] = 'yes'

    with open('config.ini', 'w') as f:
        config.write(f)


def load_example_config():
    conf = configparser.ConfigParser()
    conf.read('config.ini')
    conf.sections()
    print(conf['DEFAULT']['Attribute 1'])

if __name__ == '__main__':
    create_and_write_example_config()
    load_example_config()

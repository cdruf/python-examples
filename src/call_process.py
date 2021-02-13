#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:51:58 2020

"""
import subprocess


#%%
# with the run function (recommended)

subprocess.run(['ls', '-l'])

subprocess.run(['java',  '-jar', 'process_java_example.jar', 'arg0'])

subprocess.run(['java',  '-jar', 'process_java_example.jar', 'arg0'], 
               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


#%%
# with Popen 

process = subprocess.Popen(['java',  '-jar', 'process_java_example.jar', 'arg0'])
process.communicate()

#%%
# there are more ways to do it
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess

"""
This example script calls various external processes.

* ls
* A Java-process with arguments

"""
print(os.getcwd())
jar_path = 'resources/process_java_example.jar'

# %%

"""
# With the run function (recommended)
"""

subprocess.run(['ls', '-l'])

subprocess.run(['java', '-jar', jar_path, 'arg0'])

subprocess.run(['java', '-jar', jar_path, 'arg0'],
               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# %%

"""
# With Popen 
"""

process = subprocess.Popen(['java', '-jar', jar_path, 'arg0'])
process.communicate()

# there are more ways to do it

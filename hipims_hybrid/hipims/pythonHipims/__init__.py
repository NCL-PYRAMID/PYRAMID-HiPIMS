import os
__version__ = '1.0.0'

print("         Welcome to the HiPIMS ", __version__)

print("""
    @author: Jiaheng Zhao
    @license: (C) Copyright 2020-2025. 2025~ Apache Licence 2.0
    @contact: j.zhao@lboro.ac.uk 
    @software: hipims_torch
    @file: __init__.py
    @time: 07.01.2020
      """)

dir_path = os.path.dirname(os.path.realpath(__file__))

f = open(os.path.join(dir_path, 'banner.txt'), 'r')
file_contents = f.read()
print(file_contents)
f.close()
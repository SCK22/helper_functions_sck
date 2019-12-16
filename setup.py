import os
import sys
import glob
import subprocess
import sysconfig
from setuptools import setup, find_packages

# reference : https://docs.python.org/2/distutils/setupscript.htm
# This file builds a package of the code for the simulator

print(sysconfig.get_paths()["purelib"]) # location from which the current version of python is picking up packages
print(os.path.join(os.getcwd())) # current path
print("getting packages--------------------")
# print(find_packages(where = os.getcwd()))
packages = find_packages()
# packages = find_packages("src")
print(packages)
setup(name="helperfunctionsml",
            version="0.0.1",
            package_dir={"": os.getcwd()}, 
            packages=["ML", "Plotting"],
    author = "SCK",
    author_email = "chaithanyakumar.ds@gmail.com",
    description = "Helper functions for ml workflow",
    long_description = "Helper functions for ml workflow",
    url = "https://github.com/sck22",
    license = "MIT",
    python_requires='>=3.6')
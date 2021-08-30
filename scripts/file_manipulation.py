import os


# Find current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Use current location
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Go up one directory and enter another directory
os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'files_path'))

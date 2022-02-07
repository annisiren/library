import os

##################
# Open file
with open(file_name, 'w') as f: # 'w', 'r', 'wb', 'a'
    f.readline()
f.close()


# Save object to file
import pickle
with open(file_name, 'wb') as f:
    pickle.dump(text, f, pickle.HIGHEST_PROTOCOL)

##################
# Doesn't use regular codex
import codecs
with codecs.open(file_name, 'w', encoding='utf-8') as f:
    f.write(text)

# Find current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Use current location
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Go up one directory and enter another directory
os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'files_path'))

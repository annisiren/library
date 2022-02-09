##################
# DIRECTORIES
import os

os.getcwd() #directories
os.listdir() #current directory
os.listdir("C:\\Users") #directory passed


##################
# MOVE FILES
import shutil

# shutil.move('practice.txt','C:\\Users\\Marcial')
# shutil.move('C:\\Users\\Marcial\practice.txt',os.getcwd())

##################
# REMOVE FILES
# os.unlink(path) which deletes a file at the path your provide
# os.rmdir(path) which deletes a folder (folder must be empty) at the path your provide
# shutil.rmtree(path) this is the most dangerous, as it will remove all files and folders contained in the path.

# pip install send2trash -- to install send2trash

# send2trash.send2trash('practice.txt')


##################
# WALKING THROUGH DIRECTORY

os.getcwd()
os.listdir()
# for folder , sub_folders , files in os.walk("folder_name"):
#
#     print("Currently looking at folder: "+ folder)
#     print('\n')
#     print("THE SUBFOLDERS ARE: ")
#     for sub_fold in sub_folders:
#         print("\t Subfolder: "+sub_fold )
#
#     print('\n')
#
#     print("THE FILES ARE: ")
#     for f in files:
#         print("\t File: "+f)
#     print('\n')

    # Now look at subfolders

##################
# UNZIPPING AND ZIPPING FILES

f = open("new_file.txt",'w+')
f.write("Here is some text")
f.close()

import zipfile
comp_file = zipfile.ZipFile('comp_file.zip','w')
comp_file.write("new_file.txt",compress_type=zipfile.ZIP_DEFLATED)
comp_file.write('new_file2.txt',compress_type=zipfile.ZIP_DEFLATED)
comp_file.close()

zip_obj = zipfile.ZipFile('comp_file.zip','r')
zip_obj.extractall("extracted_content")

import shutil

# Creating a zip archive
output_filename = 'example'
# Just fill in the output_filename and the directory to zip
# Note this won't run as is because the variable are undefined
shutil.make_archive(output_filename,'zip',directory_to_zip)

# Extracting a zip archive
# Notice how the parameter/argument order is slightly different here
shutil.unpack_archive(output_filename,dir_for_extract_result,'zip')


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

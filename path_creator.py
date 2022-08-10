from os import listdir
from os.path import isfile, join

""""
This module contains some helpers to retrieve files from the system
"""


def path(file_name: str):
    return "./data/" + file_name


def get_file_names(directory: str, ends: str):
    """"
    Returns all file names from the directory given that ends with the string given
    """
    return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and f.endswith(ends)]

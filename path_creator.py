import pandas as pd
from os import listdir
from os.path import isfile, join


def path(file_name: str):
    return r"C:\Users\mhrou\Desktop\Orkg\\" + file_name


def get_file_names(directory: str):
    return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and f.endswith(".json")]


if __name__ == '__main__':
    pass




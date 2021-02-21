import os

def create_path(name=""):
    if not os.path.exists(name):
        os.makedirs(name)  
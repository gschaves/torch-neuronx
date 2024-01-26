from setuptools import setup, find_packages

VERSION_FILE = "version.txt"
INIT_FILE = "torch_neuronx/__init__.py"

def get_version():
    with open(VERSION_FILE, encoding="utf-8") as f:
        ver = f.readline()

    if ver.startswith("for"):
        with open(INIT_FILE, encoding="utf-8") as f:
            for line in f.readlines():
                if line.startswith("__version__"):
                    ver = line.split()[-1].strip('"') + "+master"

    return ver

setup(
    name='torch_neuronx',
    version=get_version(),
    packages=find_packages(),
    include_package_data=True
)

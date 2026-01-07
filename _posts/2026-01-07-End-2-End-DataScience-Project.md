# Agenda:
**1- Setup the github repository**
```
a) New environment
b) Create setup.py
c) requirements.txt
```
First create a folder for your project such as **End-2-End-ML-Project**, then open that folder in vscode and from vscode itself, open a terminal. When you do that the conda environment and all the required packages will be created inside your project folder.

The environment can be created like this:
```
conda create -p venv python=3.8 -y
```

And this is a path conda environment. meaning that to activate it, you do:
```
conda activate ./venv # or the complete path to the environment folder
```

- Then go to your github repo and open a new repository and name it something like : **End-2-End-ML-Project**. Now try to create a **README.md** file inside you project folder and push it to your github and make sure you see it there and also in your project folder inside vscode.

<img width="303" height="207" alt="image" src="https://github.com/user-attachments/assets/533cb0bb-ec7e-41ac-9a67-e1eabb1d14b2" />

- One more thing that I ususally try to do by hand but realized you can do it mych easier with github is to create .gitignore file. Just go to yur repo and add a file. name it as **.gitignore** and then just below look for a tab that you can use to change the language. There you can find templates of gitignore for python language. And boom! it will create for you a template for all the files that should be ignored during git commands. goo ahead and commit changes and you will have git ignore setup.

- Now inside your project folder, create requirements.txt and setup.py.
setup.py is very useful becasue it is going to make you project as a package that you can install anywhere and use it as python library. inside your setup.py write the follwoing line to name and version your package and also read all the requirements for your package:

```
from setuptools import find_namespace_packages, setup
from typing import List

def set_requirements(file_path:str) ->List[str]:
    ''' This function will return the list of all requirements'''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
    return requirements


setup(
name="End-2-End-ML-project",
version="0.0.1",
author="Fahimeh Baftizadeh",
author_email="fahimeh.baftizadeh@gmail.com",
packages=find_namespace_packages(),
install_requires=set_requirements('requirements.txt') # all the required libraries for your package
)
```

now inside your requirements.txt file, write the name of each python library you are going to be using and at the end in the final line put a ```-e .`` in order to make the setup.py runs everytime you open your project.

example of the requirement.txt file:
```
pandas
numpy
seaborn
-e .
```

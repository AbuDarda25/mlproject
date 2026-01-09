## Agenda 

1. Set up Github  {Repository} {Repo}

* a) New Environment
* b) setup.py
* c) Requiremnets.txt
* d) Src folder and build the Packages



1. **first step** 
   Create New Git Repo

2\. **Create new Environment** 

* open anaconda prompt
* In anaconda prompt - go to that folder in prompt and generate code . (i.e vs code instance)

-(base) C:\\Users\\abuda>cd D:\\Krish\_DataScience\\MLProject

-(base) C:\\Users\\abuda>D:

-(base) D:\\Krish\_DataScience\\MLProject>code . (vs code open hoga)

#### **In VSCode** 

##### Inside mlproject folder 

* create a venv environment (so that whatever packages i installed that all get create over here) by using these commands in new terminal 

conda create -p venv python==3.8 -y 

conda activate venv/

* after this vend folder will be created in mlproject folder with all packages and stuffs

##### **SYNC With git ,** Create clone this entire repo and  

* Go to VS Code terminal and Run these commands

echo "# mlproject" >> README.md

git init

git add README.md

git commit -m "first commit"

git branch -M main

git remote add origin https://github.com/AbuDarda25/mlproject.git

git push -u origin main

git pull

* Alternative way

git remote add origin https://github.com/AbuDarda25/mlproject.git

git branch -M main

git push -u origin main

git pull



* ###### **Create Git ignore** in GitHub inside mlproject folder 
* and file name will be { .gitignore } why  bcs some of the files that need not be committed in the GitHub, that all will get removed.





* At Last do **git pull** in vs terminal , so thatv all updation will happen



**2. Create setup.py file** ( building our application as a package itself , like anyone can install it , use it  **)**

* there r a lot of packages like seaborn , for this package to install (pip install seaborn in vs ) internally a setup.py is required , (u can search for python setu.py)
* This setup.py will be responsible in creating my machine learning **application** as a package. and from there anybody can installation and anybody can also use it.
* With the help of setup.py i will be able to **build my entire machine learning application** as a package and even **deploy** in py py
* The setup script is the centre of all activity in building , distribution and installing modules using the Distutils, The main purpose of the setup is to describe your module distribution to the Distutils, so that the various commands that operate on your modules do the right thing .
* **In Setup write these basics code , later it will be updated** 

from setuptools import find\_packages,setup



setup(

name = 'MLProject',

version='0.0.1',

author='AbuDarda',

author\_email='abudarda0025@gmail.com',

package=find\_packages(),

install\_requires=\['pandas','numpy','seaborn','matplotlib.pyplot']

)

* This setup.py , how it will be able to find out how many packages are there and all , for this create a New Folder as **src** 
* inside **src** folder create **\_\_init\_\_.py ,** so whenever find\_packages will run in setup.py , it will go to \_\_init\_\_.pu , or basically it will consider src folder as a packages itself and then it will try to build this and we can import this anywhere.
* **Entire project development will happen inside** this src folder or , \_\_init\_\_.py folder



**3. Create Requirements.txt file (** requirements.txt will have all the packages that i really need to uh , install while I'm actually implementing my project **)**





* but in setup.py we may need many paclkage , libraries , we won't read each name is this list of install\_requires=\['pandas','numpy' etc..]
* So, this change is made 
* in requirement.tct file , write name of all libraries (pnadas, numpy , seaborn , matplotlib.pyplot etc .. and '-e .' at last line and made some code change in setup.py like this 
* NEW setup.py code 

*from setuptools import find\_packages,setup*

*from typing import List*



*HYPEN\_E\_DOT = '-e .'*



*def get\_requirements(file\_path:str)->List\[str]:*

    *'''*

    *this function will return the list of requirements*

    *'''*

    *requirements=\[]*

    *with open('requirement.txt') as file\_obj:*

        *requirements = file\_obj.readlines()*

        *requirements = \[req.replace("\\n","") for req in requirements]*



        *if HYPEN\_E\_DOT in requirements :*

            *requirements.remove(HYPEN\_E\_DOT)*

    

    *return requirements*



*setup(*

*name = 'MLProject',*

*version='0.0.1',*

*author='AbuDarda',*

*author\_email='abudarda0025@gmail.com',*

*package=find\_packages(),*

*# install\_requires=\['pandas','numpy','seaborn','matplotlib.pyplot']*

*install\_requires=get\_requirements('requirement.txt')*



*)*



* and run these code , so manyb files and libraries and package will get installed and will be availbale in src folder , you can see'














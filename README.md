This is my repository for working through the exercises in the LinkedIn Learning course, [Python: Working with Predictive Analytics by Isil Berkun](https://www.linkedin.com/learning/python-working-with-predictive-analytics/)

However, instead of using Anaconda, I used Visual Studio Code with the Python extension, following the YouTube setup guide by [Aditya Thakur](https://www.youtube.com/watch?v=ThU13tikHQw)


## Create a Virtual Environment to allow import of Python modules

Create a virtual terminal by opening Terminal from Visual Studio Code, using the following command:

```
python -m venv yourpythonprojectfolder\venv
```

For example:
```
python -m venv C:\Project\tutorial-linkedin-learning-python-working-with-predictive-analytics\venv
```

The venv virtual environment will appear as a folder under Explore.

Open the terminal afresh, and your directory should be prefixed with (venv) to indicate you are in the virtual environment.


### Reopening the Virtual Environment

PowerShell will by default block running potentially harmful scripts

So you should be using cmd terminal from within Visual Studio Code, rather than the PowerShell terminal.

After closing Visual Studio Code, the environment will be stopped. To activate it again, use the following command

```
yourvenvname\Scripts\activate
```

For example:
```
venv\Scripts\activate
```

Alternatively, use Ctrl + Shift + 


### Add Virtual Environment to .gitignore
The modules take up a lot of space, so create a new .gitignore file in the directory and add venv (or whatever your virtual environment’s folder name is, inside the file).


### Deactivate and delete Virtual Environment

```
deactivate
```

Assuming the virtual environment was named as folder venv, otherwise replace with the folder name

```
rm -r venv
```


### Import Python modules

After performing the steps above (minus deactivting then deleting the environment), you can now install Python modules.

```
pip install pythonmodulename
```

For example:
```
pip install matplotlib
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
```

Once done, click Run to render your Python script.

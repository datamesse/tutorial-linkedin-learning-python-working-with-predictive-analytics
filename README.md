This is my repository for working through the exercises in the LinkedIn Learning course, [Python: Working with Predictive Analytics by Isil Berkun](https://www.linkedin.com/learning/python-working-with-predictive-analytics/)

However, instead of using Anaconda, I used Visual Studio Code with the Python extension, following the YouTube setup guide by [Aditya Thakur](https://www.youtube.com/watch?v=ThU13tikHQw)

## Create a Virtual Terminal to allow import of Python modules

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

### Import Python modules

After performing the steps above, you can now install Python modules

```
pip install pythonmodulename
```

For example:
```
pip install matplotlib
```

Once done, click Run to render your Python script.

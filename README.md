# Project euler
Solutions for project euler problems, solved purely in Python. Helper.py contains several functions that are reusable between more than one 
problem and in some cases utilities only used once. 

The `atexit.register` function is used to automatically time any calls made in `__main__` 
and print the time with pretty formatting
Individual functions can also be wrapped with the `TimeIt` class to time just one function call

import sys
sys.path.append('../rhapsody')
from rhapsody import *

# EXAMPLE
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5

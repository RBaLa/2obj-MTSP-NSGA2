import numpy as np
import matplotlib.pyplot as plt
import math
import copy

SEED = 123456789

def updateSeed():
    global SEED
    large = 2147483647;
    k = int(SEED/127773);
    SEED = 16807*(SEED-k*127773)-k*2836;
    if SEED<=0:
        SEED += large;

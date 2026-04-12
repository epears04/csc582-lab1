import sys
import ast
import pandas as pd
from sklearn.model_selection import train_test_split

# if using supervised learning what to do with movies with only 1 director → 
# allowed to subsample movies and only select movies with directors with at least (4) directors 

def main():
    # is input text optional?
    if len(sys.argv) != 2: 
        print("Usage: robotproducer.py [input.txt]")
        with open(sys.argv[1], "r") as f:
            text = f.read()

if __name__ == "__main__":
    main()
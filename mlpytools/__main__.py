import numpy as np
from mlpytools.LinearRegression import LinearRegression

def main():
    print("Hello World")
    hi = LinearRegression([1,2,3],[4,5,6])
    print(hi.fit())


def stop():
    pass


if __name__ == "__main__":
    main()
import numpy as np
from mlpytools.LinearRegression import LinearRegression

def main():
    print("Hello World")
    hi = LinearRegression([9,12,555],[4,5,6])
    print(hi.predict(4))


def stop():
    pass


if __name__ == "__main__":
    main()
import numpy
import pandas as pd


class Calculation:
    arg1 = None
    arg2 = None
    operator = None
    result = None
    def __init__(self, arg1, arg2, operator):
        self.arg1 = arg1
        self.arg2 = arg2
        self.operator = operator

    def calculate(self, df):
        if self.operator == '*':
            df[self.arg1 + self.operator + self.arg2] = df[self.arg1].multiply(df[self.arg2])

        elif self.operator == '/':
            df[self.arg1 + self.operator + self.arg2] = df[self.arg1].div(df[self.arg2])

def test_basic_calculate():
    df = pd.DataFrame(numpy.array([[1, 2, 3, 4], [1, 2, 3, 4]]).T, columns=['a', 'b'])
    calc = Calculation('a', 'b', '*')
    print(df)
    calc.calculate(df)
    print(df)
    calc = Calculation('a', 'b', '/')
    calc.calculate(df)
    print(df)

if __name__ == '__main__':
    test_basic_calculate()
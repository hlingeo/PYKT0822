y = 1000000# you can change the y  =  10, 100, 1000, then you can see different error with it.


def calculate(x):
    for i in range(0, 1000000):
        x += 0.0000001
    x -= 0.1
    return x


print(f"result={calculate(y):.6f}")
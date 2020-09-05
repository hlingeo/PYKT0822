import random


def calculateBMI(h, w):
    b = w / ((h / 100) ** 2)
    if b < 18.5:
        return 'thin'
    elif b < 25:
        return 'normal'
    else:
        return 'fat'


with open('data/bmi.csv', 'w', encoding='UTF-8') as file1:
    file1.write('height,weight,label\n')
    category = {'thin': 0, 'normal': 0, 'fat': 0}
    for i in range(300000):
        currentHeight = random.randint(110, 200)
        currentWeight = random.randint(40, 60)
        label = calculateBMI(currentHeight, currentWeight)
        category[label] += 1
        file1.write("%d,%d,%s\n" % (currentHeight, currentWeight, label))
    print("distribution=%s" % str(category))
    print("generate OK")

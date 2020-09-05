from tensorflow.keras import utils

origs = [4, 7, 13, 5, 8]
NUM_DIGITS = 20

for o in origs:
    converted = utils.to_categorical(o, NUM_DIGITS)
    print(f"{o}==>{converted}")
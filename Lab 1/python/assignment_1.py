import monkdata as m
from dtree import entropy 


def main():
    training_set1 = m.monk1test
    training_set2 = m.monk2test
    training_set3 = m.monk3test
    print("Entropy of training dataset 1: " + str(entropy(training_set1)))
    print("Entropy of training dataset 2: " + str(entropy(training_set2)))
    print("Entropy of training dataset 3: " + str(entropy(training_set3)))


if __name__ == '__main__':
    main()
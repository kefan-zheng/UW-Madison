import re
import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    # Implementing vectors e,s as lists (arrays) of length 26
    # with p[0] being the probability of 'A' and so on
    e = [0] * 26
    s = [0] * 26

    with open('e.txt', encoding='utf-8') as f:
        for line in f:
            # strip: removes the newline character
            # split: split the string on space character
            char, prob = line.strip().split(" ")
            # ord('E') gives the ASCII (integer) value of character 'E'
            # we then subtract it from 'A' to give array index
            # This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char) - ord('A')] = float(prob)
    f.close()

    with open('s.txt', encoding='utf-8') as f:
        for line in f:
            char, prob = line.strip().split(" ")
            s[ord(char) - ord('A')] = float(prob)
    f.close()

    return (e, s)


def shred(filename):
    # Using a dictionary here. You may change this to any data structure of
    # your choice such as lists (X=[]) etc. for the assignment
    X = dict()
    for char in range(ord('A'), ord('Z') + 1):
        X[chr(char)] = 0
    with open(filename, encoding='utf-8') as f:
        # TODO: add your code here
        content = f.read()
        content = "".join(re.findall(r'[a-zA-Z]', content))
        for char in content:
            if 97 <= ord(char) <= 122:
                char = chr(ord(char) - 32)
            X[char] += 1

    return X


# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
if __name__ == "__main__":
    """
    Q1
    """
    X = shred("letter.txt")
    print("Q1")
    for key, value in X.items():
        print(key, value)

    """
    Q2
    """
    e, s = get_parameter_vectors()
    print("Q2")
    print("{:.4f}".format(X['A']*math.log(e[0])))
    print("{:.4f}".format(X['A']*math.log(s[0])))

    """
    Q3
    """
    P_YE = 0.6
    P_YS = 0.4
    F_E = math.log(P_YE) + sum([x_i*math.log(p_i) for p_i, x_i in zip(e, X.values())])
    F_S = math.log(P_YS) + sum([x_i*math.log(p_i) for p_i, x_i in zip(s, X.values())])
    print("Q3")
    print("{:.4f}".format(F_E))
    print("{:.4f}".format(F_S))

    """
    Q4
    """
    diff = F_S - F_E
    P_YE_X = 0
    if diff >= 100:
        P_YE_X = 0
    elif diff <= -100:
        P_YE_X = 1
    else:
        P_YE_X = 1/(1+math.exp(diff))

    print("Q4")
    print("{:.4f}".format(P_YE_X))



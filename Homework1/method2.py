import numpy as np

COEFF = 'coeff'
TEXT_FILE = 'Ecuatii.txt'


def getCoefficients(text_path):
    fd = open(text_path, 'r')
    line = fd.readline()
    vars = ['x', 'y', 'z']
    chars = {ch: {'exists': False, COEFF: []} for ch in vars}
    r = []
    while line:
        isDigit, isNegative, isEqual, coefficient, rez, rezNegative = (False, False, False, 0, 0, False)
        for ch in line:
            if ch == '-':
                isNegative = True
                if isEqual: rezNegative = True

            if ch.isdigit():
                isDigit = True
                coefficient = coefficient * 10 + int(ch)
                if isNegative:
                    coefficient *= -1
                    isNegative = False
                if isEqual:
                    rez = rez * 10 + int(ch)
                    if rezNegative:
                        rez *= -1
                        rezNegative = False

            if ch in vars:
                chars[ch]["exists"] = True
                isDigit, isNegative = checkCoefficients(isDigit, isNegative, chars[ch][COEFF], coefficient)
                coefficient = 0

            if ch == '=':
                isEqual = True

        for ch in vars:
            if not chars[ch]['exists']: chars[ch][COEFF] += [0]
            chars[ch]['exists'] = False
        r += [[rez]]

        line = fd.readline()

    return chars, r


def checkCoefficients(isDigit, isNegative, v, coefficient):
    if isDigit:
        isDigit = False
        v += [coefficient]
    elif isNegative:
        v += [-1]
        isNegative = False
    else:
        v += [1]

    return isDigit, isNegative


def transformToMatrix(chars, r):
    colls = len(chars['x'][COEFF])
    A, B = ([], [])
    for coll in range(colls):
        row = []
        for index, char in enumerate(chars):
            row += [chars[char][COEFF][coll]]
            if len(B) != len(chars):
                B += [r[index]]
        A += [row]
    return A, B


def displayAnswer(X, chars):
    for index, char in enumerate(chars):
        print("%s = %f" % (char, X[index][0]))


def main():
    chars, r = getCoefficients(TEXT_FILE)
    A, B = transformToMatrix(chars, r)
    A = np.array([np.array(row) for row in A])
    B = np.array([np.array(row) for row in B])
    det = np.linalg.det(A)
    if det == 0:
        print('Matricea nu are inversa')
        return
    T = A.T
    inv = np.linalg.inv(A)
    X = np.dot(inv, B)
    print(X)

    print(np.linalg.solve(A, B))

    displayAnswer(X, chars)


main()

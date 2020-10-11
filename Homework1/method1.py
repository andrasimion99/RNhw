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


def getMinor(A, i, j):
    return [row[:j] + row[j + 1:] for row in A[:i] + A[i + 1:]]


def determinant(A):
    print(A)
    d = 0
    rows = len(A)
    colls = len(A[0])
    if rows == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    for j in range(colls):
        d += A[0][j] * (-1) ** j * determinant(getMinor(A, 0, j))
    return d


def transpus(chars):
    t = []
    for char in chars:
        t.append(chars[char][COEFF])
    return t


def adjugate(A):
    rows = len(A)
    colls = len(A[0])
    return [[(-1) ** (i + j) * determinant(getMinor(A, i, j)) for j in range(colls)] for i in range(rows)]


def inverse(AS, d):
    return [[AS[i][j] / d for j in range(len(AS[0]))] for i in range(len(AS))]


def solveEquation(inv, B):
    X = []
    for i in range(len(inv)):
        row = []
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(inv[0])):
                sum += inv[i][k] * B[k][j]
            row.append(sum)
        X.append(row)
    return X


def displayAnswer(X, chars):
    for index, char in enumerate(chars):
        print("%s = %f" % (char, X[index][0]))


def main():
    chars, r = getCoefficients(TEXT_FILE)
    A, B = transformToMatrix(chars, r)
    det = determinant(A)
    if det == 0:
        print('Matricea nu are inversa')
        return
    T = transpus(chars)
    AS = adjugate(T)
    inv = inverse(AS, det)
    X = solveEquation(inv, B)
    print(X)
    displayAnswer(X, chars)


main()

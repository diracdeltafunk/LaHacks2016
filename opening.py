import re
import numpy

class Tree(object):
    "Generic tree node."
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def gametree(s):
    output = Tree()
    firstparen = 1
    rootlabel = ""
    while firstparen < len(s) and s[firstparen] != "(" and s[firstparen] != ";":
        rootlabel += s[firstparen]
        firstparen += 1
#    print(rootlabel)
    output.name = rootlabel
    x = 0
    stringy = ""
    names = list()
    for i in s:
        if i == ')':
            x -= 1
        if x > 0:
            stringy += i
        if x == 0:
            if stringy != "":
                names.append(stringy)
            stringy = ""
        if i == '(':
            x += 1
    for name_i in names:
        output.add_child(gametree(name_i))
    return output

with open('fuseki.txt', 'r') as file:
    simplified = re.sub('N\[.*?\]|\n|\r', '', file.read())

#print(simplified)
opening_moves = gametree(simplified)


def boardsum0(board):
    n = 0
    for i in range(19):
        for j in range(19):
            n += board[i][j][0]
    return n

def boardsum1(board):
    n = 0
    for i in range(19):
        for j in range(19):
            n += board[i][j][1]
    return n

def boardsum(board):
    return boardsum0(board) + boardsum1(board)

diff = 0

def oriented_move(board, diff):
    nomove = True
    node = opening_moves
    check_tree = True
    depth = 0
   # print(node.name)
    while check_tree:
        if depth >= boardsum(board):
            if len(node.children) != 0:
                return True, [ord(node.children[0].name[2]) - ord('a'),  ord(node.children[0].name[3]) - ord('a')]
            else:
                return False, [0, 0]
        if len(node.children) != 0:
            flag = False
            for child in node.children:
                if board[ord(child.name[2]) - ord('a')][ord(child.name[3]) - ord('a')][(depth + diff) % 2] == 1:
                    node = child
                    depth += 1
                    flag = True
                    break
            if not flag:
                check_tree = False
        else:
            check_tree = False

    if nomove:
        return False, [0, 0]

def make_move(board):
    b0 = board
    b1 = b0[::-1]
    b2 = numpy.transpose(b1, (1, 0, 2))
    b3 = b2[::-1]
    b4 = numpy.transpose(b3, (1, 0, 2))
    b5 = b4[::-1]
    b6 = numpy.transpose(b5, (1, 0, 2))
    b7 = b6[::-1]
    dihedral_group = [b0, b1, b2, b3, b4, b5, b6, b7]
    move_pending = True
    i = 0
    if boardsum0(b0) < boardsum1(b0):
        diff = 1
    else:
        diff = 0
    while move_pending and i < 8:
        if oriented_move(dihedral_group[i], diff)[0]:
            move_pending = False
            tuple = oriented_move(dihedral_group[i], diff)[1]
            for j in reversed(range(i)):
                if j % 2 == 1:
                    tuple = tuple[::-1]
                else:
                    tuple[0] = 18 - tuple[0]
            return True, tuple
        else:
            i += 1
    if move_pending:
        return False, [0, 0]

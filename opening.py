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
    while(firstparen < len(s) and s[firstparen] != "(" and s[firstparen] != ";"):
        rootlabel += s[firstparen]
        firstparen += 1
    output.name = rootlabel
    x = 0
    stringy = ""
    names = list()
    for i in s:
        if(i == ')'):
            x -= 1
        if(x > 0):
            stringy += i
        if(x == 0):
            if(stringy != ""):
                names.append(stringy)
            stringy = ""
        if(i == '('):
            x += 1
    for name_i in names:
        output.add_child(gametree(name_i))
    return output

with open('fuseki.txt', 'r') as file:
    simplified = re.sub('N\[.*?\]|\n|\r', '', file.read())

opening_moves = gametree(simplified)

def boardsum(board):
    n = 0
    for i in range(19):
        for j in range(19):
            for k in range(2):
                n += board[i][j][k]
    return n


def oriented_move(board):
    nomove = True
    node = opening_moves
    check_tree = True
    depth = 0
    while check_tree :
        if board[ord(node.name[2]) - ord('a')][ ord(node.name[3]) - ord('a')][depth % 2] == 1:
            if len(node.children) != 0 :
                node = node.children[1]
            elif depth >= boardsum(board):
                nomove = False
                return (True, [ord(node.name[2]) - ord('a')][ ord(node.name[3]) - ord('a')])
                check_tree = False
            else:
                check_tree = False
    if nomove :
        return (False, [0,0])


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
    while move_pending and i < 8 :
        if oriented_move(dihedral_group[i]):
            move_pending = False
            return oriented_move(dihedral_group[i])
        else:
            i += 1
    if(move_pending):
        return (False, [0,0])






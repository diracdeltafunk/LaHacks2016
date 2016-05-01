import re

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

for i in opening_moves.children:
    print(i.name)
    for j in i.children:
        print("     " + j.name)
        for k in j.children:
            print("          " + k.name)
            for l in k.children:
                print("               " + l.name)
                for m in l.children:
                    print("                    " + m.name)
                    for n in m.children:
                        print("                         " + n.name)
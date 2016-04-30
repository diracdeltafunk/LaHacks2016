import numpy as np
import tarfile
import re

def getdata(tar, filename):
    member = tar.getmember("pro/" + filename)
    f = tar.extractfile(member)
    data = str(f.read())
    black = np.zeros((19, 19), dtype=np.int)
    white = np.zeros((19, 19), dtype=np.int)
    ko = np.zeros((19, 19), dtype=np.int)
    pos = np.zeros((19, 19, 3), dtype=np.int)
    answer = np.zeros((19, 19), dtype=np.int)
    positions, moves = [], []
    starts = [m.start() for m in re.finditer(';', data)]
    for i in starts[1:]:
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        answer[x][y] = 1
        if data[i + 1] == 'B':
            for i in range(19):
                for j in range(19):
                    pos[i][j][0] = black[i][j]
                    pos[i][j][1] = white[i][j]
                    pos[i][j][2] = ko[i][j]
            positions.append(np.copy(pos))
            black[x][y] = 1
        else:
            for i in range(19):
                for j in range(19):
                    pos[i][j][0] = white[i][j]
                    pos[i][j][1] = black[i][j]
                    pos[i][j][2] = ko[i][j]
            positions.append(np.copy(pos))
            white[x][y] = 1
        moves.append(np.copy(answer))
        answer[x][y] = 0
    return [positions, moves]

# res = getdata(tar, "00010.sgf")
# arr = res[11][3]
# np.savetxt('test.txt', arr, '%d')

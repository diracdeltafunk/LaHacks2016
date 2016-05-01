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
    starts = [m.start() for m in re.finditer(';', data)]
    length = len(starts) - 1
    j = 0
    positions, moves = np.zeros((length, 19, 19, 3)), np.zeros((length, 19, 19))
    for i in starts[1:]:
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        answer[x][y] = 1
        if data[i + 1] == 'B':
            for i in range(19):
                for j in range(19):
                    pos[i][j][0] = black[i][j]
                    pos[i][j][1] = white[i][j]
                    pos[i][j][2] = ko[i][j]
            positions[j] = np.copy(pos)
            black[x][y] = 1
        else:
            for i in range(19):
                for j in range(19):
                    pos[i][j][0] = white[i][j]
                    pos[i][j][1] = black[i][j]
                    pos[i][j][2] = ko[i][j]
            positions[j] = np.copy(pos)
            white[x][y] = 1
        moves[j] = np.copy(answer)
        answer[x][y] = 0
        j += 1
    return [positions, moves]

# res = getdata(tar, "00010.sgf")
# arr = res[11][3]
# np.savetxt('test.txt', arr, '%d')

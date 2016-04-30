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
    answer = np.zeros((19, 19), dtype=np.int)
    res = []
    starts = [m.start() for m in re.finditer(';', data)]
    for i in starts[1:]:
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        answer[x][y] = 1
        if data[i + 1] == 'B':
            res.append([np.copy(black), np.copy(white), ko, np.copy(answer)])
            black[x][y] = 1
        else:
            res.append([np.copy(white), np.copy(black), ko, np.copy(answer)])
            white[x][y] = 1
        answer[x][y] = 0
    return res

# res = getdata(tar, "00010.sgf")
# arr = res[11][3]
# np.savetxt('test.txt', arr, '%d')

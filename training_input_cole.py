import numpy as np
import boardchange as bc
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
    positions, moves = np.zeros((length, 19, 19, 3), dtype=np.int), np.zeros((length, 19, 19), dtype=np.int)
    for j1, i in enumerate(starts[1:]):
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        answer[x][y] = 1
        positions[j1] = np.copy(pos)
        pos = bc.boardchange(np.copy(pos), [x, y])
        moves[j1] = np.copy(answer)
        answer[x][y] = 0
    return [positions, moves]

tar = tarfile.open("pro.tar.gz", 'r:gz')
res = getdata(tar, "00010.sgf")
arr = res[0][4]
f = open('test.txt', 'w')
for i in range(19):
    for j in range(19):
        f.write(str(arr[i][j][0]) + ' ')
    f.write('\n')

# np.savetxt('test.txt', arr, '%d')

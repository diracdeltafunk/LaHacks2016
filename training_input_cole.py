import numpy as np
import boardchange as bc
import tarfile
import re
import random


def getdata(tar, filename):
    member = tar.getmember("pro/" + filename)
    f = tar.extractfile(member)
    data = str(f.read())
    starts = [m.start() for m in re.finditer(';', data)]
    length = len(starts) - 1
    if length < 60:
        return True, -1, -1
    pos = np.zeros((19, 19, 3), dtype=np.int)
    answer = np.zeros((19, 19), dtype=np.int)
    st = random.randrange(length - 51)
    positions, moves = np.zeros((length, 19, 19, 3), dtype=np.int), np.zeros((length, 19, 19), dtype=np.int)
    for j1, i in enumerate(starts[1:]):
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        if x < 0 or x >= 19 or y < 0 or y >= 19:
            continue
        answer[x][y] = 1
        positions[j1] = np.copy(pos)
        pos = bc.boardchange(np.copy(pos), [x, y])
        moves[j1] = np.copy(answer)
        answer[x][y] = 0
    return False, positions[st:st + 50], moves[st:st + 50]

#tar = tarfile.open("pro.tar.gz", 'r:gz')
#res = getdata(tar, "00373.sgf")
#arr = res[0][0]
#f = open('test.txt', 'w')
#for i in range(19):
 #   for j in range(19):
#        if arr[j][i][0] == 1:
 #           f.write('O ')
  #      elif arr[j][i][1] == 1:
   #         f.write('# ')
#        elif arr[j][i][2] == 1:
 #           f.write('k ')
  #      else:
   #         f.write('+ ')
 #   f.write('\n')

# np.savetxt('test.txt', arr, '%d')
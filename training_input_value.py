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
    if length < 60 or '+' not in data:
        return True, -1, -1
    ind = data.index('+')
    pos = np.zeros((19, 19, 3), dtype=np.int)
    st = random.randrange(length - 51)
    positions, wins = np.zeros((length, 19, 19, 3), dtype=np.int), np.zeros((length, 2), dtype=np.int)
    for j1, i in enumerate(starts[1:]):
        x, y = ord(data[i + 3]) - ord('a'), ord(data[i + 4]) - ord('a')
        if x < 0 or x >= 19 or y < 0 or y >= 19:
            continue
        move = j1 - 1
        positions[move] = np.copy(pos)
        pos = bc.boardchange(np.copy(pos), [x, y])
        if (data[ind - 1] == 'B' and move % 2 == 0) or (data[ind - 1] == 'W' and move % 2 == 1):
            wins[move] = [1, 0]
        else:
            wins[move] = [0, 1]
    return False, positions[st:st + 50], wins[st:st + 50]

#tar = tarfile.open("pro.tar.gz", 'r:gz')
#res = getdata(tar, "00373.sgf")
#print(res[2][1])
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
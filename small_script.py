import numpy as np
import opening as op

pos = np.zeros((19, 19, 3), dtype=np.int)
pos[3][3][1] = 1
print(op.make_move(pos))
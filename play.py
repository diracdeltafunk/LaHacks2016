import tensorflow as tf
import numpy as np
import opening as op
import boardchange as bc
import itertools
import random
import sys

def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    pre_x = tf.placeholder(tf.float32, [19, 19, 3])
    x = tf.expand_dims(pre_x, 0)
    w1 = weight([7, 7, 3, 48])
    b1 = bias([48])

    conv1 = tf.nn.relu(conv2d(x, w1) + b1)

    w2 = weight([5, 5, 48, 32])
    b2 = bias([32])

    conv2 = tf.nn.relu(conv2d(conv1, w2) + b2)

    w3 = weight([5, 5, 32, 32])
    b3 = bias([32])

    conv3 = tf.nn.relu(conv2d(conv2, w3) + b3)

    w4 = weight([5, 5, 32, 32])
    b4 = bias([32])

    conv4 = tf.nn.relu(conv2d(conv3, w4) + b4)

    w5 = weight([19 * 19 * 32, 2048])
    b5 = bias([2048])

    flat = tf.reshape(conv4, [1, 19 * 19 * 32])
    dense0 = tf.nn.relu(tf.matmul(flat, w5) + b5)

    keep_prob = tf.placeholder(tf.float32)
    dense = tf.nn.dropout(dense0, keep_prob)

    w6 = weight([2048, 19 * 19])
    b6 = bias([19 * 19])

    res_flat = tf.nn.softmax(tf.matmul(dense, w6) + b6)

    res = tf.reshape(res_flat, [1, 19, 19])

with g2.as_default():
    pre_x_v = tf.placeholder(tf.float32, [19, 19, 3])
    x_v = tf.expand_dims(pre_x_v, 0)
    w1_v = weight([7, 7, 3, 48])
    b1_v = bias([48])

    conv1_v = tf.nn.relu(conv2d(x_v, w1_v) + b1_v)

    w3_v = weight([5, 5, 48, 32])
    b3_v = bias([32])

    conv3_v = tf.nn.relu(conv2d(conv1_v, w3_v) + b3_v)

    w4_v = weight([5, 5, 32, 32])
    b4_v = bias([32])

    conv4_v = tf.nn.relu(conv2d(conv3_v, w4_v) + b4_v)

    w5_v = weight([19 * 19 * 32, 2048])
    b5_v = bias([2048])

    flat_v = tf.reshape(conv4_v, [1, 19 * 19 * 32])
    dense0_v = tf.nn.relu(tf.matmul(flat_v, w5_v) + b5_v)

    keep_prob_v = tf.placeholder(tf.float32)
    dense_v = tf.nn.dropout(dense0_v, keep_prob_v)

    w6_v = weight([2048, 2])
    b6_v = bias([2])

    res_flat_v = tf.nn.softmax(tf.matmul(dense_v, w6_v) + b6_v)

    res_v = tf.reshape(res_flat_v, [1, 2])

sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g2)

with g1.as_default():
    saver1 = tf.train.Saver()
with g2.as_default():
    saver2 = tf.train.Saver()

with g1.as_default():
    saver1.restore(sess1, 'saved_policy_network_final.ckpt')
with g2.as_default():
    saver2.restore(sess2, 'saved_value_network_final.ckpt')

## END NEURAL NET CODE
## BEGIN GAMEPLAY CODE

num_moves_considered = 5
##MUST BE ODD
##MUST BE MORE THAN ONE
depth_to_consider = 6

class GameTree(object):
    def __init__(self, name=[None, None], children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, GameTree)
        self.children.append(node)

gamestate = np.zeros((19,19,3), dtype=int)
gametree = GameTree(name=[gamestate,None])

def flip(gs):
    t = np.copy(gs[:,:,0])
    gs[:,:,0] = gs[:,:,1]
    gs[:,:,1] = t
    return gs

def showBoard():
    result = "   a b c d e f g h i j k l m n o p q r s \n"
    for i in range(19):
        result += str(i).zfill(2) + ' '
        for j in range(19):
            if gamestate[i][j][0] == 1:
                result += '0 ' enumerate(
            else:
                result += '+ '
        result += '\n'
    return result

def do_move(gs, pos, isBlackMove):
    if not isBlackMove:
        gs = flip(gs)
    gs = bc.boardchange(np.copy(gs), pos)
    if isBlackMove:
        gs = flip(gs)
    return gs

def updateTree(root, width, depth):
    if depth == 0:
        root.name[1] = sess2.run(res_v, feed_dict={pre_x_v: root.name[0], keep_prob_v: 1.0})[0][0]
        root.children = None
        return root
    if root.children is not None and root.children != []:
        root.children = [updateTree(c,width,depth-1) for c in root.children]
        return root
    p_predicts = sess1.run(res, feed_dict={pre_x: root.name[0], keep_prob: 1.0})[0]
    p_indices = [[i, j] for i, j in itertools.product(*[range(19), range(19)]) if not np.any(root.name[0][i][j])]
    p_indices.sort(key=lambda x: p_predicts[x[0]][x[1]], reverse=True)
    root.children = [updateTree(GameTree(name=do_move(np.copy(root.name[0]), x, True)), width, depth-1) for x in p_indices[:width]]
    return root

def moveTree(direction):
    global gametree
    gametree = gametree.children[direction]

def matchTreeToState():
    global gamestate
    global gametree
    if gametree.children is None or gametree.children == []:
        gametree = GameTree(name=[gamestate, None])
        return
    for c in gametree.children:
        if gamestate == c.name[0]:
            gametree = c
            return
    gametree = GameTree(name=[gamestate, None])

def playMove():
    global gamestate
    global depth_to_consider
    global num_moves_considered
    global gametree
    fuseki_match, fuseki_move = op.make_move(gamestate)
    if fuseki_match:
        print('Found a cool fuseki move!')
        gamestate = do_move(gamestate, fuseki_move, True)
        matchTreeToState()
        return
    print('Thinking hard about this one...')
    gametree = updateTree(gametree, num_moves_considered, depth_to_consider)
    print('I\'ve made a game tree!')
    moveTree(tMiniMax(gametree))
    gamestate = np.copy(gametree.name[0])

def tMin(tree):
    if tree.children is None or tree.children == []]:
        return tree.name[1]
    return min([tMax(t) for t in tree.children])

def tMax(tree):
    if tree.children is None or tree.children == []]:
        return tree.name[1]
    return max([tMin(t) for t in tree.children])

def tMiniMax(tree):
    if tree.children is None or tree.children == []:
        return tree.name[1]
    max_possibility = tMax(tree)
    return list(map(tMin, tree.children)).index(max_possibility)

gameDone = False
while not gameDone:
    playMove()
    print(showBoard())
    player_entered_move = False
    while not player_entered_move:
      player_move = input('Enter your move, e.g. a15, or q to be done\n').lower()[:3]
      if len(player_move) >= 2:
        player_entered_move = True
      elif player_move == 'q':
        gameDone = True
      else:
        print('Couldn\'t understand move.')
      if not gameDone:
        player_move_formatted = [int(player_move[1:]), ord(player_move[0]) - ord('a')]
        gamestate = do_move(gamestate, player_move_formatted, False)
        matchTreeToState()

sess1.close()
sess2.close()
sys.exit(0)

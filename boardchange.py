import numpy

def boardchange(input, position):
    board = numpy.transpose(input, (2, 0, 1))
    my_stones = board[0]
    my_stones[position[0]][position[1]] = 1
    op_stones = board[1]

    def dfs(point_in, boundary):
        kill_list = []
        visited = numpy.copy(boundary)
        visited[point_in[0]][point_in[1]] = 1

        def recursive_dfs(point):
            visited[point[0]][point[1]] = 1
            ptlist = []
            liberty = False
            surround = True
            if point[0] != 0:
                if my_stones[point[0] - 1][point[1]] == 0 and op_stones[point[0] - 1][point[1]] == 0:
                    liberty = True
                if visited[point[0] - 1][point[1]] == 0 and op_stones[point[0] - 1][point[1]] == 1:
                    ptlist.append([point[0] - 1, point[1]])
            if point[0] != 18:
                if my_stones[point[0] + 1][point[1]] == 0 and op_stones[point[0] + 1][point[1]] == 0:
                    liberty = True
                if visited[point[0] + 1][point[1]] == 0 and op_stones[point[0] + 1][point[1]] == 1:
                    ptlist.append([point[0] + 1, point[1]])
            if point[1] != 0:
                if my_stones[point[0]][point[1] - 1] == 0 and op_stones[point[0]][point[1] - 1] == 0:
                    liberty = True
                if visited[point[0]][point[1]-1] == 0 and op_stones[point[0]][point[1] - 1] == 1:
                    ptlist.append([point[0], point[1]-1])
            if point[1] != 18:
                if my_stones[point[0]][point[1] + 1] == 0 and op_stones[point[0]][point[1] + 1] == 0:
                    liberty = True
                if visited[point[0]][point[1] + 1] == 0 and op_stones[point[0]][point[1] + 1] == 1:
                    ptlist.append([point[0], point[1] + 1])
            if liberty:
                return False
            else:
                for pt in ptlist:
                    surround = surround and recursive_dfs(pt)
                if surround:
                    kill_list.append([point[0], point[1]])
                return surround

        if point_in[0] != 0:
            recursive_dfs([point_in[0] - 1, point_in[1]])
        if point_in[0] != 18:
            visited = numpy.copy(boundary)
            recursive_dfs([point_in[0] + 1, point_in[1]])
        if point_in[1] != 0:
            visited = numpy.copy(boundary)
            recursive_dfs([point_in[0], point_in[1] - 1])
        if point_in[1] != 18:
            visited = numpy.copy(boundary)
            recursive_dfs([point_in[0], point_in[1] + 1])

        return kill_list

    kill_list = dfs([position[0], position[1]], numpy.copy(my_stones))
    ko = numpy.zeros((19, 19), dtype=numpy.int)
    if len(kill_list) == 1:
        kopt = True
        if position[0] != 0:
            kopt = kopt and (op_stones[position[0] - 1][position[1]])
        if position[0] != 18:
            kopt = kopt and (op_stones[position[0] + 1][position[1]])
        if position[1] != 0:
            kopt = kopt and (op_stones[position[0]][position[1] - 1])
        if position[1] != 18:
            kopt = kopt and (op_stones[position[0]][position[1] + 1])
        if kopt:
            ko[kill_list[0][0]][kill_list[0][1]] = 1
    for point in kill_list:
        op_stones[point[0]][point[1]] = 0
    return numpy.transpose([op_stones,my_stones,ko],(1,2,0))

initial = numpy.zeros((3, 19, 19), dtype=numpy.int)


import numpy as np
import csv
import time
import math
import operator
def GetSurroundingContent(row, col, grid):
    rows, cols = np.shape(grid)
    neighboringCells = []
    actionlist=[]
    if (row >= 1):
        # add north edge to the selected cell
        if(grid[row - 1][col]!="1"):
            neighboringCells.append((row - 1, col))
            actionlist.append("UP")
    if (row < rows - 1):
        # add south edge to the selected cell
        if (grid[row + 1][col] != "1"):
            neighboringCells.append((row + 1, col))
            actionlist.append("DOWN")
    if (col >= 1):
        # add west edge to the selected cell
        if (grid[row][col - 1] != "1"):
            neighboringCells.append((row, col - 1))
            actionlist.append("LEFT")
    if (col < cols - 1):
        # add east edge to the selected cell
        if (grid[row][col + 1] != "1"):
            neighboringCells.append((row, col + 1))
            actionlist.append("RIGHT")

    return neighboringCells,actionlist
def GetSurroundingBlockSum(row, col, grid):
    rows, cols = np.shape(grid)
    neighboringCells = []
    number = 0
    rows, cols = np.shape(grid)
    startRow = 0
    endRow = rows - 1
    startCol = 0
    endCol = cols - 1
    warningCells = []
    if row - 1 >= 0:
        startRow = row - 1
    if row + 1 < rows:
        endRow = row + 1
    if col - 1 >= 0:
        startCol = col - 1
    if col + 1 < cols:
        endCol = col + 1

    for i in range(startRow, endRow + 1):
        for j in range(startCol, endCol + 1):
            if ((i == row and j == col)):
                continue
            if (grid[i][j] == "1"):
                neighboringCells.append((i, j))
                number = number + 1


    return neighboringCells, number

def GetSurroundingAvailableCellSum(row, col, grid):
    rows, cols = np.shape(grid)
    neighboringCells = []
    number = 0
    rows, cols = np.shape(grid)
    startRow = 0
    endRow = rows - 1
    startCol = 0
    endCol = cols - 1
    warningCells = []
    if row - 1 >= 0:
        startRow = row - 1
    if row + 1 < rows:
        endRow = row + 1
    if col - 1 >= 0:
        startCol = col - 1
    if col + 1 < cols:
        endCol = col + 1

    for i in range(startRow, endRow + 1):
        for j in range(startCol, endCol + 1):
            if ((i == row and j == col)):
                continue
            if (grid[i][j] != "1"):
                neighboringCells.append((i, j))
                number = number + 1

    return neighboringCells, number


def manhattan(a, b):
    a_x, a_y = a;
    b_x, b_y = b;

    return math.fabs(a_x - b_x) + math.fabs(a_y - b_y)
def GetManhattanDistanceAllCombinnation(xList,yList, matrixmaze):
    glocx,glocy=np.where(matrixmaze=="G")
    glocx = glocx[0]
    glocy = glocy[0]
    cost = []
    pairOfCells=[]
    for i in range(len(xList)-1):
        for j in range(i+1,len(xList),1):
            if(manhattan((xList[i],yList[i]) , (glocx,glocy)) < manhattan((xList[j],yList[j]) , (glocx,glocy))):
                pair1=(xList[i],yList[i])
                pair2=(xList[j],yList[j])
            else:
                pair1 = (xList[j], yList[j])
                pair2 = (xList[i], yList[i])
            cost.append(manhattan(pair1,pair2))
            pairOfCells.append((pair1,pair2))

    return pairOfCells, cost

def FindSourceAndDestination(x,y,x2,y2, matrixmaze):
    glocx,glocy=np.where(matrixmaze=="G")
    glocx = glocx[0]
    glocy = glocy[0]


    if(manhattan((x,y) , (glocx,glocy)) < manhattan((x2,y2) , (glocx,glocy))):
        pair1=(x,y)
        pair2=(x2,y2)
    else:
        pair1 = (x2, y2)
        pair2 = (x, y)


    return pair1,pair2, manhattan(pair1,pair2)


def MovePopulation(randomaction,row,col,  beliefMatrix1a,matrixmaze):
    neighboringCells, actionlist = GetSurroundingContent(row, col, matrixmaze)
    previousrow = row
    previouscol = col
    if (randomaction == "UP" and (row - 1, col) in neighboringCells):
        row = row - 1
        beliefMatrix1a[row][col] = beliefMatrix1a[previousrow][previouscol] + beliefMatrix1a[row][col]
        # if(beliefMatrix1a[previousrow][previouscol] ==1):
        beliefMatrix1a[previousrow][previouscol] = 0

    elif (randomaction == "DOWN" and (row + 1, col) in neighboringCells):
        row = row + 1
        beliefMatrix1a[row][col] = beliefMatrix1a[previousrow][previouscol] + beliefMatrix1a[row][col]
        # if (beliefMatrix1a[previousrow][previouscol] == 1):
        beliefMatrix1a[previousrow][previouscol] = 0

    elif (randomaction == "LEFT" and (row, col - 1) in neighboringCells):
        col = col - 1
        beliefMatrix1a[row][col] = beliefMatrix1a[previousrow][previouscol] + beliefMatrix1a[row][col]
        # if (beliefMatrix1a[previousrow][previouscol] == 1):
        beliefMatrix1a[previousrow][previouscol] = 0

    elif (randomaction == "RIGHT" and (row, col + 1) in neighboringCells):
        col = col + 1
        beliefMatrix1a[row][col] = beliefMatrix1a[previousrow][previouscol] + beliefMatrix1a[row][col]
        # if (beliefMatrix1a[previousrow][previouscol] == 1):
        beliefMatrix1a[previousrow][previouscol] = 0
    return beliefMatrix1a,row,col

def BFS(originx,originy, gx,gy, maze_matrix,beliefMatrix,sequenceOfmMovesList):
    print('------BFS---------')
    # start = time.time()

    origin = (originx, originy)
    visited, queue = list(), [(origin, '')]
    #visited_path = list()
    # mazeH, mazeW= (maze.shape)
    # goal = (mazeH-1, mazeW-1)
    goal = (gx,gy)
    previousrow=originx
    previouscol=originy
    while queue:
        vertex, path = queue.pop(0)
        # print(vertex)

        if vertex not in visited:
            visited.append(vertex)
            #visited_path.append(path)

            row=vertex[0]
            col=vertex[1]
            if(path!=""):
                sequenceOfmMovesList.append(path)
                beliefMatrix[row][col] = beliefMatrix[previousrow][previouscol] + beliefMatrix[row][col]
                beliefMatrix[previousrow][previouscol]=0
            previousrow=row
            previouscol=col
        else:
            continue
        if vertex == goal:
            break
            # print ('Time upon completion: ' + str(time.time() - start) + ' seconds')

        # print (maze_dict[vertex])
        neighboringCells, actionlist=GetSurroundingContent(vertex[0],vertex[1],maze_matrix)
        index=0
        for children in neighboringCells:
            if children not in visited:
                path=actionlist[index]
                queue.append((children, path))
            index=index+1

    print('Not Found')
    print('Number of vertices visited: ' + str(len(visited)))
    # print ('Time upon completion: ' + str(time.time() - start) + ' seconds')
    return vertex[0],vertex[1] , sequenceOfmMovesList,beliefMatrix




def MoveBasedOnlyTwoPop(source, destination, matrixmaze, beliefMatrix,sequenceOfmMovesList):
    # target_x, target_y is the absolute max belief cell (where we would go in one jump, if we could)
    # first pick neighbors with minimum cost
    x=source[0]
    y=source[1]
    target_x=destination[0]
    target_y=destination[1]
    visited=[]
    previousrow=x
    previouscol=y
    while(x!=target_x or y!=target_y):
        neighboringCells, actionlist = GetSurroundingContent(x, y, matrixmaze)

        cellAndAction={}
        costDict = {}
        index=0
        numberOfSurroundingBlockedCells=[]

        cost = []
        for cell in neighboringCells:
            if (cell not in visited):
                cost.append((cell, manhattan((target_x, target_y), cell)))
                cellAndAction[cell] = actionlist[index]
            index = index + 1
        if(len(cost)>0):
            min_cost_cell, min_cost = min(cost, key=lambda t: t[1])
            min_cost_list = [item for item in cost if item[1] == min_cost]

            for item in min_cost_list:
                cell, cost = item
                i, j = cell
                action= cellAndAction[(i,j)]
                visited.append(cell)
                row=i
                col=j

                if (action == "UP" ):
                    row = row - 1
                    beliefMatrix[row][col] = beliefMatrix[previousrow][previouscol] + beliefMatrix[row][col]
                    # if(beliefMatrix1a[previousrow][previouscol] ==1):
                    beliefMatrix[previousrow][previouscol] = 0
                    sequenceOfmMovesList.append(action)

                elif (action == "DOWN" ):
                    row = row + 1
                    beliefMatrix[row][col] = beliefMatrix[previousrow][previouscol] + beliefMatrix[row][col]
                    # if (beliefMatrix1a[previousrow][previouscol] == 1):
                    beliefMatrix[previousrow][previouscol] = 0
                    sequenceOfmMovesList.append(action)

                elif (action == "LEFT"):
                    col = col - 1
                    beliefMatrix[row][col] = beliefMatrix[previousrow][previouscol] + beliefMatrix[row][col]
                    # if (beliefMatrix1a[previousrow][previouscol] == 1):
                    beliefMatrix[previousrow][previouscol] = 0
                    sequenceOfmMovesList.append(action)

                elif (action == "RIGHT" ):
                    col = col + 1
                    beliefMatrix[row][col] = beliefMatrix[previousrow][previouscol] + beliefMatrix[row][col]
                    # if (beliefMatrix1a[previousrow][previouscol] == 1):
                    beliefMatrix[previousrow][previouscol] = 0
                    sequenceOfmMovesList.append(action)
                x = row
                y = col
                previousrow = row
                previouscol = col
        else:
            x,y,sequenceOfmMovesList, beliefMatrix=BFS(x, y, target_x ,target_y, matrixmaze, beliefMatrix, sequenceOfmMovesList)

    return  sequenceOfmMovesList,beliefMatrix
def Question1b():
    with open("Localization.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)

    matrixmaze = np.array(d)

    height, width = matrixmaze.shape
    # x_str = np.array_repr(matrixmaze).replace('\n', '')
    xrange, yrange = np.where(matrixmaze != "1")  # select cells with minimum probability
    probfNonBlockCell = 1 / len(xrange)

    ##########################
    beliefMatrix1a = np.zeros((height, width))

    sequenceOfmMovesList = []

    for index2 in range(len(xrange)):
        row = xrange[index2]
        col = yrange[index2]
        beliefMatrix1a[row][col] = 1

    listofactions = ["UP", "DOWN", "RIGHT", "LEFT"]
    currentProbability = 1 / len(xrange)
    randomaction = ""
    newactiontochoose = []
    cutoof = len(xrange) / 2
    foundmorethan = False

    g_xloc, g_yloc = np.where(matrixmaze == "G")

    start = time.time()
    end = time.time()

    for i in range(36):
        newactiontochoose.append("UP")
    for i in range(10):
        newactiontochoose.append("LEFT")
    for i in range(10):
        newactiontochoose.append("DOWN")
    for i in range(10):
        newactiontochoose.append("RIGHT")
    for i in range(10):
        newactiontochoose.append("UP")

    findgoal=False
    while (findgoal==False):  # len(newactiontochoose)>0):

        for randomaction in newactiontochoose:
            indexaction = 0
            sequenceOfmMovesList.append(randomaction)
            for index2 in range(len(xrange)):
                row = xrange[index2]
                col = yrange[index2]

                beliefMatrix1a, row, col = MovePopulation(randomaction, row, col, beliefMatrix1a, matrixmaze)

            xrange, yrange = np.where(beliefMatrix1a > 0)
            atleasthalfx, atleasthalfy = np.where(beliefMatrix1a >= cutoof)
            if(len(atleasthalfy)>0):
                print(atleasthalfx)
                print(atleasthalfy)
        popDict={}
        population_x, population_y = np.where(beliefMatrix1a > 1)
        for i in range(len(population_x)):
            popDict[(population_x[i],population_y[i])]=beliefMatrix1a[population_x[i]][population_y[i]]
        sorted_max = sorted(popDict.items(), key=operator.itemgetter(1))
        halffound=False
        while(halffound==False):
            max1 = sorted_max[len(sorted_max)-1][0]
            #for si in range(len(sorted_max)-1,1,-1):
            max2=sorted_max[len(sorted_max)-2][0]
            pair1, pair2, costmax=FindSourceAndDestination(max1[0],max1[1],max2[0],max2[1],matrixmaze)

            sequenceOfmMovesList, beliefMatrix1a = MoveBasedOnlyTwoPop(pair1, pair2, matrixmaze,beliefMatrix1a,sequenceOfmMovesList)

            atleasthalfx, atleasthalfy = np.where(beliefMatrix1a >= cutoof)
            if (len(atleasthalfy) > 0):

                halffound=True
                break
            popDict = {}
            population_x, population_y = np.where(beliefMatrix1a > 1)
            for i in range(len(population_x)):
                popDict[(population_x[i], population_y[i])] = beliefMatrix1a[population_x[i]][population_y[i]]
            sorted_max = sorted(popDict.items(), key=operator.itemgetter(1))
        indicesx,indicesy = np.where(beliefMatrix1a == beliefMatrix1a.max())
        row, col, sequenceOfmMovesList, beliefMatrix1a=BFS(indicesx[0], indicesy[0], g_xloc[0], g_yloc[0], matrixmaze, beliefMatrix1a, sequenceOfmMovesList)
        findgoal=True
        end = time.time()
    print(beliefMatrix1a)
    print(sequenceOfmMovesList)
    print(len(sequenceOfmMovesList))
def Question1C():
    with open("Localization.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)

    matrixmaze = np.array(d)

    height, width = matrixmaze.shape
    # x_str = np.array_repr(matrixmaze).replace('\n', '')
    xrange, yrange = np.where(matrixmaze != "1")  # select cells with minimum probability
    probfNonBlockCell = 1 / len(xrange)

    ##########################
    beliefMatrix1a = np.zeros((height, width))

    sequenceOfmMovesList = []

    for index2 in range(len(xrange)):
        row = xrange[index2]
        col = yrange[index2]
        beliefMatrix1a[row][col] = 1

    listofactions = ["UP", "DOWN", "RIGHT", "LEFT"]
    currentProbability = 1 / len(xrange)
    randomaction = ""
    newactiontochoose = []
    cutoof = len(xrange) / 2
    foundmorethan = False

    g_xloc, g_yloc = np.where(matrixmaze == "G")

    start = time.time()
    end = time.time()

    for i in range(36):
        newactiontochoose.append("UP")
    for i in range(10):
        newactiontochoose.append("LEFT")
    for i in range(10):
        newactiontochoose.append("DOWN")
    for i in range(10):
        newactiontochoose.append("RIGHT")
    for i in range(10):
        newactiontochoose.append("UP")

    findgoal=False
    while (findgoal==False):  # len(newactiontochoose)>0):

        for randomaction in newactiontochoose:
            indexaction = 0
            sequenceOfmMovesList.append(randomaction)
            for index2 in range(len(xrange)):
                row = xrange[index2]
                col = yrange[index2]

                beliefMatrix1a, row, col = MovePopulation(randomaction, row, col, beliefMatrix1a, matrixmaze)

            xrange, yrange = np.where(beliefMatrix1a > 0)
            atleasthalfx, atleasthalfy = np.where(beliefMatrix1a >= cutoof)
            if(len(atleasthalfy)>0):
                print(atleasthalfx)
                print(atleasthalfy)
        popDict={}
        population_x, population_y = np.where(beliefMatrix1a > 1)
        for i in range(len(population_x)):
            popDict[(population_x[i],population_y[i])]=beliefMatrix1a[population_x[i]][population_y[i]]
        sorted_max = sorted(popDict.items(), key=operator.itemgetter(1))
        onpopulationleft=False
        while(onpopulationleft==False):
            max1 = sorted_max[len(sorted_max)-1][0]
            #for si in range(len(sorted_max)-1,1,-1):
            max2=sorted_max[len(sorted_max)-2][0]
            pair1, pair2, costmax=FindSourceAndDestination(max1[0],max1[1],max2[0],max2[1],matrixmaze)

            sequenceOfmMovesList, beliefMatrix1a = MoveBasedOnlyTwoPop(pair1, pair2, matrixmaze,beliefMatrix1a,sequenceOfmMovesList)

            atleasthalfx, atleasthalfy = np.where(beliefMatrix1a > 0)
            if (len(atleasthalfy) ==1):
                onpopulationleft=True
                break
            popDict = {}
            population_x, population_y = np.where(beliefMatrix1a > 1)
            for i in range(len(population_x)):
                popDict[(population_x[i], population_y[i])] = beliefMatrix1a[population_x[i]][population_y[i]]
            sorted_max = sorted(popDict.items(), key=operator.itemgetter(1))
        indicesx,indicesy = np.where(beliefMatrix1a == beliefMatrix1a.max())
        row, col, sequenceOfmMovesList, beliefMatrix1a=BFS(indicesx[0], indicesy[0], g_xloc[0], g_yloc[0], matrixmaze, beliefMatrix1a, sequenceOfmMovesList)
        findgoal=True
        end = time.time()
    print(beliefMatrix1a)
    print(sequenceOfmMovesList)
    print(len(sequenceOfmMovesList))

def Question1a():
    with open("Localization.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)


    matrixmaze = np.array(d)


    height, width = matrixmaze.shape
    # x_str = np.array_repr(matrixmaze).replace('\n', '')
    xrange, yrange = np.where(matrixmaze != "1")
    probfNonBlockCell = 1 / len(xrange)
    print("1. a) What is the probability you are at G?")
    print("Answer: " + str(probfNonBlockCell))
def Question1d1():
    with open("Localization.txt") as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)


    matrixmaze = np.array(d)


    height, width = matrixmaze.shape
    # x_str = np.array_repr(matrixmaze).replace('\n', '')
    xrange, yrange = np.where(matrixmaze != "1")

    probfNonBlockCell = 1 / len(xrange)
    print("1. a) What is the probability you are at G?")
    print("Answer: " + str(probfNonBlockCell))

    print(len(xrange))
    fiveblockcell = {}
    print(probfNonBlockCell)
    beliefMatrixfor5blocked = np.zeros((height, width))
    for row in range(height):
        for col in range(width):
            neighboringCells, number = GetSurroundingBlockSum(row, col, matrixmaze)
            if (number == 5):
                neighboringCells2, number2 = GetSurroundingAvailableCellSum(row, col, matrixmaze)
                if ((row, col - 1) in neighboringCells2):
                    neighboringCells3, number3 = GetSurroundingBlockSum(row, col - 1, matrixmaze)
                    if (number3 == 5):
                        fiveblockcell[(row, col - 1)] = neighboringCells

    for key, value in fiveblockcell.items():
        print(key)
        beliefMatrixfor5blocked[key[0]][key[1]] = 1 / len(fiveblockcell)
    xx,yy = np.where(beliefMatrixfor5blocked>0)

    for ind in range(len(xx)):
        print("Cell : "+ str(xx[ind])+","+str(yy[ind]) + " prob : "+str(beliefMatrixfor5blocked[xx[ind]][yy[ind]]))

    #np.savetxt('outputquestion1d.txt', beliefMatrixfor5blocked)

    #print(len(fiveblockcell))
    #print(beliefMatrixfor5blocked)



if __name__ == '__main__':
    Question1a()
    #
    Question1b()
    Question1C()
    Question1d1()




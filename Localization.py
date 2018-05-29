import numpy as np
import sys
import time
import tkinter as tk
import csv
import random
import time
import math
import os
import tkinter as tk

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
def ChooseNextCellWithouBlockedCellKnowledge(x, y, target_x, target_y, grid):

    neighborcells = GetSurroundingContent(x, y, grid)
    cost = []
    for cell in neighborcells:
        cost.append((cell, manhattan((target_x, target_y), cell)))

    min_cost_cell, min_cost = min(cost, key=lambda t: t[1])
    min_cost_list = [x for x in cost if x[1] == min_cost]




    x, y = min_cost_list

    return x, y
def ChooseNextCell(x, y, target_x, target_y, grid, beliefMatrix):

    neighborcells = GetSurroundingContent(x, y, grid)
    cost = []
    for cell in neighborcells:
        cost.append((cell, manhattan((target_x, target_y), cell)))

    min_cost_cell, min_cost = min(cost, key=lambda t: t[1])
    min_cost_list = [x for x in cost if x[1] == min_cost]

    belief = []
    for item in min_cost_list:
        cell, cost = item
        i, j = cell
        belief.append((cell, beliefMatrix[i][j]))

    # Take one of the max belief cells (with min cost)
    max_belief_cell, max_belief = max(belief, key=lambda t: t[1])
    x, y = max_belief_cell

    return x, y

with open("Localization.txt") as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)
    print(d)

matrixmaze= np.array(d)
print(matrixmaze.shape)

height, width = matrixmaze.shape
#x_str = np.array_repr(matrixmaze).replace('\n', '')
xrange, yrange = np.where(matrixmaze != "1")  # select cells with minimum probability
probfNonBlockCell=1/len(xrange)
print("1. a) What is the probability you are at G?")
print("Answer: "+str(probfNonBlockCell))

##########################
listofmatrix=[]
for index3 in range(len(xrange)):
    listofmatrix.append( np.zeros((height, width)))

sequenceOfmMovesList=[]
for index2 in range(len(xrange)):
    for index4 in range(len(xrange)):
        row=xrange[index4]
        col=yrange[index4]
        listofmatrix[index2][row][col]=1
        sequenceOfmMovesList.append([])
for index1 in range(len(xrange)):
    row = xrange[index1]
    col = yrange[index1]
    neighboringCells, mainactionlist=GetSurroundingContent(row,col,matrixmaze)
    currentProbability =1/len(xrange)
    start = time.time()
    print(index1)
    while(currentProbability<=1 and matrixmaze[row][col] != "G"):
        randomaction = random.sample(mainactionlist, 1).pop()

        for step in range(height):
            sequenceOfmMovesList[index1].append(randomaction)
            previousrow = row
            previouscol = col
            if (randomaction=="UP" and (row-1,col) in neighboringCells):
                row=row-1
            elif (randomaction=="DOWN" and (row+1,col) in neighboringCells):
                row=row+1
            elif (randomaction=="LEFT" and (row,col-1) in neighboringCells):
                col=col-1
            elif (randomaction=="RIGHT" and (row,col+1) in neighboringCells):
                col=col+1
            else:

                neighboringCells, mainactionlist = GetSurroundingContent(row, col, matrixmaze)
                if(randomaction=="RIGHT"):
                    mainactionlist.remove("LEFT")
                elif (randomaction == "LEFT"):
                    mainactionlist.remove("RIGHT")
                elif (randomaction == "UP"):
                    mainactionlist.remove("DOWN")
                elif (randomaction == "DOWN"):
                    mainactionlist.remove("UP")
                break
            if (matrixmaze[previousrow][previouscol] != "G"):

                neighboringCells, actionlist = GetSurroundingContent(row, col, matrixmaze)

                #print(listofmatrix[index1][row][col])
                allactionzero=True
                if (listofmatrix[index1][row][col] == 0):
                    counter = 0
                    for cell in neighboringCells:
                        cellrow = cell[0]
                        cellcol = cell[1]
                        if (cellrow!=previousrow and cellcol!=previouscol and listofmatrix[index1][cellrow][cellcol] > 0):
                            randomaction = actionlist[counter]
                            row=cellrow
                            col=cellcol
                            allactionzero=False
                            break;
                        counter = counter + 1
                    if(allactionzero):
                        end = time.time()
                        #print(end - start)
                        if((end - start)>10):
                            currentProbability=2
                            break;
                    # if(allactionzero):
                    #     if (randomaction == "RIGHT"):
                    #         actionlist.remove("LEFT")
                    #     elif (randomaction == "LEFT"):
                    #         actionlist.remove("RIGHT")
                    #     elif (randomaction == "UP"):
                    #         actionlist.remove("DOWN")
                    #     elif (randomaction == "DOWN"):
                    #         actionlist.remove("UP")
                    #     continueLoop=True
                    #     if(len(actionlist)>0):
                    #         while(continueLoop==True):
                    #             randomactionindex = random.sample(range(len(actionlist)), 1).pop()
                    #             row=neighboringCells[randomactionindex][0]
                    #             col=neighboringCells[randomactionindex][1]
                    #
                    #             randomaction=actionlist[randomactionindex]
                    #             if (previousrow == row and previouscol == col):
                    #                 continueLoop = True
                    #             else:
                    #                 continueLoop = False
                    #     neighboringCells, mainactionlist = GetSurroundingContent(row, col, matrixmaze)
                    #     if (randomaction == "RIGHT"):
                    #         mainactionlist.remove("LEFT")
                    #     elif (randomaction == "LEFT"):
                    #         mainactionlist.remove("RIGHT")
                    #     elif (randomaction == "UP"):
                    #         mainactionlist.remove("DOWN")
                    #     elif (randomaction == "DOWN"):
                    #         mainactionlist.remove("UP")
                listofmatrix[index1][row][col] = listofmatrix[index1][previousrow][previouscol] + listofmatrix[index1][row][col]
                listofmatrix[index1][previousrow][previouscol] = 0
                currentProbability=listofmatrix[index1][row][col] /len(xrange)


                #print(listofmatrix[index1][row][col])
    if(matrixmaze[row][col] == "G"):
        print("Found G")
        atleasthalfx,atleasthalfy= np.where(listofmatrix[index1]/len(xrange)>=(1/2))
        print(atleasthalfx)
        if(len(atleasthalfx)>0):
            print(sequenceOfmMovesList[index1])

            #if (randomaction =="DOWN"):
            #if (randomaction == "LEFT"):
            #if (randomaction == "RIGHT")



########################

#getRandomFirstPosition = random.sample(range(xrange.size), 1).pop()  # select one of the minimum probability cells at random
#x = xrange[getRandomFirstPosition]
#y = yrange[getRandomFirstPosition]
#targetx, targety = np.where(matrixmaze == "G")
#beliefMatrix = np.ones((height, width)) * (1 / (height * width))
#while(targetx[0]!=x and targety[0]!=y):
#    x,y=ChooseNextCellWithouBlockedCellKnowledge(x,y,targetx[0],targety[0],matrixmaze)
#prob_matrix = np.ones((width, height))*-1
#print(matrixmaze)
#inputGrid = np.loadtxt("Localization.txt", dtype='i', delimiter='\t')
#print(inputGrid)
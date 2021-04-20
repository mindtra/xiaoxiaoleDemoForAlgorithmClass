import matplotlib
import numpy as np
import math
import random
import time
from copy import copy, deepcopy
from enum import Enum

from PIL import Image, ImageDraw, ImageFont, ImageQt

from timecount import time_count
import matplotlib.pyplot as plt
import matplotlib.image as mpimage  # 读取图片

colors = ['lime', 'r', 'b', 'm', 'slategrey', 'c', 'y', 'k', 'sandybrown', 'g']
markers = ['o', 'x', '+', '^', 'v', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd',
           'P', 'X']  # 圆圈， x, +, 正三角， 倒三角...
styles = [x + y for x in colors for y in markers]  # 这样就有5*5=25中不同组合
matplotlib.rcParams['axes.unicode_minus'] = False  # 设置字体
ROW_MARK = 1
COL_MARK = 2


class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __str__(self):
        return str(self.row) + ', ' + str(self.col)

    def setPoint(self, row, col):
        self.row = row
        self.col = col

    def getRow(self):
        return self.row

    def getCol(self):
        return self.col


class ArrowE(Enum):
    UP = (1, '^', 'up', Point(-1, 0))
    RIGHT = (2, ">", "right", Point(0, 1))
    DOWN = (3, "v", "down", Point(1, 0))
    LEFT = (4, "<", "left", Point(0, -1))

    # arrowDictReverse = {UP: {DOWN}, DOWN: {UP}, LEFT: {RIGHT}, RIGHT: {LEFT}}
    #
    # def getReverse(self):
    #     # print(self)
    #     return ArrowE.arrowDictReverse.value[self.value]


def getReverse(arr):
    if arr == ArrowE.UP:
        return ArrowE.DOWN
    elif arr == ArrowE.DOWN:
        return ArrowE.UP
    elif arr == ArrowE.LEFT:
        return ArrowE.RIGHT
    else:
        return ArrowE.LEFT


class DetectionRange:
    def __init__(self, rowTop=0, rowBottom=0, colLeft=0, colRight=0, score=0):
        self.rowTop = rowTop
        self.rowBottom = rowBottom
        self.colLeft = colLeft
        self.colRight = colRight
        self.score = score

    def merge(self, detectionRange):
        self.rowTop = min(self.rowTop, detectionRange.rowTop)
        self.rowBottom = max(self.rowBottom, detectionRange.rowBottom)
        self.colLeft = min(self.colLeft, detectionRange.colLeft)
        self.colRight = max(self.colRight, detectionRange.colRight)
        self.score = self.score + detectionRange.score

    def __str__(self):
        return "DetectionRange [rowTop=" + str(self.rowTop) \
               + ", rowBottom=" + str(self.rowBottom) \
               + ", colLeft=" + str(self.colLeft) + \
               ", colRight=" + str(self.colRight) + \
               ", score=" + str(self.score) + "]"


def adjustRange(detectRange, top, bottom, left, right):
    detectRange.rowTop = min(detectRange.rowTop, top)
    detectRange.rowBottom = max(detectRange.rowBottom, bottom)
    detectRange.colLeft = min(detectRange.colLeft, left)
    detectRange.colRight = max(detectRange.colRight, right)


def calScore(num):
    if num >= 5:
        return 10
    elif num == 4:
        return 4
    elif num == 3:
        return 1
    else:
        return 0


class IllegalOperationError(Exception):
    def __str__(self):
        print("Illegal Operation.")


class XiaoXiaoLeGame:
    def __init__(self, K=0, M=0, N=0, drawMode=False):
        self.n_type = K
        self.row = M
        self.col = N
        # self.step = S
        self.chessboard = [[random.randint(1, self.n_type) for n in range(self.col)] for m in range(self.row)]
        self.totalScore = 0
        self.drawMode = drawMode
        self.chessboardStatus = []
        if self.drawMode is True and self.n_type != 0 and self.row != 0 and self.col != 0:
            self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard))
        # 对随机生成的棋盘，可能一开始就会有可消除的
        self.totalScore += self.detectAndAdjustAllUntilNonadjustable()

        print('chessboard:')
        for row in self.chessboard:
            print(row)

    def traverse(self):
        self.row, self.col = self.col, self.row
        self.chessboard = np.array(self.chessboard).T.tolist()
        print('chessboard:')
        for row in self.chessboard:
            print(row)
        # detectRange = DetectionRange(0, self.row - 1, 0, self.col - 1)
        # marks = np.zeros((self.row, self.col))
        # for i in range(self.row):
        #     for j in range(self.col):
        #         if self.getChessVal(i, j) == 0:
        #             marks[i, j] = 1
        # self.eliminateAndDropDownAdjust(detectRange, marks)

        # 转置后的棋盘，可能会有可消除的
        # self.totalScore += self.detectAndAdjustAllUntilNonadjustable()

    def getScore(self):
        return self.totalScore

    def setDrawMode(self, drawMode):
        if self.drawMode is True:
            return
        self.drawMode = drawMode
        if self.drawMode is True and self.n_type != 0 and self.row != 0 and self.col != 0:
            self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard))

    def printChessboard(self):
        print('chessboard:')
        for row in self.chessboard:
            print(row)

    def setChessboard(self, newChessboard, n_type):
        self.chessboard = deepcopy(newChessboard)
        self.n_type = n_type
        self.row = len(self.chessboard)
        self.col = len(self.chessboard[0])
        # self.step = n_steps
        # self.printChessboard()
        self.chessboardStatus.clear()
        if self.drawMode is True and self.n_type != 0 and self.row != 0 and self.col != 0:
            self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard))

    def getRowNum(self):
        return self.row

    def getColNum(self):
        return self.col

    def getChessVal(self, row, col):
        return self.chessboard[row][col]

    def isPointInChessboard(self, rowIndex, colIndex):
        """
        判断点是否在棋盘内
        :param rowIndex:
        :param colIndex:
        :return: Boolean
        """
        if 0 <= rowIndex < self.row and 0 <= colIndex < self.col:
            return True
        return False
        #     done

    def dropColDownTo(self, row, col, offset):
        """
        根据offset对(row, col)开始往上的方向调整棋盘下落
        :param row:
        :param col:
        :param offset:
        :return:
        """
        if offset == 0:
            return

        while row - offset >= 0:
            # 保证不越界的情况下，将上方的元素移动下来
            self.chessboard[row][col] = self.chessboard[row - offset][col]
            row -= 1

        while row >= 0 and self.chessboard[row][col] != 0:
            # 如果 chessboard[row][col] 为0， 则上方的所有点都是0，提前跳出循环
            # 剩下的 chessboard[row][col] 上方的所有点都赋为0
            self.chessboard[row][col] = 0
            row -= 1
        #     done

    def countSameValuePointFrom(self, p, arrow):
        num = 0
        arrVal = arrow.value
        originalValue = self.chessboard[p.row][p.col]
        rowIndex = arrVal[3].row + p.row
        colIndex = arrVal[3].col + p.col
        while self.isPointInChessboard(rowIndex, colIndex) and self.chessboard[rowIndex][colIndex] != 0:
            if originalValue != self.chessboard[rowIndex][colIndex]:
                break
            num += 1
            rowIndex += arrVal[3].row
            colIndex += arrVal[3].col
        return num
        # done

    def autoDetectAndMarkFrom(self, p, marks):
        """
        检测点的四个方向是否有可以消除的
        :param p:
        :param marks:
        :return:
        """
        detectRange = DetectionRange(0, 0, self.col - 1, 0)
        row = p.row
        col = p.col
        increment = 0
        if marks[row, col] == 0 or marks[row, col] == ROW_MARK:
            # 未被消除或被标记为行消除，此时还可被列消除
            numUp = self.countSameValuePointFrom(p, ArrowE.UP)
            numDown = self.countSameValuePointFrom(p, ArrowE.DOWN)
            # print(str(p) + ", numUp: " + str(numUp) + ", numDown: " + str(numDown))
            colScore = calScore(numUp + numDown + 1)
            if colScore > 0:
                adjustRange(detectRange, 0, row + numDown, col, col)
                r = row - numUp
                while r <= row + numDown:
                    marks[r, col] += COL_MARK
                    r += 1
            increment += colScore

        if marks[row, col] == 0 or marks[row, col] == COL_MARK:
            # 未被消除或被标记为列消除，此时还可被行消除
            numLeft = self.countSameValuePointFrom(p, ArrowE.LEFT)
            numRight = self.countSameValuePointFrom(p, ArrowE.RIGHT)
            # print(str(p) + ", numLeft: " + str(numLeft) + ", numRight: " + str(numRight))
            rowScore = calScore(numLeft + numRight + 1)
            if rowScore > 0:
                adjustRange(detectRange, 0, row, col - numLeft, col + numRight)
                c = col - numLeft
                while c <= col + numRight:
                    marks[row, c] += ROW_MARK
                    c += 1
            increment += rowScore
        # 更新分数
        detectRange.score = detectRange.score + increment
        return detectRange
        # done

    def eliminateAndDropDownAdjust(self, detectRange, marks):
        """
        根据标记找到需要调整的列
        :param detectRange:
        :param marks:
        :return:
        """
        c = detectRange.colLeft
        while c <= detectRange.colRight:
            # 对每一列进行调整，从左到右
            r = detectRange.rowTop
            while r <= detectRange.rowBottom:
                # 计算被消除的列长度，一行一行找，从上到下
                originalR = r
                while r <= detectRange.rowBottom and marks[r, c] != 0:
                    # 取消标记
                    marks[r, c] = 0
                    r += 1

                self.dropColDownTo(r - 1, c, r - originalR)
                r += 1
            c += 1

    #    done

    def detectAndAdjustFrom2ExchangedPoints(self, p1, p2):
        """
        对两个点检测周围是否可消除并调整
        :param p1:
        :param p2:
        :return:
        """
        detectRange = DetectionRange()
        # 默认所有数值为0
        # 初始化marks
        marks = np.zeros((self.row, self.col))
        detectRange.merge(self.autoDetectAndMarkFrom(p1, marks))
        detectRange.merge(self.autoDetectAndMarkFrom(p2, marks))

        # print("detect from " + str(p1) + " and " + str(p2) + "\nget range " + str(detectRange))

        # 如果没有分数，那么就没有标记，这里就不会调整
        if self.drawMode is True and detectRange.score != 0:
            # if detectRange.score == 0:
            #     self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard))
            # else:
            # 标记
            self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard, marks=marks))
            self.eliminateAndDropDownAdjust(detectRange, marks)
            self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard))
        else:
            self.eliminateAndDropDownAdjust(detectRange, marks)

        return detectRange.score

    #     done

    def detectAndAdjustAllOneTime(self):
        """
        对棋盘中每个点，判断是否可消除
        :return:
        """
        # 默认所有数值为0
        marks = np.zeros((self.row, self.col))

        # 检测可以消除的行列，并确定最小需要调整范围
        detectionRange = DetectionRange()
        # 棋盘上方都是0，因而可以从下往上检查，以减少对0的点的检查
        # print('current chessboard:')
        # self.printChessboard()
        row = self.row - 1
        while row >= 0:
            col = 0
            zeroCount = 0
            while col < self.col:
                # 对每个点进行一次消除检测
                if self.chessboard[row][col] == 0:
                    zeroCount += 1
                    col += 1
                    continue
                detectionRange.merge(self.autoDetectAndMarkFrom(Point(row, col), marks))
                col += 1
            if zeroCount == self.col:
                break
            row -= 1
        if detectionRange.score > 0:
            if self.drawMode is True:
                self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard, marks=marks))
            # 同时消除所有需要消除的元素并同时进行下落操作
            self.eliminateAndDropDownAdjust(detectionRange, marks)
            if self.drawMode is True:
                self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard))

        return detectionRange.score

    #         done

    def detectAndAdjustAllUntilNonadjustable(self):
        """
        一直自动检测并调整，直到再也没有找到可以自动消除的
        :return:
        """
        score = 0
        while True:
            increment = self.detectAndAdjustAllOneTime()
            if increment == 0:
                break
            score += increment

        return score

    #         done
    def checkPointInChessboard(self, p):
        if not self.isPointInChessboard(p.getRow(), p.getCol()):
            raise IllegalOperationError

    def isPointZero(self, p):
        return self.chessboard[p.getRow()][p.getCol()] == 0

    def checkPointNotZero(self, p):
        if self.isPointZero(p):
            raise IllegalOperationError

    def is2PointClose(self, p1, p2):
        if p1.getRow() == p2.getRow():
            return abs(p1.getCol() - p2.getCol()) == 1
        else:
            return (p1.getCol() == p2.getCol()) and (abs(p1.getRow() - p2.getRow()) == 1)

    def check2PointClose(self, p1, p2):
        if not self.is2PointClose(p1, p2):
            raise IllegalOperationError

    def exchange(self, p1, p2):
        """
        交换两个点，计算分数
        :param p1:
        :param p2:
        :return:
        """
        self.checkPointInChessboard(p1)
        self.checkPointInChessboard(p2)
        self.checkPointNotZero(p1)
        self.checkPointNotZero(p2)
        self.check2PointClose(p1, p2)

        self.chessboard[p1.row][p1.col], self.chessboard[p2.row][p2.col] = self.chessboard[p2.row][p2.col], \
                                                                           self.chessboard[p1.row][p1.col]  # swap
        # print('in exchange(self, p1, p2) current chessboard:')
        # self.printChessboard()
        if self.drawMode is True:
            self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard, arrowPoint1=p1, arrowPoint2=p2))
        increment = self.detectAndAdjustFrom2ExchangedPoints(p1, p2)
        if increment == 0:
            # 没分就恢复原样
            # print('cancel')
            self.chessboard[p1.row][p1.col], self.chessboard[p2.row][p2.col] = self.chessboard[p2.row][p2.col], \
                                                                               self.chessboard[p1.row][p1.col]  # swap
            if self.drawMode is True:
                self.chessboardStatus.append(ChessboardPlotWidget(self.chessboard))
        increment += self.detectAndAdjustAllUntilNonadjustable()
        self.totalScore += increment
        return increment
        #         done

    def exchangeWith(self, row, col, arrow):
        arrVal = arrow.value
        # print('exchanging (' + str(row) + ',' + str(col) + ') ' + arrow.name)
        score = self.exchange(Point(row, col), Point(row + arrVal[3].getRow(), col + arrVal[3].getCol()))
        return score


def getIcon(chessVal, size):
    iconPath = r'E:\PythonProject\Algorithm Design and Analysis\xiaoXiaoLe\icon'
    # return image.imread('icon_'+str(self.getChessVal(row, col))+'.png')
    # img = img.resize((width, height), Image.ANTIALIAS)
    if chessVal == 0:
        return Image.new("RGB", (size, size), 'grey')
    return Image.open(iconPath + '\\gem' + str(chessVal) + '.png')


def countSameValuePointFrom(chessboard, p, arrow):
    num = 0
    arrVal = arrow.value
    originalValue = chessboard[p.row][p.col]
    rowIndex = arrVal[3].row + p.row
    colIndex = arrVal[3].col + p.col
    while 0 <= rowIndex < len(chessboard) and 0 <= colIndex < len(chessboard[0]) and chessboard[rowIndex][
        colIndex] != 0:
        if originalValue != chessboard[rowIndex][colIndex]:
            break
        num += 1
        rowIndex += arrVal[3].row
        colIndex += arrVal[3].col
    return num


def drawWidget(path, chessboardPlotWidget, imageNum):
    # fig = plt.figure()
    # for i in range(self.row):
    #     for j in range(self.col):
    #         icon = self.getIcon(i, j)
    #         plt.subplot(self.row, self.col, i*self.col + j + 1)
    #         plt.axis('off')
    #         plt.xticks([])  # 去掉横坐标值
    #         plt.yticks([])  # 去掉纵坐标值
    #         # plt.title('PCA')
    #         plt.imshow(icon)
    #         # LDA
    #         # plt.subplot(2, 4, j + 5)
    #         # plt.axis('off')
    #         # plt.xticks([])  # 去掉横坐标值
    #         # plt.yticks([])  # 去掉纵坐标值
    #         # plt.title('LDA')
    #         # plt.imshow(img)
    # plt.tight_layout()
    # plt.savefig('E:\\机器学习\\实验报告\\实验2\\eigenface与fisherface(pinv).png')
    # plt.show()

    # 上下边框：20，左右边框：15，间隔10
    # 首先要知道icon的大小，一般为正方形图标，这里默认为46
    row = chessboardPlotWidget.row
    col = chessboardPlotWidget.col
    size = 64
    image = Image.new("RGB", (2 * 15 + (size + 10) * col - 10,
                              2 * 20 + (size + 10) * row - 10), 'white')
    font = ImageFont.truetype("simhei.ttf", 5, encoding="utf-8")
    # draw.text((x, y), 'original', "black")
    y = 20
    for i in range(row):
        x = 15
        for j in range(col):
            # img = Image.fromarray(train_data[0].reshape((size, size)))
            icon = getIcon(chessboardPlotWidget.chessboard[i][j], size)
            image.paste(icon, (x, y))
            x += 10 + size
        y += 10 + size

    draw = ImageDraw.Draw(image)  # image1的画笔工具
    if chessboardPlotWidget.arrowPoint1 is not None and chessboardPlotWidget.arrowPoint2 is not None:
        p1 = chessboardPlotWidget.arrowPoint1
        p2 = chessboardPlotWidget.arrowPoint2
        if p1.row == p2.row:
            if p1.col < p2.col:
                x1 = 3 + (size + 10) * (p1.col + 1)
                x2 = 18 + (size + 10) * p2.col
                y = 20 + size // 2 + p1.row * (size + 10)
                draw.line((x1, y, x2, y), 'red', width=10)
            else:
                x1 = 3 + (size + 10) * (p2.col + 1)
                x2 = 18 + (size + 10) * p1.col
                y = 20 + size // 2 + p1.row * (size + 10)
                draw.line((x1, y, x2, y), 'red', width=10)
        elif p1.col == p2.col:
            if p1.row < p2.row:
                y1 = 8 + (size + 10) * (p1.row + 1)
                y2 = 22 + (size + 10) * p2.row
                x = 15 + size // 2 + p1.col * (size + 10)
                draw.line((x, y1, x, y2), 'red', width=10)
            else:
                y1 = 8 + (size + 10) * (p2.row + 1)
                y2 = 22 + (size + 10) * p1.row
                x = 15 + size // 2 + p1.col * (size + 10)
                draw.line((x, y1, x, y2), 'red', width=10)
    if chessboardPlotWidget.marks is not None:
        rectWidgetList = []
        # startFlag = 0
        # endFlag = 1
        startPoint = Point(0, 0)
        endPoint = Point(0, 0)
        for i in range(row):
            for j in range(col):
                numRight = 0
                numDown = 0
                if chessboardPlotWidget.marks[i][j] != 0:
                    numRight = 1 + countSameValuePointFrom(chessboardPlotWidget.chessboard, Point(i, j), ArrowE.RIGHT)
                    numDown = 1 + countSameValuePointFrom(chessboardPlotWidget.chessboard, Point(i, j), ArrowE.DOWN)
                    startPoint.setPoint(15 + i * (size + 10), 10 + j * (size + 10))  # row, col
                if numRight < 3:
                    numRight = 0
                else:
                    endPoint.setPoint(15 + (i + 1) * (size + 10), 10 + (j + numRight) * (size + 10))
                    rectWidgetList.append(RectWidget(startPoint, endPoint))
                if numDown < 3:
                    numDown = 0
                else:
                    endPoint.setPoint(15 + (i + numDown) * (size + 10), 10 + (j + 1) * (size + 10))
                    rectWidgetList.append(RectWidget(startPoint, endPoint))
                while numRight > 0:
                    chessboardPlotWidget.marks[i][j + numRight - 1] -= 1
                    numRight -= 1
                while numDown > 0:
                    chessboardPlotWidget.marks[i + numDown - 1][j] -= 2
                    numDown -= 1
        for rectWidget in rectWidgetList:
            # row和col与x和y是相反的
            draw.rectangle(
                (rectWidget.p1.getCol(), rectWidget.p1.getRow(), rectWidget.p2.getCol(), rectWidget.p2.getRow()),
                outline='red', width=3)
    # 画矩形
    # 。。。
    # image.save(path + 'example_' + str(imageNum) + '.png', quality=95)
    image.save(path + '\\example_' + str(imageNum) + '.png')
    image.show()


def drawFigure(path, chessboardPlotWidgetList):
    imageNum = 0
    for chessboardPlotWidget in chessboardPlotWidgetList:
        drawWidget(path, chessboardPlotWidget, imageNum)
        imageNum += 1


class TipsPoint(Point):
    def __init__(self, row, col, arrow):
        super().__init__(row, col)
        self.arrow = arrow

    def __str__(self):
        return '(' + str(self.row) + ', ' + str(self.col) + '), ' + self.arrow.name


class Player:
    def __init__(self, xiaoXiaoLe, stepsLeft, mode=1, w=0.5):
        self.tips = []
        self.xiaoXiaoLe = deepcopy(xiaoXiaoLe)
        self.stepsLeft = stepsLeft
        self.totalScore = 0
        self.scoreDict = {}
        self.mode = mode
        self.w = w

    # def initializeScoreParam(self):

    @time_count
    def play(self):
        self.totalScore = self.getMaxScore(self.xiaoXiaoLe, self.stepsLeft, self.tips)

    def exchangeRight(self, xiaoXiaoLe, r, c, stepsLeft, tips):
        return self.exchange(xiaoXiaoLe, r, c, stepsLeft, ArrowE.RIGHT, tips)

    def exchangeUp(self, xiaoXiaoLe, r, c, stepsLeft, tips):
        return self.exchange(xiaoXiaoLe, r, c, stepsLeft, ArrowE.UP, tips)

    def exchange(self, xiaoXiaoLe, r, c, stepsLeft, arrow, tips):
        increasement = 0
        try:
            tips.append(TipsPoint(r, c, arrow))
            increasement += xiaoXiaoLe.exchangeWith(r, c, arrow)

            if self.mode == 1 and increasement != 0:
                increasement += self.getMaxScore(xiaoXiaoLe, stepsLeft - 1, tips)
            if self.mode == 2 and increasement != 0:
                name = str(xiaoXiaoLe.chessboard) + str(stepsLeft - 1)
                if name in self.scoreDict.keys():
                    increasement += self.scoreDict[name]
                else:
                    self.scoreDict[name] = self.getMaxScore(xiaoXiaoLe, stepsLeft - 1, tips)
                    increasement += self.scoreDict[name]
            if self.mode == 3 and increasement != 0:
                if random.random() < self.w:
                    increasement += self.getMaxScore(xiaoXiaoLe, stepsLeft - 1, tips)
        except IllegalOperationError:
            pass
            # print('IllegalOperationError occurred')
        finally:
            if increasement == 0:
                tips.pop()
        return increasement

    def getMaxScore(self, xiaoXiaoLe, stepsLeft, tips):
        if stepsLeft == 0:
            return 0
        maxScore = 0
        tipsPicked = []
        r = xiaoXiaoLe.getRowNum() - 1
        # 遍历棋盘，分别向左和向上交换，找到最大分数
        # self.xiaoXiaoLe.printChessboard()
        while r >= 0:
            c = 0
            while c < xiaoXiaoLe.getColNum():
                if xiaoXiaoLe.getChessVal(r, c) == 0:
                    c += 1
                    continue
                # else:
                # print('For point (' + str(r) + ', ' + str(c) + '):')
                tipsRight = []
                tipsUp = []
                xiaoXiaoLeRight = deepcopy(xiaoXiaoLe)
                xiaoXiaoLeUp = deepcopy(xiaoXiaoLe)
                scoreRight = self.exchangeRight(xiaoXiaoLeRight, r, c, stepsLeft, tipsRight)
                scoreUp = self.exchangeUp(xiaoXiaoLeUp, r, c, stepsLeft, tipsUp)
                maxIncrement = max(scoreRight, scoreUp)
                maxScore = max(maxIncrement, maxScore)
                if maxScore == maxIncrement:
                    if maxScore == scoreRight:
                        if scoreRight == scoreUp:
                            tipsPicked = tipsRight if len(tipsRight) < len(tipsUp) else tipsUp
                        else:
                            tipsPicked = tipsRight
                    else:
                        tipsPicked = tipsUp
                c += 1
            r -= 1
        tips.extend(tipsPicked)
        return maxScore


class ChessboardPlotWidget:
    def __init__(self, chessboard, marks=None, arrowPoint1=None, arrowPoint2=None):
        self.chessboard = deepcopy(chessboard)
        self.row = len(self.chessboard)
        self.col = len(self.chessboard[0])
        self.arrowPoint1 = None
        self.arrowPoint2 = None
        if arrowPoint1 is not None and arrowPoint2 is not None:
            self.arrowPoint1 = deepcopy(arrowPoint1)
            self.arrowPoint2 = deepcopy(arrowPoint2)

        if marks is not None:
            self.marks = deepcopy(marks)
        else:
            self.marks = None


class RectWidget:
    def __init__(self, p1, p2):
        self.p1 = deepcopy(p1)
        self.p2 = deepcopy(p2)


class ManualPlayer:
    def __init__(self, xiaoXiaoLe, drawMode=False):
        self.tips = []
        self.xiaoXiaoLe = deepcopy(xiaoXiaoLe)
        self.xiaoXiaoLe.setDrawMode(drawMode)

    def setDrawMode(self, drawMode):
        self.xiaoXiaoLe.setDrawMode(drawMode)

    def playWithTips(self, tips):
        for tip in tips:
            self.xiaoXiaoLe.exchangeWith(tip.getRow(), tip.getCol(), tip.arrow)
            self.xiaoXiaoLe.printChessboard()

    def getStatus(self):
        return self.xiaoXiaoLe.chessboardStatus


# up = ArrowE.UP
# print(getReverse(up).value[1])
# print(ArrowE.arrowDictReverse.value.keys())
# print(ArrowE.arrowDictReverse.value[ArrowE.UP.value])
# for data in ArrowE.arrowDictReverse.
# print()

if __name__ == '__main__':
    chessboard1 = [[3, 3, 4, 3],
                   [3, 2, 3, 3],
                   [2, 4, 3, 4],
                   [1, 3, 4, 3],
                   [3, 3, 1, 1],
                   [3, 4, 3, 3],
                   [1, 4, 4, 3],
                   [1, 2, 3, 2]
                   ]
    # myXiaoXiaoLe = XiaoXiaoLeGame(3, 10, 10)

    myXiaoXiaoLe = XiaoXiaoLeGame()
    myXiaoXiaoLe.setChessboard(chessboard1, 4)
    player = Player(myXiaoXiaoLe, 4, mode=2, w=0.5)
    print('time cost:')
    player.play()
    print()
    print('result:')
    myXiaoXiaoLe.printChessboard()
    print(player.totalScore)
    for tip in player.tips:
        print(tip)
    print('done')
    manualPlayer = ManualPlayer(myXiaoXiaoLe)
    manualPlayer.setDrawMode(True)
    manualPlayer.playWithTips(player.tips)
    path = r'E:\算法设计与分析实验\实验3\example'
    drawFigure(path, manualPlayer.getStatus())

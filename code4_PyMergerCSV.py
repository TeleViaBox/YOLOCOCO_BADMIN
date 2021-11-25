import torch
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List


def cocktail_sort(nums1: List[float], nums2 : List[float]) -> List[float]:
    swap = True
    start = 0
    end = len(nums1) - 1
    while swap:
        swap = False
        for i in range(start, end):
            if nums1[i] > nums1[i + 1]:
                nums1[i], nums1[i + 1] = nums1[i + 1], nums1[i]
                nums2[i], nums2[i + 1] = nums2[i + 1], nums2[i]
                swap = True

        if not swap: break
        swap = False
        end -= 1

        for i in range(end, start, -1):
            if nums1[i] < nums1[i - 1]:
                nums1[i], nums1[i - 1] = nums1[i - 1], nums1[i]
                nums2[i], nums2[i - 1] = nums2[i - 1], nums2[i]
                swap = True

        start += 1

    return nums1, nums2


mpl.use('TkAgg')
NUM = 10
RxLIST = np.arange(NUM, dtype=np.float)
RyLIST = np.arange(NUM, dtype=np.float)
LxLIST = np.arange(NUM, dtype=np.float)
LyLIST = np.arange(NUM, dtype=np.float)
fifthLIST = np.arange(NUM, dtype=np.float)
sixthLIST = np.arange(NUM, dtype=np.float)
newLxLIST = [0] * NUM  # np.arange(NUM, dtype=np.float)
newRxLIST = [0] * NUM  # np.arange(NUM, dtype=np.float)
newLyLIST = [0] * NUM  # np.arange(NUM, dtype=np.float)
newRyLIST = [0] * NUM  # np.arange(NUM, dtype=np.float)
start = 24
for index in range(start, 26):
    print(index)
    # index = 18
    ufolder = 'B' + str(index)
    print(ufolder)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # num of input photos
    # aLIST = np.arange(NUM, dtype=np.float)
    # bLIST = np.arange(NUM, dtype=np.float)
    i = 1
    while i < NUM:
        img = cv2.imread('./MYdatabase/' + str(ufolder)+'/1 (' + str( i ) + ').png')
        image = cv2.resize(img, (800, 480))
        results = model(image)
        results.print()
        print(results.xyxy[0])
        a = results.xyxy
        temp2 = a[0]
        # cv2.imshow('1 (' + str( i ) + ').png', np.squeeze(results.render()))
        cv2.destroyAllWindows()
        cv2.waitKey(0)
        print( str(ufolder)+ ' ' + '1 (' + str( i ) + ').png')
        RxLIST[i] = temp2[1, 1].item()
        LxLIST[i] = temp2[1, 2].item()
        RyLIST[i] = temp2[1, 3].item()
        LyLIST[i] = temp2[1, 4].item()
        fifthLIST[i] = temp2[1, 5].item()
        sixthLIST[i] = temp2[1, 5].item()
        i = i + 1
        print('hereA')
    # for j in range(1, NUM):
    print(index)
    if index == start:
        print('here2')
        newRxLIST = RxLIST.copy()
        newLxLIST = LxLIST.copy()
        newRyLIST = RyLIST.copy()
        newLyLIST = LyLIST.copy()
        print('here2.5')
        print(newRxLIST)
        print(RxLIST)
    else:
        # newRxLIST = newRxLIST + RxLIST
        # newLxLIST = newLxLIST + LxLIST
        # newRyLIST = newRyLIST + RyLIST
        # newLyLIST = newLyLIST + LyLIST
        print(newRxLIST)
        print(RxLIST)
        newRxLIST = np.hstack((newRxLIST, RxLIST))
        newRyLIST = np.hstack((newRyLIST, RyLIST))
        newLxLIST = np.hstack((newLxLIST, LxLIST))
        newLyLIST = np.hstack((newLyLIST, LyLIST))
        print(newRxLIST, 'D')
        # result = []
        # for element in RxLIST:
        #     result.append(element)
        # for element in newRxLIST:
        #     result.append(element)
        # print(result)
        # # result == newRxLIST
        print('here3')
        print(len(RxLIST))

        for k in range(len(newRxLIST)):
            newRxLIST[k] += newLxLIST[k]
            newRyLIST[k] += newLyLIST[k]
        print("Before sort:", newRxLIST, newRyLIST)

        Xs, Ys = cocktail_sort(newRxLIST, newRyLIST)

        print("After sort:", Xs, Ys)

    # newRyLIST.sort()
    # newLxLIST.sort()
    # newRxLIST.sort()
    print(newRxLIST, 'newRxLIST')
    # # calculate
    # aX[i] = (newRxLIST[i] + bX[i + 1] + bX[i + 2]) / 5
    # aY[i] = (bY[i] + bY[i + 1] + bY[i + 2] + ...!!) / 5
    # i = i + 5
    # CsValue[i] = abs(inDataX[i] - aX[i]) + abs(inDataY[i] - aY[i])

    # # B classifier
    # for i in range(1, 500):
    #     if (Bx > 440 & & Bx < 890 & & By > 250 & & 420)
    #         count = count + 1
    # if (count > 7)
    #     print('it is B')
    # count = 0
    # # D classifier
    # for i in range(1, 5):
    #     if CsValue[i] ==
    #         count + +
    # for i in range(6, 10):
    #     if CsValue[i] ==
    #         count + +
    # if (count > 7)
    #     print('it is D')
    # # A, C classifer: by error
    # for i in range(1, 10):
    #     costA = sum(CsValue - aX_A[i]) - sum(CSValue - aY_A[i])
    #     costC = sum(CsValue - aX_C[i]) - sum(CSValue - aY_C[i])
    #     if costA > costC
    #         print('it is C')
    #     else
    #         print('it is A')

    Me_x = np.array(RxLIST) + np.array(LxLIST)
    Me_y = np.array(RyLIST) + np.array(LyLIST)
    print(Me_x, 'Me_x')
    print(Me_y, 'Me_y')
    mpl.use('TkAgg')
    # Me_y = Me_y[1:]
    # Me_x = Me_x[1:]
    plt.scatter(Me_x, Me_y)
    plt.savefig(str(ufolder) + '.png')
    plt.plot(Me_x, Me_y)
    # plt.savefig(str(ufolder) + '_lined.png')
with open(str(ufolder)+ '_newLIST.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(['Rx', 'Lx', 'Ry', 'Ly', 'fifthLIST', 'sixthLIST'])
    writer.writerow(newRxLIST)
    writer.writerow(newLxLIST)
    writer.writerow(newRyLIST)
    writer.writerow(newLyLIST)
    # writer.writerow(fifthLIST)
    # writer.writerow(sixthLIST)
    # writer.writerow(['new'])

# # step1
# # cpystr() = We_Y
# # step2
# sorting
# # step3
# first5 a group
# # step4 Max Likelihood

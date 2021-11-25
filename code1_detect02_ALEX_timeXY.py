import torch
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

for index in range(1, 10):
    # index = 18
    ufolder = 'B' + str(index)
    print(ufolder)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    NUM = 10
    # num of input photos
    # aLIST = np.arange(NUM, dtype=np.float)
    # bLIST = np.arange(NUM, dtype=np.float)

    timeXY = np.arange(NUM, dtype=np.float)
    velY = np.arange(NUM, dtype=np.float)
    accY = np.arange(NUM, dtype=np.float)
    RxLIST = np.arange(NUM, dtype=np.float)
    RyLIST = np.arange(NUM, dtype=np.float)
    LxLIST = np.arange(NUM, dtype=np.float)
    LyLIST = np.arange(NUM, dtype=np.float)
    fifthLIST = np.arange(NUM, dtype=np.float)
    sixthLIST = np.arange(NUM, dtype=np.float)

    i = 1
    while i < NUM:
        img = cv2.imread('./MYdatabase/' + str(ufolder) + '/1 (' + str(i) + ').png')
        image = cv2.resize(img, (800, 480))
        results = model(image)
        results.print()
        print(results.xyxy[0])
        a = results.xyxy
        temp2 = a[0]
        # cv2.imshow('1 (' + str( i ) + ').png', np.squeeze(results.render()))
        cv2.destroyAllWindows()
        cv2.waitKey(0)
        print(str(ufolder) + ' ' + '1 (' + str(i) + ').png')
        RxLIST[i] = temp2[1, 1].item()
        LxLIST[i] = temp2[1, 2].item()
        RyLIST[i] = temp2[1, 3].item()
        LyLIST[i] = temp2[1, 4].item()
        fifthLIST[i] = temp2[1, 5].item()
        sixthLIST[i] = temp2[1, 5].item()
        i = i + 1
    with open(str(ufolder) + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Rx', 'Lx', 'Ry', 'Ly', 'fifthLIST', 'sixthLIST'])
        writer.writerow(RxLIST)
        writer.writerow(LxLIST)
        writer.writerow(RyLIST)
        writer.writerow(LyLIST)
        writer.writerow(fifthLIST)
        writer.writerow(sixthLIST)
        # writer.writerow(['new'])
    Me_x = np.array(RxLIST) + np.array(LxLIST)
    Me_y = np.array(RyLIST) + np.array(LyLIST)
    print(Me_x, 'Me_x')
    print(Me_y, 'Me_y')
    mpl.use('TkAgg')
    # Me_y = Me_y[1:]
    # Me_x = Me_x[1:]

    tLIST = [i for i in range(0, 10)]
    print(tLIST, 'tLIST')
    plt.figure(1)
    # plt.scatter(Me_x, Me_y)
    plt.plot(tLIST, Me_x)
    plt.savefig(str(ufolder) + '.png')
    f2 = plt.figure(2)
    # plt.plot(Me_x, Me_y)
    plt.plot(tLIST, Me_y)
    # plt.savefig(str(ufolder) + '_lined.png')

    dt = 1
    for vel in range(1, NUM-1):
        velY[vel] = (Me_y[vel] - Me_y[vel - 1]) / dt
    print(velY)
    for acc in range(1, NUM-2):
        accY[acc] = (velY[acc] - velY[acc - 1]) / dt
    print(accY)
    print(accY.size, 'accY.size')
    print(timeXY.size, 'timeXY.size')
    print(timeXY, 'timeXY')
    print(Me_y, 'Me_y')
    f3 = plt.figure(3)
    plt.plot(timeXY, Me_y)
    # plt.title("Connected Scatterplot points with line")
    # plt.xlabel("x")
    # plt.ylabel("sinx")
    f4 = plt.figure(4)
    plt.plot(timeXY, velY)
    f5 = plt.figure(5)
    plt.plot(timeXY, accY)

    # plt.show()

    # TODO: store vel, acc into CSV
plt.savefig(str(ufolder) + '_timeX.png')
plt.savefig(str(ufolder) + '_timeY.png')
plt.show()




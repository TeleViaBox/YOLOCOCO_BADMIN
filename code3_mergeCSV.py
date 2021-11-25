# import os
# import glob
# import pandas as pd
# # # REFERENCE:  https://www.lsbin.com/2960.html
# # path = "./MYdatabase/trymerge/"
# #
# # all_files = glob.glob(os.path.join(path, "data_*.csv"))
# # df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
# # df_merged = pd.concat(df_from_each_file, ignore_index=True)
# # df_merged.to_csv("merged.csv")
#
#
# # # import pandas as pd
# # # import glob
# # path = r'./MYdatabase/trymerge/'  # use your path
# # all_files = glob.glob(path + "/*.csv")
# # li = []
# # for filename in all_files:
# #     df = pd.read_csv(filename, index_col=None, header=0)
# #     li.append(df)
# # frame = pd.concat(li, axis=0, ignore_index=True)
# #
#
# '''
# Data:2017-07-13
# Auther;JXNU Kerwin
# Description:使用Pandas拼接多個CSV檔案到一個檔案（即合併）
# '''
# import pandas as pd
# import os
# Folder_Path = r'./MYdatabase/trymerge/'       #要拼接的資料夾及其完整路徑，注意不要包含中文
# SaveFile_Path = r'./MYdatabase/trymerge/'       #拼接後要儲存的檔案路徑
# SaveFile_Name = r'all.csv'            #合併後要儲存的檔名
# # 修改當前工作目錄
# os.chdir(Folder_Path)
# # 將該資料夾下的所有檔名存入一個列表
# file_list = os.listdir()
# # 讀取第一個CSV檔案幷包含表頭
# df = pd.read_csv(Folder_Path + file_list[0])   # 編碼預設UTF-8，若亂碼自行更改
# # 將讀取的第一個CSV檔案寫入合併後的檔案儲存
# df.to_csv(SaveFile_Path+ SaveFile_Name, encoding="utf_8_sig", index=False)
# # 迴圈遍歷列表中各個CSV檔名，並追加到合併後的檔案
# for i in range(1,len(file_list)):
#     df = pd.read_csv(Folder_Path + file_list[i])
#     df.to_csv(SaveFile_Path + SaveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+')

x = [1, 2, 3]
u = [4, 5, 6]

x = u
print(x)
for i in range(1,3):
    x = x + u
print(x)

import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])
D = np.hstack((A, B)) # horizontal stack
print(D)
# [1,1,1,2,2,2]
print(A.shape,D.shape)
# (3,) (6,)

# DD = np.array([[100][100]])
# for i in range(1,3):
#     for j in range(1,3):
#         DD[[i][1]] = 100
# print(DD, 'DD')


column, row = 3, 5
A = [[0]*row for _ in range(column)]
print(A)
# https://www.delftstack.com/zh-tw/howto/python/how-to-initiate-2-d-array-in-python/
# https://blog.csdn.net/w417950004/article/details/86253721
# https://www.796t.com/post/Ym5oeTA=.html




x1 =
y1 =
x2 =
y2 =
reference =
UP = 0
DOWN = 0
if x1 > Me_x & y1> Me_y:
    # right-up and left-bottom
    UP = UP + 1
elif x2 < Me_x & y2 > Me_y:
    # the other boundary
    DOWN = DOWN + 1
else:
    break
Q57 = 0
# # if 500 < & < 700
# Q57 = Q57 + 1
# if Q57 < 2 or 3
#     print("motion B")
# # then it is
# if 600 < Me_x & Me_x < 700 & 270 < Me_x < 320
if UP + DOWN > 7:
    print("the A or B motion")
    if max(Me_x) < reference:
        print(' motion A')
    else
        print('motion ')
elif UP + DOWN > 3:
    print("the motion C")
else
    print("the motion D")
# # Y: 0~250 & X: 0~500 & X: 700~900 & Y: 250~450

# # ACCURACY
# float
# yes/total
# testA1~6
# testB1~6
# print()


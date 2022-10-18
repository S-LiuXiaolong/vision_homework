import cv2
import numpy as np
img = cv2.imread("rice.jfif")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转换为灰度图

# 使用局部阈值的自适应阈值操作进行图像二值化
dst = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101, 1)
# res ,dst = cv2.threshold(gray,0 ,255, cv2.THRESH_OTSU)
# 形态学去噪
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))
# 开运算去噪
dst=cv2.morphologyEx(dst,cv2.MORPH_OPEN,element)
# 轮廓检测函数
contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
cv2.drawContours(dst,contours,-1,(120,0,0),2)


count=0 # 米粒总数
# 遍历找到的所有米粒
for cont in contours:
    # 计算包围性状的面积
    ares = cv2.contourArea(cont)
    # 过滤面积小于50的形状
    if ares<5:   
        continue
    if ares>1000:
        count+=1
    count+=1
    # 打印出每个米粒的面积
    print("{}-blob:{}".format(count,ares),end="  ") 
    # 提取矩形坐标（x,y）
    rect = cv2.boundingRect(cont) 
    # 打印坐标
    print("x:{} y:{}".format(rect[0],rect[1]))
    # 绘制矩形
    cv2.rectangle(img,rect,(0,0,255),1)
    # 防止编号到图片之外（上面）,因为绘制编号写在左上角，所以让最上面的米粒的y小于10的变为10个像素
    y=10 if rect[1]<10 else rect[1] 
    # 在米粒左上角写上编号
    cv2.putText(img,str(count), (rect[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1) 
    if ares>1000:
        cv2.putText(img,"Two here", (rect[0], y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1) 
    # print('编号坐标：',rect[0],' ', y)
print('个数',count)
 
cv2.namedWindow("imgshow", cv2.WINDOW_NORMAL)   #创建一个窗口
cv2.imshow('imgshow', img)    #显示原始图片（添加了外接矩形）
 
cv2.namedWindow("dst", cv2.WINDOW_NORMAL)   #创建一个窗口
cv2.imshow("dst", dst)  #显示灰度图
 
cv2.waitKey()

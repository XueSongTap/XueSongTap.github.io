---
layout: articles
title: houmo.ai day5
tags: intern
---

# houmo.ai day5

## matlab cameraCalibrator 输出参数

### 内参矩阵
cameraParams.IntrinsicMatrix


注：和opencv里面矩阵行列是不一样的

### 径向畸变：
cameraParams.RadialDistortion

径向畸变3个（k1,k2,k3）


### 切向畸变：
cameraParms.TangentialDistortion

切向畸变2个(p1,p2)


注：OpenCV中的畸变系数的排列（这点一定要注意k1，k2，p1，p2，k3）

### 相机外参数：

RotationMatrices，TranslationVectors


## 参数导出

### right

内参：
```
1018.55237604410	0	0
0	1018.52663537727	0
942.381384311739	548.833475463328	1
```

radialDistortion:
```
-0.334714897106180	0.0954037089247972
```

TangentialDistortion  
```
0	0
```

```python
mtx =  np.array([[1018.55237604410, 0.000000, 942.381384311739],
                 [0.000000, 1018.52663537727, 548.833475463328],
                 [0.000000, 0.000000, 1.000000]])
dist = np.array([-0.334714897106180,0.0954037089247972, 0, 0, 0.000000])
```


### left

内参：

```
1054.01458997620	0	0
0	1062.72708961751	0
915.166683577737	511.700573702185	1
```


radialDistortion:
```
-0.309862295063518	0.0695210936014637
```

TangentialDistortion  
```
0	0
```


```python
mtx =  np.array([[1054.01458997620, 0.000000, 915.166683577737],
                 [0.000000, 1062.72708961751, 511.700573702185],
                 [0.000000, 0.000000, 1.000000]])
dist = np.array([-0.309862295063518, 0.0695210936014637, 0, 0, 0.000000])
```

## 激光雷达-摄像头联合标定脚本

```python
import cv2
import numpy as np
#img = cv2.imread("img.jpg")
#size = img.shape
#https://www.delftstack.com/zh/howto/python/opencv-solvepnp/
image_points_2D = np.array([
(1150 ,1172 ),
(1260 ,1325 ),
(1175 ,1484 ),
(1065 ,1334 ),
(972  ,1200 ),
(1111 ,1403 ),
(1004 ,1605 ),
(862  ,1410 ),
(1714 ,597  ),
(2104, 1155 ),
(2227, 1329 ),
(2118, 1505 ),
(1998, 1339 )], dtype="double")

figure_points_3D = np.array([
(4.034783,-2.087860,-0.477486),                          
(3.935709,-2.277781,-0.720224),
(4.007325,-2.105677,-0.962215),
(4.095161,-1.958558,-0.718973),
(3.331,-1.51,-0.485),         
(3.23,-1.645,-0.737),         
(3.290,-1.486,-1.0) ,         
(3.381,-1.34,-0.74) ,         
(2.573023,-2.356089,0.366686),
(2.575,-3.185,-0.467),
(2.44,-3.33,-0.725)  ,
(2.55,-3.17,-1.015) ,
(2.695,-3.096,-0.724)
])

#distortion_coeffs = np.zeros((4,1))
distortion_coeffs = np.array([-0.325893908628975,0.109455293617572,0.001132683883665,-0.0009245260801665216,-0.017367024013193])
#focal_length = size[1]
#center = (size[1]/2, size[0]/2)
matrix_camera = np.array(
                         [
                         [1984.33462308811, 0, 1890.36728700865],
                         [0, 1986.49403470224, 1046.81436326671],
                         [0, 0, 1]], 
                         dtype = "double"
                         )
#1. cv2.SOLVEPNP_ITERATIVE=0
#2、cv2.SOLVEPNP_EPNP=1
#3、cv2.SOLVEPNP_P3P=2
#4、cv2.SOLVEPNP_DLS=3
#5、cv2.SOLVEPNP_UPNP=4
#6、cv2.SOLVEPNP_AP3P=5
success, vector_rotation, vector_translation = cv2.solvePnP(figure_points_3D, image_points_2D, matrix_camera, distortion_coeffs, flags=cv2.SOLVEPNP_DLS)
print("success flag is:",success)
print("vector_rotation is:",vector_rotation)
print("vector_translation is:",vector_translation)

#R需要转换成旋转矩阵，T不需要转换
#生成的R旋转矩阵，直接用，不需要转置
#RT 结果选 SOLVEPNP_EPNP、SOLVEPNP_UPNP、SOLVEPNP_DLS结果差不多， 
#P3P and AP3P not working, 参数形式不对
R, _ = cv2.Rodrigues(vector_rotation)
T, _ = cv2.Rodrigues(vector_translation)
print("R is:",R)
print("T is:",T)

if 0:
    nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), vector_rotation, vector_translation, matrix_camera, distortion_coeffs)
    for p in image_points_2D:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    point1 = ( int(image_points_2D[0][0]), int(image_points_2D[0][1]))
    
    point2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    
    cv2.line(img, point1, point2, (255,255,255), 2)
    
    cv2.imshow("Final",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```
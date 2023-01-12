---
layout: articles
title: houmo.ai day8
tags: intern
---


# 改用undistort的二维图片进行标定尝试

## left



结果

```python
$ python opencv_pnp_extrinsic_left.py
success flag is: True
vector_rotation is: [[ 1.07354498]
 [-1.13650777]
 [ 1.21627133]]
vector_translation is: [[-0.42826049]
 [ 0.55470646]
 [ 1.7962894 ]]
R is: [[ 0.01224004 -0.99807654 -0.06077343]
 [ 0.12822949  0.06184291 -0.98981445]
 [ 0.99166899  0.00432242  0.1287398 ]]
T is: [[-0.28320717 -0.95900883 -0.00978603]
 [ 0.78650932 -0.23808078  0.56984263]
 [-0.54881398  0.15368672  0.82169557]]
```


matlab
```
worldOrientation =

    0.0363   -0.9977    0.0566
    0.0014   -0.0565   -0.9984
    0.9993    0.0363   -0.0007


worldLocation =

   -0.9024   -0.4617    0.2545
```

## opencv undistort 源码部分

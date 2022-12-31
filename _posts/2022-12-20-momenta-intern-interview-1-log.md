

## 姿态表示有哪些

## 四元数求逆

## slam用什么算法

## 卡尔曼滤波

## imu高通 低通分别是什么
在姿态的估计中，可以使用的传感器有加速度计，磁传感器和陀螺仪。这三种传感器的特点是，加速度计和磁传感器的测量值噪声较大，但是误差不会随着时间累计；而陀螺仪的噪声比较小，但是在较长的时间对角速度进行积分会有累计误差。从频域的角度来看，加速度计和磁传感器的低频部分是比较有用的，而陀螺仪的高频部分是比较有用的，因此可以使用低通滤波和高通滤波来获得需要的信息。
## 旋转矩阵

## c++重载

## eigen库熟悉吗

## lambda表达式

## 左值 右值 move



## 业务场景题目


```cpp

标题
根据已知车位重排车位角点顺序

题目描述
struct Point{​
double x;​
double y;​
};​
bool reorderSlotCornersAccordingTheSlot(std::vector<Point>& reorder_slot_corners,const std::vector<Point>& according_slot_corners);

```

```cpp
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
struct Point{
    double x;
    double y;
};



class Reorder{
public:
    bool reorderSlotCornersAccordingTheSlot(std::vector<Point>& reorder_slot_corners,const std::vector<Point>& according_slot_corners){

        for (int i = 0;  i < 4; i ++) {
            double min_D = 1000;
            int min_Point = -1;
            int j = 0;
            for (; j < 4; j ++){
                int d = getDistance(reorder_slot_corners[i], according_slot_corners[j]);
                if (d < min_D) {
                    min_D = d;
                    min_Point = j;
                }
            }
            Point tmp = reorder_slot_corners[i];
            reorder_slot_corners[i] = reorder_slot_corners[j];
            reorder_slot_corners[j] = tmp;
        }
        // 误差值 
    };
private:
    double getDistance(Point a, Point b) {
        return sqrt((a.x - b.x) *(a.x - b.x) + (a.y-b.y)*(a.y-b.y));
    }
};

int main() {
    class Reorder reorder;

}






```
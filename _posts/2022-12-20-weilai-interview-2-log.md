---
layout: articles
title: 蔚来2面
tags: interview
---

## c++的多态指的哪些方面
动态多态


## sort 函数的实现

快排+插入排序



## 一道算法
```cpp

//给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
//请注意 ，必须在不复制数组的情况下原地对数组进行操作。
#include <vector>
#include <iostream>
using namespace std;


class MoveZero{
public:
    void move(vector<int>& a) {
        int n = a.size();
        int j = 0;
        for (int i =  0; i < n; i ++){
            if (a[i] != 0) {
                a[j] =  a[i];
                j ++;
            }
        }

        for (;j < n; j ++){
            a[j] = 0;
        }

    }
};

int main() {
    class MoveZero a;
    vector<int> input = {1, 0, 2, 4, 0, 8};

    a.move(input);
    

    for (int i = 0; i < input.size(); i ++) {
        cout << input[i] << " ";
    }

    cout << endl;

}
```
---
layout: articles
title: 7022. 限制条件下元素之间的最小绝对差问题
tags: leetcode 周赛 双指针 set lower_bound()
---

## 题目

https://leetcode.cn/problems/minimum-absolute-difference-between-elements-with-constraint/description/

给你一个下标从 0 开始的整数数组 nums 和一个整数 x 。

请你找到数组中下标距离至少为 x 的两个元素的 差值绝对值 的 最小值 。

换言之，请你找到两个下标 i 和 j ，满足 abs(i - j) >= x 且 abs(nums[i] - nums[j]) 的值最小。

请你返回一个整数，表示下标距离至少为 x 的两个元素之间的差值绝对值的 最小值 。


## 题解

```cpp
class Solution {
public:
    int minAbsoluteDifference(vector<int>& nums, int x) {
        int ans = INT_MAX, n = nums.size();
        set<int> s= {INT_MAX, INT_MIN / 2}; //哨兵

        for (int i = x; i < n; ++ i) {
            s.insert(nums[i-x]);
            int y = nums[i];
            auto it = s.lower_bound(y);
            cout << "i: " << i << " *it: " << *it << endl;
            ans = min(ans, min(*it - y, y -*--it));
        }

        return ans;
    }
};
```
在这段代码中，将两个哨兵元素 `INT_MIN / 2` 和 `INT_MAX` 插入到有序集合 `s` 中。这样做的目的是为了处理边界情况。

1. `INT_MIN / 2`：这个哨兵元素比数组中的任何元素都小，它的作用是确保在查找大于等于当前元素 `y` 的元素时，即使没有找到比 `y` 更大的元素，也能保证 `it` 不指向 `s` 的末尾，而是指向 `INT_MAX`。这样可以确保下面的计算 `*it - y` 的结果是有效的。
2. `INT_MAX`：这个哨兵元素比数组中的任何元素都大，它的作用是确保在查找小于等于当前元素 `y` 的元素时，即使没有找到比 `y` 更小的元素，也能保证 `--it` 不越界，而是指向 `INT_MIN / 2`。这样可以确保下面的计算 `y - *--it` 的结果是有效的。

通过插入这两个哨兵元素，可以保证在边界情况下，计算差值绝对值时不会出现错误。这是一种常用的技巧，在处理有序集合的边界情况时非常有用。

`*--it` 返回的是减完之后的元素，即当前迭代器 `it` 前移一位后所指向的元素。

在代码中的 `*--it` 操作是先对迭代器 `it` 进行前移操作 `--it`，然后再通过解引用操作 `*it` 获取前移后的元素值。

具体来说，在 `auto it = s.lower_bound(y);` 这行代码中，`it` 是一个指向 `s` 中大于等于 `y` 的元素的迭代器。然后，`--it` 操作会将 `it` 前移一位，指向 `s` 中小于等于 `y` 的元素。最后，`*--it` 操作会返回前移后的元素值。

所以，`*--it` 返回的是当前迭代器 `it` 前移一位后所指向的元素。


## 样例理解

#### 样例2

输入：nums = [5,3,2,10,15], x = 1

i: 1 *it: 5
i: 2 *it: 3
i: 3 *it: 2147483647
i: 4 *it: 2147483647



## lower_bound()

用于获取集合中任何元素的下限,该迭代器指向刚好大于val的下一个直接元素

https://cplusplus.com/reference/set/set/lower_bound/
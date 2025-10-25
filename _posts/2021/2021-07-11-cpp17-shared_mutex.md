---
layout: article
title: cpp17 新特性shared_mutex 读写锁
tags: cpp17 新特性 读写锁 shared_mutex
---


C++17引入了shared_mutex，可以实现读写锁

std::shared_mutex是C++17引入的一种共享互斥锁,它具有以下主要特征:

支持两种互斥的访问模式:独占(exclusive)和共享(shared)。

多个线程可以同时获得shared ownership,从而实现并发读。

但只能有一个线程可以获得exclusive ownership,从而对数据的修改是互斥的。

shared_mutex通过读写锁(shared_lock、unique_lock)进行访问控制。

std::shared_mutex适用于读多写少的场景,例如:

std::shared_mutex mutex;

// 写操作需要unique锁
void write_data() {
  std::unique_lock lock(mutex);
  // 修改数据
}

// 读操作需要shared锁
void read_data() {
  std::shared_lock lock(mutex);
  // 读数据
}
与std::mutex相比,std::shared_mutex可以提高并发读的效率。但实现也更复杂,有时效果可能不明显。

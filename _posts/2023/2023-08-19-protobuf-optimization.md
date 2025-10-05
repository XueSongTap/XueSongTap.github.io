---
layout: articles
title: Protobuf性能优化原理解析：二进制编码与序列化效率
tags: protobuf
---

### 优化方式
二进制编码：protobuf 使用紧凑的二进制格式进行数据编码，相比于文本格式（如 JSON、XML），二进制格式在存储和传输上更加高效。二进制编码不仅减少了数据的体积，还降低了序列化和反序列化的时间开销。

压缩算法：protobuf 提供了多种压缩算法，如 Varint 编码、ZigZag 编码等，用于对整数和布尔类型数据进行压缩。这些算法可以减小整数类型数据的存储空间，从而减少序列化和反序列化的时间和网络传输的带宽消耗。

字段标签和有限字段集：protobuf 使用字段标签来标识消息中的字段，而不是使用字段名称。通过使用数字标签，可以减小数据的体积，并且在序列化和反序列化时可以更快地定位和访问字段。此外，protobuf 还支持有限字段集，即只序列化消息中定义的字段，忽略未定义的字段，从而减小数据的体积和处理时间。

预分配空间：protobuf 在序列化和反序列化时，可以预先分配足够的空间来存储数据，避免频繁的内存分配和释放操作，提高性能。

编码器/解码器优化：protobuf 的编码器和解码器实现经过优化，使用了高效的算法和数据结构，以提高序列化和反序列化的速度。例如，使用位操作和缓冲区来处理数据，减少不必要的拷贝操作。

### 参考

https://www.bilibili.com/video/BV1Qh411Q7yY/

https://www.bilibili.com/video/BV1E7411q7QK/

https://zhuanlan.zhihu.com/p/561275099

https://zhuanlan.zhihu.com/p/53339153


https://www.bwangel.me/2022/03/01/variant_zigzag/
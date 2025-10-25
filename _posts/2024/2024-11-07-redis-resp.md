---
layout: article
title:  RESP协议 (Redis Serialization Protocol) 解析
tags: Redis
---

## RESP 协议概述

Redis Serialization Protocol (RESP) 是Redis客户端与服务器之间通信的标准协议。它被设计为一个平衡了多种需求的协议：

- **简单易实现**：降低客户端开发难度
- **解析高效**：能够快速处理大量命令
- **可读性强**：便于调试和理解

RESP本质上是一个二进制安全的序列化协议，可以编码多种数据类型，包括整数、字符串、数组等。尽管主要用于Redis，但也可用于其他客户端-服务器项目。

## RESP 协议演进

- **Redis 1.2**：首次引入RESP协议，使用是可选的
- **Redis 2.0**：RESP2成为标准通信协议
- **Redis 6.0**：引入RESP3（RESP2的超集），通过HELLO命令支持协议版本升级
- **当前状态**：Redis同时支持RESP2和RESP3，未来可能以RESP3为主

## 网络层与请求-响应模型

RESP协议通常基于TCP连接（默认端口6379）或Unix套接字使用。它主要遵循请求-响应模型：

1. 客户端发送命令（格式化为字符串数组）
2. 服务器处理命令并返回响应

特殊情况下，协议会变为推送模式：
- 订阅Pub/Sub频道时
- 执行MONITOR命令时
- RESP3中的Push类型事件

## RESP 数据类型

RESP使用首字节标识数据类型，每个部分以`\r\n`（CRLF）结束。

### RESP2 基本类型

1. **简单字符串** (Simple String)：`+OK\r\n`
   - 首字节为`+`
   - 不能包含CR或LF字符

2. **错误信息** (Error)：`-ERR unknown command\r\n`
   - 首字节为`-`
   - 通常格式为：`-ERROR_TYPE error message`

3. **整数** (Integer)：`:1000\r\n`
   - 首字节为`:`
   - 表示64位有符号整数

4. **批量字符串** (Bulk String)：`$5\r\nhello\r\n`
   - 首字节为`$`
   - 格式为`$长度\r\n数据\r\n`
   - 空值表示：`$-1\r\n`

5. **数组** (Array)：`*2\r\n$3\r\nGET\r\n$4\r\nname\r\n`
   - 首字节为`*`
   - 格式为`*元素数量\r\n元素1元素2...`
   - 空值表示：`*-1\r\n`

### RESP3 新增类型

RESP3引入了更多语义化的数据类型：

- **空值** (Null)：`_\r\n`
- **布尔值** (Boolean)：`#t\r\n`或`#f\r\n`
- **双精度浮点数** (Double)：`,3.14159\r\n`
- **大整数** (Big Number)：`(3492890328409238509324850943850943825024385\r\n`
- **批量错误** (Bulk Error)：`!21\r\nSYNTAX invalid syntax\r\n`
- **原样字符串** (Verbatim String)：`=15\r\ntxt:Some string\r\n`
- **映射** (Map)：`%2\r\n+first\r\n:1\r\n+second\r\n:2\r\n`
- **属性** (Attribute)：`|1\r\n+ttl\r\n:3600\r\n`
- **集合** (Set)：`~3\r\n:1\r\n:2\r\n:3\r\n`
- **推送** (Push)：`>2\r\n+message\r\n+hello\r\n`

## 前缀长度的优势

RESP协议最大的特点之一是使用**前缀长度**来传输数据，这带来几个重要优势：

1. **二进制安全**：可以传输任何字节序列，包括NULL字节和特殊字符
2. **高效解析**：接收方预先知道数据长度，无需扫描寻找终止符
3. **无需转义**：不需要对特殊字符进行编码/转义处理
4. **简单实现**：解析器可以采用单次读取操作，减少复杂度

例如，批量字符串"hello"的编码为：`$5\r\nhello\r\n`，解析器只需读取长度(5)，然后精确读取5个字节，就能完成一次无歧义的数据读取。

## 客户端-服务器通信示例

客户端命令（如`LLEN mylist`）被编码为RESP数组：

```
*2\r\n$4\r\nLLEN\r\n$6\r\nmylist\r\n
```

服务器回复（假设长度为48293）：

```
:48293\r\n
```

## AOF文件格式示例

Redis的AOF持久化文件实际上就是RESP格式的命令序列：

```
*2\r\n
$6\r\n
SELECT\r\n
$1\r\n
0\r\n
*7\r\n
$4\r\n
mset\r\n
$4\r\n
name\r\n
$6\r\n
yuming\r\n
$3\r\n
age\r\n
$2\r\n
22\r\n
$6\r\n
servsr\r\n
$13\r\n
redis-service\r\n
```

这个AOF文件包含了两个命令：
1. `SELECT 0` - 选择数据库0
2. `MSET name yuming age 22 servsr redis-service` - 设置多个键值对

RESP协议的简洁设计使得Redis能够高效处理大量命令，同时保持了协议的可读性和实现的简单性，是Redis高性能的重要基础之一。

*补充*： 感觉主要就是有一个前缀长度？
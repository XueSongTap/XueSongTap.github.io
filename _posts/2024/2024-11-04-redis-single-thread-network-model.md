---
layout: article
title:  Redis的单线程网络模型解析：高性能背后的设计
tags: Redis
---

Redis在6.0版本之前采用单线程网络模型，尽管后续版本引入了多线程处理，但很多生产环境仍在使用基于单线程模型的Redis 4.0/5.0。

众多业务证明，单线程网络模型足够处理大部分情况，本文深入解析Redis单线程架构的设计原理及其高性能背后的秘密。

## 单线程Reactor模型概述

Redis采用的是单线程Reactor模式结合IO多路复用技术，

1. 事件循环(Event Loop)：
- 核心组件，负责管理和分发所有事件
- 实现在ae.c中的aeEventLoop结构
- 基于epoll/kqueue/select机制
2. 事件处理：
- 连接接受事件：新客户端连接到达时触发acceptHandler
- 读事件：客户端发送命令时触发readQueryFromClient
- 写事件：向客户端发送响应时触发sendReplyToClient
3. 命令处理流水线：
- readQueryFromClient → 从socket读取数据
- processInputBuffer → 解析RESP协议
- processCommand → 验证并准备执行命令
- 命令处理函数(如setCommand) → 执行具体命令
- addReply → 生成响应到输出缓冲区
4. 响应发送机制：
- beforeSleep → 主动尝试发送所有待处理响应
- 如果无法全部发送，则注册写事件
- 写事件触发时调用sendReplyToClient继续发送

```gherkin
Redis单线程进程
+------------------------------------------------------------------------------+
|                                                                              |
|  +--------------------+                                                      |
|  |                    |                                                      |
|  |   Event Loop       |<--------------------------------------------+        |
|  |   (aeEventLoop)    |                                             |        |
|  |                    |                                             |        |
|  +--------+-----------+                                             |        |
|           |                                                         |        |
|           |  轮询事件                                               |        |
|           v                                                         |        |
|  +--------+-----------+    +---------------+    +----------------+  |        |
|  |                    |    |               |    |                |  |        |
|  |   Socket事件分发   +--->+ 连接请求事件  +--->+ acceptHandler  |  |        |
|  |                    |    |               |    |                |  |        |
|  +--------+-----------+    +---------------+    +----------------+  |        |
|           |                                            |             |        |
|           |                                            | 创建        |        |
|           |                                            v             |        |
|           |                                     +----------------+   |        |
|           |                                     |                |   |        |
|           |                                     |  Client对象    |   |        |
|           |                                     |                |   |        |
|           |                                     +----------------+   |        |
|           |                                                          |        |
|           |                                                          |        |
|           |  +---------------+    +------------------+               |        |
|           +->|               |    |                  |               |        |
|              | 客户端读事件  +--->+ readQueryFrom    |               |        |
|              |               |    | Client           +---+           |        |
|              +---------------+    |                  |   |           |        |
|                                   +------------------+   |           |        |
|                                                          |           |        |
|                                                          v           |        |
|                                              +----------------------+ |        |
|                                              |                      | |        |
|                                              | processInputBuffer   | |        |
|  +----------------+                          |                      | |        |
|  |                |                          +----------------------+ |        |
|  | beforeSleep    |                                     |             |        |
|  | (写入数据)     |                                     v             |        |
|  |                |                          +----------------------+ |        |
|  +-------+--------+                          |                      | |        |
|          ^                                   | processCommand       | |        |
|          |                                   |                      | |        |
|          |                                   +----------------------+ |        |
|          |                                                |           |        |
|          |                                                v           |        |
|          |                                   +----------------------+ |        |
|          |                                   |                      | |        |
|          |                                   | 命令处理函数         | |        |
|          |                                   | (如setCommand)       | |        |
|          |                                   |                      | |        |
|          |                                   +----------------------+ |        |
|          |                                                |           |        |
|          |                                                v           |        |
|          |                                   +----------------------+ |        |
|          |                                   |                      | |        |
|          +-----------------------------------+ addReply(生成响应)   | |        |
|                                              |                      | |        |
|                                              +----------------------+ |        |
|                                                                       |        |
|  +---------------+    +------------------+                            |        |
|  |               |    |                  |                            |        |
|  | 客户端写事件  +--->+ sendReplyToClient+----------------------------+        |
|  |               |    |                  |                                     |
|  +---------------+    +------------------+                                     |
|                                                                                |
+------------------------------------------------------------------------------+

  +----------------+       +------------------+
  |                |       |                  |
  | Client 1       | <---> | Redis Protocol   |
  |                |       | (RESP)           |
  +----------------+       |                  |
                           +------------------+
  +----------------+               ^
  |                |               |
  | Client 2       | <-------------+
  |                |
  +----------------+

  +----------------+
  |                |
  | Client N       | <-------------+
  |                |
  +----------------+
```

线程模型工作流程
1. 主事件循环不断轮询所有注册的文件描述符(FD)，检查是否有事件发生
2. 当检测到事件，调用相应的回调函数处理
3. 所有网络IO和命令处理都在同一个线程中完成
4. 每完成一轮事件处理，调用beforeSleep处理一些周期性任务
5. 非阻塞IO确保单个请求不会长时间占用CPU



## 单线程网络模型的关键实现

### 1. 客户端连接创建

当Redis服务启动时，会在指定端口监听连接请求，并注册接收连接的回调函数：

```c
// server.c
if (createSocketAcceptHandler(&server.ipfd, acceptTcpHandler) != C_OK) {
    serverPanic("Unrecoverable error creating TCP socket accept handler.");
}
```

当有新的客户端连接到来时：
1. `acceptTcpHandler`函数被调用，接受连接获得客户端socket
2. 通过`connCreateAcceptedSocket`将socket封装为`connection`对象
3. 调用`acceptCommonHandler`处理连接，并最终调用`createClient`创建客户端对象

```c
// networking.c
client *createClient(connection *conn) {
    client *c = zmalloc(sizeof(client));
    
    // 初始化客户端结构
    // ...
    
    // 注册socket可读事件的处理函数
    connSetReadHandler(conn, readQueryFromClient);
    
    // ...
    return c;
}
```

### 2. 命令读取与解析

当客户端发送命令时，事件循环检测到客户端socket可读，调用`readQueryFromClient`函数：

```c
void readQueryFromClient(connection *conn) {
    client *c = connGetPrivateData(conn);
    // 从socket读取数据到客户端的querybuf
    // ...
    
    // 处理输入缓冲区中的命令
    processInputBuffer(c);
}
```

在`processInputBuffer`中，Redis根据RESP协议解析命令并存储在`client->argv`中：

```c
void processInputBuffer(client *c) {
    // 解析命令
    // ...
    
    // 处理并重置客户端
    processCommandAndResetClient(c);
}
```

### 3. 命令执行

`processCommandAndResetClient`函数对命令进行各种检查，包括权限验证、命令是否存在或被禁用等：

```c
int processCommandAndResetClient(client *c) {
    // 命令合法性检查
    // ...
    
    // 执行命令
    call(c, CMD_CALL_FULL);
    
    // ...
    return C_OK;
}
```

`call`函数找到对应的命令处理函数并执行。例如，`SET`命令对应`setCommand`函数：

```c
void setCommand(client *c) {
    // 命令逻辑实现
    // ...
    
    // 添加响应
    addReply(c, shared.ok);
}
```

命令处理完成后，通过`addReply*`系列函数将响应存储在客户端的输出缓冲区中：
- `client->buf`：固定大小的缓冲区，用于小型响应
- `client->reply`：由内存块组成的链表，用于大型响应

### 4. 响应发送

Redis在每轮事件循环的`beforeSleep`回调中处理待发送的响应：

```c
void beforeSleep(struct aeEventLoop *eventLoop) {
    // ...
    
    // 处理有待发送数据的客户端
    handleClientsWithPendingWrites();
    
    // ...
}
```

`handleClientsWithPendingWrites`尝试将客户端输出缓冲区中的数据发送出去：

```c
int handleClientsWithPendingWrites(void) {
    listIter li;
    listNode *ln;
    
    // 遍历所有有待发送数据的客户端
    listRewind(server.clients_pending_write, &li);
    while((ln = listNext(&li))) {
        client *c = listNodeValue(ln);
        
        // 尝试写入数据
        writeToClient(c->conn, 0);
        
        // 如果数据未全部发送完，注册可写事件
        if (clientHasPendingReplies(c))
            connSetWriteHandler(c->conn, sendReplyToClient);
    }
    
    // ...
    return processed;
}
```

如果客户端socket的发送缓冲区已满（非阻塞IO导致的`EAGAIN`错误），Redis会注册一个可写事件回调函数`sendReplyToClient`，等待socket可写时再发送数据。

## Redis单线程模型的性能表现

以下是使用memtier_benchmark工具对Redis进行的性能测试结果：

```
$ memtier_benchmark -p 11002 -n 10000 --key-maximum=10000 --ratio=1:1

ALL STATS
=========================================================================
Type         Ops/sec     Hits/sec   Misses/sec      Latency       KB/sec 
-------------------------------------------------------------------------
Sets        61016.80          ---          ---      1.64000      4522.31 
Gets        61016.80     24260.28     36756.52      1.63500      3003.09 
Waits           0.00          ---          ---      0.00000          --- 
Totals     122033.60     24260.28     36756.52      1.63700      7525.40
```

这个测试显示Redis能够处理每秒超过12万次操作，平均延迟仅为1.64毫秒


## 单线程模型的优势与局限

### 优势
1. **简化设计**：没有复杂的多线程同步问题
2. **避免锁开销**：不需要考虑线程安全和锁竞争
3. **性能可预测**：单线程执行确保命令的原子性，无需额外同步
4. **充分利用CPU**：针对内存操作优化的代码，能够高效利用CPU

### 局限
1. **CPU密集型命令阻塞**：如`SORT`、`LRANGE`等复杂度较高的命令会阻塞整个服务
2. **单核心限制**：无法充分利用多核CPU资源
3. **网络IO瓶颈**：在高并发场景下，网络IO可能成为瓶颈

## 实践建议

1. **避免长时间运行的命令**：如`KEYS`、`FLUSHALL`等线上禁止使用，考虑使用`SCAN`等替代方案
2. **合理使用管道技术**：使用管道(Pipeline)批量发送命令，减少网络往返
3. **考虑多实例部署**：在多核心服务器上部署多个Redis实例，更好地利用CPU资源
4. **升级到Redis 6.0+**：如果网络IO是瓶颈，考虑升级到支持多线程网络模型的Redis 6.0+


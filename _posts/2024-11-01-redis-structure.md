---
layout: articles
title: Redis核心架构解析：从数据结构到主从复制
tags: Redis
---


Redis作为高性能的内存键值数据库，凭借其丰富的数据结构和灵活的功能设计，已成为现代应用架构中不可或缺的组件。本文将深入剖析Redis的整体架构设计，帮助开发者和架构师更好地理解Redis的内部工作原理。

## Redis核心特性概览

作为内存KV数据库，Redis提供了以下核心功能：

- 丰富的数据结构：string、list、hash、set、sorted-set、stream、geo、hyperloglog等
- 发布订阅机制：通过pubsub模块实现消息的发布与订阅
- 持久化方案：AOF和RDB两种数据持久化方式
- 内存管理：基于jemalloc实现高效的内存分配与回收

## Redis服务器核心架构

### 主体结构

Redis的核心是`redisServer`结构体，它包含了服务器运行所需的所有信息：

```c
struct redisServer {
    redisDb *db;        // 数据库数组
    int dbnum;          // 数据库数量
    
    aeEventLoop *el;    // 事件循环
    
    dict *commands;     // 命令表
    
    list *clients;      // 客户端列表
};
```

这个结构体是整个Redis服务器的核心，管理着数据库、事件循环、命令表和客户端连接。

### 数据存储架构

`redisServer->db`是Redis存储数据的地方，它包含`dbnum`个`redisDB`元素，每个元素对应一个数据库，支持Redis的多数据库特性。

```c
struct redisDb {
    dict *dict;     // 存储键值对的字典
    dict *expires;  // 存储键过期时间的字典
}
```

数据存储的核心是`dict`字典，其中键是SDS（Simple Dynamic String）字符串，值是`redisObject`结构体：

```c
struct redisObject {
    unsigned type:4;        // 类型，如STRING、LIST、HASH等
    unsigned encoding:4;    // 编码方式，如INT、RAW、HT等
    unsigned lru:LRU_BITS;  // 最近访问时间，用于LRU淘汰
    int refcount;           // 引用计数，用于内存管理
    void *ptr;              // 指向实际数据结构的指针
}
```

这种设计使Redis能够针对不同类型的数据使用最优的内部表示方式，既保证了接口统一，又能实现存储效率最大化。

## 网络模型

Redis采用IO多路复用网络模型，在`ae.h`和`ae.c`中封装了不同平台的IO多路复用接口，提供了跨平台的统一抽象。在Linux上，Redis使用epoll + 水平触发模式。

服务器初始化时的关键步骤：

```c
// 创建并监听服务端口
if (server.port != 0 && listenToPort(server.port,&server.ipfd) == C_ERR) {
    serverLog(LL_WARNING, "Failed listening on port %u (TCP), aborting.", server.port);
    exit(1);
}

// 创建事件循环
server.el = aeCreateEventLoop(server.maxclients+CONFIG_FDSET_INCR);

// 创建接受TCP连接的处理器
if (createSocketAcceptHandler(&server.ipfd, acceptTcpHandler) != C_OK) {
    serverPanic("Unrecoverable error creating TCP socket accept handler.");
}
```

当新的客户端连接到来时，会调用`acceptTcpHandler`处理连接，并创建`connection`结构体和`client`对象，同时注册读事件处理函数：

```c
if (conn) {
    connEnableTcpNoDelay(conn);
    if (server.tcpkeepalive)
        connKeepAlive(conn,server.tcpkeepalive);
    connSetReadHandler(conn, readQueryFromClient);
    connSetPrivateData(conn, c);
}
```

## 命令处理流程

Redis客户端由`client`结构体表示：

```c
struct client {
    connection *conn;    // 客户端连接
    redisDb *db;         // 当前选择的数据库
    sds querybuf;        // 读缓冲区
    struct redisCommand *cmd;  // 当前执行的命令
    int argc;            // 命令参数个数
    robj **argv;         // 命令参数数组
    list *reply;         // 回复链表
    char buf[PROTO_REPLY_CHUNK_BYTES];  // 固定长度输出缓冲区
};
```

命令处理流程：

1. 当客户端连接有数据可读时，调用`readQueryFromClient`函数读取数据
2. 调用`processInputBuffer`按照Redis协议解析命令
3. 通过`processCommandAndResetClient`执行命令
4. `processCommand`函数在`redisServer->commands`字典中查找对应的命令处理函数
5. 调用`call`函数执行命令的处理函数`cmd->proc`
6. 使用`addReply*`系列函数生成命令响应并写入客户端的输出缓冲区

响应数据会先写入`client->buf`，如果空间不足，则写入`reply`链表。在事件循环的`beforeSleep`回调中，遍历客户端列表，为有待发送数据的客户端注册写事件，最终通过`sendReplyToClient`函数发送响应数据。

## 数据恢复机制

Redis支持两种数据持久化方案：

1. **AOF（Append Only File）**：记录所有修改数据库的命令
2. **RDB（Redis Database）**：周期性地将数据库快照保存到磁盘

Redis启动时会尝试从本地磁盘加载数据，优先使用AOF：

```c
void loadDataFromDisk(void) {
    if (server.aof_state == AOF_ON) {
        // 尝试从AOF文件加载
        int ret = loadAppendOnlyFile(server.aof_filename);
        if (ret == AOF_FAILED || ret == AOF_OPEN_ERR)
            exit(1);
        // ...
    } else {
        // 尝试从RDB文件加载
        rdbSaveInfo rsi = RDB_SAVE_INFO_INIT;
        if (rdbLoad(server.rdb_filename,&rsi,rdb_flags) == C_OK) {
            // ...
        } 
        // ...
    }
}
```

## 数据持久化策略

### AOF持久化

启用AOF后，Redis将执行的写命令追加到AOF缓冲区（`server->aof_buf`），然后在适当时机写入文件：

1. **缓冲：** 命令先写入AOF缓冲区，避免频繁磁盘操作
2. **写入：** 在事件循环的`beforeSleep`回调中，调用`flushAppendOnlyFile`将缓冲区内容写入文件
3. **同步：** 根据配置策略执行`fsync`同步到磁盘

Redis提供三种同步策略：
- **always**：每次写入后立即执行fsync，最安全但性能最差
- **everysec**：每秒执行一次fsync，平衡了安全性和性能
- **no**：不主动执行fsync，依赖操作系统定期刷新（Linux约30秒），性能最好但最不安全

### RDB持久化

RDB持久化通过周期性创建数据库快照来实现，保存频率由配置决定：

```bash
save 600 10  # 600秒内修改10次以上，执行RDB持久化
save 60 1000 # 60秒内修改1000次以上，执行RDB持久化
```

Redis在`serverCron`中检查是否满足保存条件，满足则调用`rdbSaveBackground`异步保存RDB文件：

```c
for (j = 0; j < server.saveparamslen; j++) {
    struct saveparam *sp = server.saveparams+j;

    if (server.dirty >= sp->changes &&
        server.unixtime-server.lastsave > sp->seconds)
    {
        // 触发RDB保存
        rdbSaveBackground(server.rdb_filename,rsiptr);
        break;
    }
}
```

其中，`server.dirty`计数器记录了数据库被修改的次数，由各个命令执行时更新。

## 主从复制机制

Redis主从复制始于从节点向主节点发送`REPLICAOF`命令，随后通过`PSYNC`命令同步数据：

1. **握手阶段：** 从节点连接主节点并验证身份
2. **数据同步：** 
   - 全量同步：主节点创建RDB文件并发送给从节点
   - 增量同步：主节点发送积累的写命令

主节点在`syncCommand`中实现了一个状态机，完成握手、保存RDB、发送RDB和增量数据等步骤。

当RDB发送完毕，从节点进入online状态，此时调用`putSlaveOnline`修改从节点连接的写回调为`sendReplyToClient`。

此后，主节点每执行一次写命令，通过`propagate`函数将命令内容写入从节点的发送缓冲区：

```c
void propagate(int dbid, robj **argv, int argc, int flags) {
    if (!server.replication_allowed)
        return;

    // 写入AOF文件
    if (server.aof_state != AOF_OFF && flags & PROPAGATE_AOF)
        feedAppendOnlyFile(dbid,argv,argc);
    
    // 发送给从节点
    if (flags & PROPAGATE_REPL)
        replicationFeedSlaves(server.slaves,dbid,argv,argc);
}
```

缓冲区中的命令最终由`sendReplyToClient`回调函数发送给从节点，实现了主从数据的实时同步。

## 总结

Redis通过精心设计的模块化架构，实现了高性能、可扩展的内存数据库服务。其核心包括：

- 基于redisObject的统一数据表示
- 高效的IO多路复用网络模型
- 清晰的命令处理流程
- 灵活的数据持久化策略
- 可靠的主从复制机制

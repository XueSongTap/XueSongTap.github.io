---
layout: articles
title: 当我们在启动redis的时候，我们在启动什么
tags: redis
---


## 省流版本
```
main() server.c 

-> InitServer aeCreateEventLoop  listenToPort

-> aeMain() -> aeProcessEvents() -> aeApiPoll -> epoll
```

## gdb 调试环境

### 获取代码
使用redis-3.0-annotation版本的代码进行分析，这个版本带有详细的中文注释。


### 编译设置

首先编译Lua相关代码:

```bash
cd deps/lua/src
make linux
```

然后编译Redis,开启调试信息:


```bash
make MALLOC=libc CFLAGS="-g -O0"  -j 16
```


### GDB调试

```bash
gdb ./redis-server
b main
run
```

## 调试结果

```bash
Breakpoint 1, main (argc=1, argv=0x7fffffffe278) at redis.c:3933
warning: Source file is more recent than executable.
3933	int main(int argc, char **argv) {
3941	    setlocale(LC_COLLATE,"");
3942	    zmalloc_enable_thread_safeness();
3943	    zmalloc_set_oom_handler(redisOutOfMemoryHandler);
3944	    srand(time(NULL)^getpid());
3945	    gettimeofday(&tv,NULL);
3946	    dictSetHashFunctionSeed(tv.tv_sec^tv.tv_usec^getpid());
3949	    server.sentinel_mode = checkForSentinelMode(argc,argv);
3952	    initServerConfig();
3959	    if (server.sentinel_mode) {
3965	    if (argc >= 2) {
4023	        redisLog(REDIS_WARNING, "Warning: no config file specified, using the default config. In order to specify a config file use %s /path/to/%s.conf", argv[0], server.sentinel_mode ? "sentinel" : "redis");
4027	    if (server.daemonize) daemonize();
4030	    initServer();
[413526] 24 Nov 16:27:30.554 * Increased maximum number of open files to 10032 (it was originally set to 1024).
[New Thread 0x7ffff6c006c0 (LWP 413608)]
[New Thread 0x7ffff62006c0 (LWP 413609)]
4033	    if (server.daemonize) createPidFile();
4036	    redisSetProcTitle(argv[0]);
4039	    redisAsciiArt();
                _._
           _.-``__ ''-._
      _.-``    `.  `_.  ''-._           Redis 2.9.11 (8e60a758/1) 64 bit
  .-`` .-```.  ```\/    _.,_ ''-._
 (    '      ,       .-`  | `,    )     Running in stand alone mode
 |`-._`-...-` __...-.``-._|'` _.-'|     Port: 6379
 |    `-._   `._    /     _.-'    |     PID: 413526
  `-._    `-._  `-./  _.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |           http://redis.io
  `-._    `-._`-.__.-'_.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |
  `-._    `-._`-.__.-'_.-'    _.-'
      `-._    `-.__.-'    _.-'
          `-._        _.-'
              `-.__.-'

4042	    if (!server.sentinel_mode) {
4045	        redisLog(REDIS_WARNING,"Server started, Redis version " REDIS_VERSION);
[413526] 24 Nov 16:27:35.192 # Server started, Redis version 2.9.11
4048	        linuxOvercommitMemoryWarning();
4051	        loadDataFromDisk();
4053	        if (server.cluster_enabled) {
4062	        if (server.ipfd_count > 0)
4063	            redisLog(REDIS_NOTICE,"The server is now ready to accept connections on port %d", server.port);
[413526] 24 Nov 16:27:36.884 * The server is now ready to accept connections on port 6379
4065	        if (server.sofd > 0)
4073	    if (server.maxmemory > 0 && server.maxmemory < 1024*1024) {
4078	    aeSetBeforeSleepProc(server.el,beforeSleep);
4079	    aeMain(server.el);
```


## 启动过程分析


### 1. 基础初始化

#### 1.1 进程标题初始化（spt_init/setproctitle）
```c
#ifdef INIT_SETPROCTITLE_REPLACEMENT
    spt_init(argc, argv);
#endif
```
这段代码用于初始化进程标题修改机制。在不同操作系统下表现不同:
- 在不支持setproctitle的系统(如Ubuntu): 使用`INIT_SETPROCTITLE_REPLACEMENT`
- 在FreeBSD等原生支持的系统: 直接使用系统提供的功能

实验对比:
```bash
# 默认情况
$ ps aux | grep redis
yxc 4135102 0.0 0.0 36140 4224 pts/0 Sl+ 21:54 0:00 ./redis-server *:6379

# 注释掉spt_init后
$ ps aux | grep redis  
yxc 4135541 0.0 0.0 36136 4096 pts/0 Sl+ 21:55 0:00 ./redis-server
```
可以看到进程标题信息的差异, 没有初始化标题的进程，缺失了更多的信息

#### 1.2 zmalloc 内存分配初始化
```c
zmalloc_enable_thread_safeness();
zmalloc_set_oom_handler(redisOutOfMemoryHandler);
```

这里完成两个设置:
1. 启用内存分配的线程安全机制:
```c
void zmalloc_enable_thread_safeness(void) {
    zmalloc_thread_safe = 1;
}
```

2. 设置内存溢出处理函数:
```c
void redisOutOfMemoryHandler(size_t allocation_size) {
    redisLog(REDIS_WARNING,"Out Of Memory allocating %zu bytes!");
    redisPanic("Redis aborting for OUT OF MEMORY");
}
```

具体内容得在zmalloc.c 里面看了

#### 1.3 随机数与哈希种子初始化
```c
// 设置随机数种子
srand(time(NULL)^getpid());

// 设置哈希函数种子
struct timeval tv;
gettimeofday(&tv,NULL);
dictSetHashFunctionSeed(tv.tv_sec^tv.tv_usec^getpid());
```

这种设计通过组合时间戳、微秒数和进程ID来生成随机性更好的种子，主要用于:
- 提高哈希表的安全性
- 防止哈希冲突攻击
- 改善哈希分布

### 2. 服务器配置初始化(initServerConfig)

这个函数完成服务器各项参数的默认值设置，包括:

```c
void initServerConfig() {
    // 服务器状态设置
    server.hz = REDIS_DEFAULT_HZ;
    server.port = REDIS_SERVERPORT;
    server.tcp_backlog = REDIS_TCP_BACKLOG;
    
    // 数据库配置
    server.dbnum = REDIS_DEFAULT_DBNUM;
    
    // 持久化相关配置  
    server.aof_state = REDIS_AOF_OFF;
    server.rdb_compression = REDIS_DEFAULT_RDB_COMPRESSION;
    
    // 创建命令表
    server.commands = dictCreate(&commandTableDictType,NULL);
    populateCommandTable();
    
    // 其他重要配置...
}
```

### 3. 服务器初始化(initServer)

这个阶段完成服务器的实际初始化工作:

#### 3.1 基础数据结构创建
```c
server.clients = listCreate();  // 客户端列表
server.slaves = listCreate();   // 从服务器列表
server.monitors = listCreate(); // 监视器列表
```

#### 3.2 网络监听设置

支持TCP和Unix域套接字两种方式:
```c
// TCP监听
if (server.port != 0) {
    listenToPort(server.port,server.ipfd,&server.ipfd_count);
}

// Unix域套接字
if (server.unixsocket != NULL) {
    unlink(server.unixsocket);
    server.sofd = anetUnixServer(/*...*/);
}
```

实际上大部分都是TCP监听，Unixsocket 只适合本机通信
```bash
yxc@yxc-MS-7B89:~/code/2411/ipc-bench/build$ ./unix_thr 16384 100000
message size: 16384 octets
message count: 100000
average throughput: 347018 msg/s
average throughput: 45484 Mb/s
yxc@yxc-MS-7B89:~/code/2411/ipc-bench/build$ ./tcp_local_lat 127.0.0.1 38888 16384 100000
message size: 16384 octets
roundtrip count: 100000
^C
yxc@yxc-MS-7B89:~/code/2411/ipc-bench/build$ ./unix_thr 16384 100000^C
yxc@yxc-MS-7B89:~/code/2411/ipc-bench/build$ ./tcp_thr
usage: tcp_thr <message-size> <message-count>
yxc@yxc-MS-7B89:~/code/2411/ipc-bench/build$ ./tcp_thr 16384 100000
message size: 16384 octets
message count: 100000
average throughput: 147888 msg/s
average throughput: 19383 Mb/s
```
#### 3.3 后台线程初始化
通过bioInit()创建三个后台线程:


```c
void bioInit(void) {
    // 线程1: 处理文件关闭
    pthread_create(&thread, &attr, bioProcessBackgroundJobs,
        (void*)REDIS_BIO_CLOSE_FILE);
        
    // 线程2: 处理AOF同步
    pthread_create(&thread, &attr, bioProcessBackgroundJobs,
        (void*)REDIS_BIO_AOF_FSYNC);
        
    // 线程3: 处理延迟释放
    pthread_create(&thread, &attr, bioProcessBackgroundJobs,
        (void*)REDIS_BIO_LAZY_FREE);
}
```

##### 1 REDIS_BIO_CLOSE_FILE 关闭文件线程

```c
// 主要处理文件关闭操作
void bioProcessCloseFile(void *arg) {
    close((long)arg);  // 关闭文件描述符
}
```

- 异步处理文件关闭操作
- 避免文件关闭时阻塞主线程
- 处理RDB保存、AOF重写等操作后的文件关闭
- 防止文件描述符泄露

##### 2 REDIS_BIO_AOF_FSYNC (AOF同步线程)
```c
// 处理AOF文件的fsync操作
void bioProcessAOFFsync(void *arg) {
    fsync((long)arg);  // 将AOF缓冲区数据刷新到磁盘
}
```
作用：
- 执行AOF持久化的fsync操作
- 确保AOF文件数据安全写入磁盘
- 降低主线程的I/O等待时间
- 支持不同的AOF同步策略(always/everysec/no)

aof fsync设置强相关
```bash
appendfsync everysec    # 推荐，每秒同步一次

```

##### 3 REDIS_BIO_LAZY_FREE (延迟释放线程)
```c
// 处理大对象的内存释放
void bioProcessLazyFree(void *arg) {
    lazyfreeInfo *info = (lazyfreeInfo *)arg;
    freeObjectAsync(info);  // 异步释放对象内存
}
```
作用：
- 异步释放大对象的内存
- 处理大型集合、哈希表的删除
- 避免主线程因释放大量内存而阻塞
- 提高删除大键值对的性能

工作机制：
```c
typedef struct bio_job {
    time_t time;     // 任务创建时间
    void *arg1;      // 任务参数
    void *arg2;
    void *arg3;
    int type;        // 任务类型
} bio_job;

// 任务处理流程
void *bioProcessBackgroundJobs(void *arg) {
    bio_job *job;
    while(1) {
        // 从任务队列获取任务
        job = listFirst(bio_jobs);
        
        // 根据任务类型处理
        switch(job->type) {
            case REDIS_BIO_CLOSE_FILE:
                bioProcessCloseFile(job->arg1);
                break;
            case REDIS_BIO_AOF_FSYNC:
                bioProcessAOFFsync(job->arg1);
                break;
            case REDIS_BIO_LAZY_FREE:
                bioProcessLazyFree(job->arg1);
                break;
        }
    }
}
```

优势：
1. 提高性能
- 减少主线程阻塞
- 并行处理I/O操作
- 异步处理耗时任务

2. 提升稳定性
- 内存管理更稳定
- I/O操作更可靠
- 避免主线程卡顿

3. 资源管理
- 更好的文件描述符管理
- 内存释放更高效
- 磁盘写入更可控

跟lazyfree相关：
```bash
# redis.conf配置
# AOF同步策略

# Lazy Free配置
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
```


### 4. 数据加载(loadDataFromDisk)

主要是尝试load aof 和rdb

```c
void loadDataFromDisk(void) {
    long long start = ustime();
    
    if (server.aof_state == REDIS_AOF_ON) {
        // 优先从AOF文件加载
        loadAppendOnlyFile(server.aof_filename);
    } else {
        // 否则尝试从RDB文件加载
        rdbLoad(server.rdb_filename);
    }
}
```

### 5. 事件循环启动

最后启动事件循环处理各种事件:
```c
aeSetBeforeSleepProc(server.el,beforeSleep);
aeMain(server.el);
```

事件循环的核心实现:
```c
void aeMain(aeEventLoop *eventLoop) {
    eventLoop->stop = 0;
    while (!eventLoop->stop) {
        if (eventLoop->beforesleep != NULL)
            eventLoop->beforesleep(eventLoop);
        aeProcessEvents(eventLoop, AE_ALL_EVENTS);
    }
}
```

```c
int aeProcessEvents(aeEventLoop *eventLoop, int flags) {
    int processed = 0, numevents;
    
    // 1. 检查是否有事件需要处理
    if (!(flags & AE_TIME_EVENTS) && !(flags & AE_FILE_EVENTS)) 
        return 0;
    
    // 2. 准备文件事件和时间事件处理
    if (eventLoop->maxfd != -1 || ((flags & AE_TIME_EVENTS) && !(flags & AE_DONT_WAIT))) {
        aeTimeEvent *shortest = NULL;
        struct timeval tv, *tvp;
        
        // 3. 获取最近的时间事件
        if (flags & AE_TIME_EVENTS && !(flags & AE_DONT_WAIT))
            shortest = aeSearchNearestTimer(eventLoop);
            
        // 4. 设置阻塞超时
        if (shortest) {
            // 计算时间差，设置到 tvp
        } else {
            // 根据 AE_DONT_WAIT 决定是否阻塞
            tvp = (flags & AE_DONT_WAIT) ? &tv : NULL;
        }
        
        // 5. 处理文件事件
        numevents = aeApiPoll(eventLoop, tvp);
        for (int j = 0; j < numevents; j++) {
            // 处理读事件
            if (fe->mask & mask & AE_READABLE) {
                fe->rfileProc(eventLoop,fd,fe->clientData,mask);
            }
            // 处理写事件
            if (fe->mask & mask & AE_WRITABLE) {
                fe->wfileProc(eventLoop,fd,fe->clientData,mask);
            }
            processed++;
        }
    }
    
    // 6. 处理时间事件
    if (flags & AE_TIME_EVENTS)
        processed += processTimeEvents(eventLoop);
        
    return processed;
}
```


主要是调用 aeApiPoll -> epoll O(1) 时间复杂度
```c

/*
 * 获取可执行事件
 */
static int aeApiPoll(aeEventLoop *eventLoop, struct timeval *tvp) {
    aeApiState *state = eventLoop->apidata;
    int retval, numevents = 0;

    // 等待时间
    retval = epoll_wait(state->epfd,state->events,eventLoop->setsize,
            tvp ? (tvp->tv_sec*1000 + tvp->tv_usec/1000) : -1);

    // 有至少一个事件就绪？
    if (retval > 0) {
        int j;

        // 为已就绪事件设置相应的模式
        // 并加入到 eventLoop 的 fired 数组中
        numevents = retval;
        for (j = 0; j < numevents; j++) {
            int mask = 0;
            struct epoll_event *e = state->events+j;

            if (e->events & EPOLLIN) mask |= AE_READABLE;
            if (e->events & EPOLLOUT) mask |= AE_WRITABLE;
            if (e->events & EPOLLERR) mask |= AE_WRITABLE;
            if (e->events & EPOLLHUP) mask |= AE_WRITABLE;

            eventLoop->fired[j].fd = e->data.fd;
            eventLoop->fired[j].mask = mask;
        }
    }
    
    // 返回已就绪事件个数
    return numevents;
}

```



## 参考

http://www.petermao.com/redis/80.html

https://www.cnblogs.com/bianqi/p/12184215.html

https://heshaobo2012.medium.com/redis%E6%BA%90%E7%A0%81%E4%B9%8B%E5%86%85%E5%AD%98%E5%88%86%E9%85%8D%E5%88%86%E6%9E%90-f7374cae0eaa
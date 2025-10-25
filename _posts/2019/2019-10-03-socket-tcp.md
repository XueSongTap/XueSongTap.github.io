---
layout: article
title: 深入理解TCP三次握手:从Socket API视角分析连接建立过程
tags: socket tcp 
---
TCP连接建立过程可以分为三个阶段，分别是：建立连接阶段、数据传输阶段和关闭连接阶段。下面从socket的角度解释TCP连接建立过程：

建立连接阶段：
在socket编程中，当客户端调用connect函数时，会向服务器发送一个SYN包，表示请求建立连接。服务器收到SYN包后，会回复一个SYN+ACK包，表示同意建立连接，并告诉客户端自己的序列号。客户端收到SYN+ACK包后，会回复一个ACK包，表示确认建立连接，并告诉服务器自己的序列号。此时，TCP连接建立成功，可以进行数据传输。

数据传输阶段：
在socket编程中，当客户端调用send函数发送数据时，数据会被分成多个TCP报文段进行传输。每个TCP报文段都会带有序列号和确认号，用于保证数据的可靠传输。服务器收到数据后，会发送一个ACK包进行确认。如果某个TCP报文段没有收到确认，客户端会重传该报文段，直到收到确认为止。

关闭连接阶段：
在socket编程中，当客户端调用close函数时，会向服务器发送一个FIN包，表示要关闭连接。服务器收到FIN包后，会回复一个ACK包进行确认，并告诉客户端自己的序列号。此时，服务器进入CLOSE_WAIT状态，表示等待客户端关闭连接。当服务器也调用close函数时，会向客户端发送一个FIN包，客户端收到FIN包后，会回复一个ACK包进行确认，并告诉服务器自己的序列号。此时，TCP连接关闭成功。


从Socket编程角度解释TCP连接的建立过程,并给出C/C++代码示例如下:

1. TCP连接需要客户端和服务器端均创建socket,用于发送或接收数据。

客户端:
```cpp
int sockfd = socket(AF_INET, SOCK_STREAM, 0);
```
服务器端:
```cpp
int listenfd = socket(AF_INET, SOCK_STREAM, 0);
```

2. 服务器端需要绑定IP地址和端口,并启动监听:
```cpp 
struct sockaddr_in servaddr;
memset(&servaddr, 0, sizeof(servaddr));
servaddr.sin_family = AF_INET;
servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
servaddr.sin_port = htons(5000);

bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr));

listen(listenfd, 10);
```

3. 客户端向服务器发起连接请求:
```cpp
struct sockaddr_in servaddr; 
memset(&servaddr, 0, sizeof(servaddr));
servaddr.sin_family = AF_INET;
servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
servaddr.sin_port = htons(5000);

connect(sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr));
```

4. 服务器端接受客户端连接:
```cpp
int connfd = accept(listenfd, NULL, NULL);
```

至此, TCP三次握手完成,客户端和服务器建立连接,可以进行Socket编程的数据收发了。
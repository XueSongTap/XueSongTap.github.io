---
layout: articles
title: 从socket的角度解释TCP的连接建立过程
tags: socket tcp 
---
TCP连接建立过程可以分为三个阶段，分别是：建立连接阶段、数据传输阶段和关闭连接阶段。下面从socket的角度解释TCP连接建立过程：

建立连接阶段：
在socket编程中，当客户端调用connect函数时，会向服务器发送一个SYN包，表示请求建立连接。服务器收到SYN包后，会回复一个SYN+ACK包，表示同意建立连接，并告诉客户端自己的序列号。客户端收到SYN+ACK包后，会回复一个ACK包，表示确认建立连接，并告诉服务器自己的序列号。此时，TCP连接建立成功，可以进行数据传输。

数据传输阶段：
在socket编程中，当客户端调用send函数发送数据时，数据会被分成多个TCP报文段进行传输。每个TCP报文段都会带有序列号和确认号，用于保证数据的可靠传输。服务器收到数据后，会发送一个ACK包进行确认。如果某个TCP报文段没有收到确认，客户端会重传该报文段，直到收到确认为止。

关闭连接阶段：
在socket编程中，当客户端调用close函数时，会向服务器发送一个FIN包，表示要关闭连接。服务器收到FIN包后，会回复一个ACK包进行确认，并告诉客户端自己的序列号。此时，服务器进入CLOSE_WAIT状态，表示等待客户端关闭连接。当服务器也调用close函数时，会向客户端发送一个FIN包，客户端收到FIN包后，会回复一个ACK包进行确认，并告诉服务器自己的序列号。此时，TCP连接关闭成功。
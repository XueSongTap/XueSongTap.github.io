`sudo nethogs`是一个在Linux系统中使用的命令行工具，用于监控网络流量。通过这个命令，用户可以实时查看哪些进程正在使用网络带宽。这对于诊断网络问题、监控网络使用情况或者仅仅是为了了解哪些应用程序正在访问网络非常有用。

### 安装 Nethogs

在大多数Linux发行版中，`nethogs`不会预装。你可以通过包管理器来安装它。

- 对于基于Debian的系统（如Ubuntu），可以使用：
  ```
  sudo apt-get install nethogs
  ```

### 使用 Nethogs

以下命令启动`nethogs`：

```
sudo nethogs
```

`sudo`是必需的，因为`nethogs`需要足够的权限来监控网络接口和查看所有进程的网络活动。

### 功能和输出解释

启动`nethogs`后，它会显示一个实时更新的界面，列出了当前使用网络的进程。对于每个进程，`nethogs`显示以下信息：

- **PID**：进程的ID。
- **用户**：运行该进程的用户。
- **程序名**：进程的名称。
- **发送**：该进程发送数据的速度（KB/s）。
- **接收**：该进程接收数据的速度（KB/s）。
- **总计**：该进程总的数据传输速度（发送+接收，KB/s）。

示例的输出如下：
```
NetHogs version 0.8.6-3

    PID USER     PROGRAM                  DEV         SENT      RECEIVED      
 986182 admin    python                   tun0      375.434    8884.452 KB/sec
  11989 root     openvpn                  enp4s0    496.379    7077.351 KB/sec
 952909 admin    python                   tun0       19.713     268.779 KB/sec
4164806 admin    sshd: admin@pts/21       enp4s0   2429.762      33.829 KB/sec
1588799 admin    /data/usr/tools/clash..  enp4s0      0.013       0.013 KB/sec
2160320 admin    /snap/code/152/usr/sh..  enp4s0      0.000       0.000 KB/sec
   1520 root     /usr/sbin/NetworkMana..  enp4s0      0.000       0.000 KB/sec
   1618 root     /usr/local/sunlogin/b..  enp4s0      0.000       0.000 KB/sec
      ? root     unknown TCP                          0.000       0.000 KB/sec

  TOTAL                                            3321.300   16264.423 KB/sec
```


### 过滤和排序

默认情况下，`nethogs`会监控所有网络接口上的流量。如果你只想监控特定的网络接口（如`eth0`），可以在命令后面指定接口名称：

```
sudo nethogs eth0
```

在`nethogs`的显示界面中，你可以使用键盘快捷键来改变排序方式，例如按发送或接收数据量排序。

### 结束 Nethogs

可以通过按`q`键退出`nethogs`。




### 小结

`nethogs`是一个非常有用的工具，它能够帮助用户监控和识别哪些进程正在使用网络带宽。这对于管理网络流量、诊断网络问题或者仅仅是为了监控哪些应用正在访问互联网非常有价值。它的用户界面简单直观，即便是Linux新手也能够轻松使用。

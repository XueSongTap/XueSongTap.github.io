---
layout: article
title: Mac 上 SSH 提示 Connection closed by remote host，最后发现是 TUN 代理的问题
tags: SSH macOS 网络
---

今天排查了一个很绕的 SSH 问题，现象看起来像服务端有问题，实际上是本机代理软件的 TUN 模式把 SSH 流量接管了。

## 现象

```bash
ssh -vvv -i ~/Downloads/ssh-key-2026-05-17.key ubuntu@<公网ip>
```

关键报错：

```text
kex_exchange_identification: Connection closed by remote host
Connection closed by <公网ip> port 22
```

连接在 SSH 握手早期就被关掉了，还没走到公钥认证阶段。

## 排除常见原因

### 私钥和端口

```bash
chmod 600 ~/Downloads/ssh-key-2026-05-17.key
ssh-keygen -lf ~/Downloads/ssh-key-2026-05-17.key   # 能正常输出指纹
nc -vz <公网ip> 22                                   # succeeded
```

私钥有效，TCP 22 端口可达，排除密钥损坏和网络不通。

### 本地 SSH 配置

绕过 `~/.ssh/config`：

```bash
ssh -F /dev/null -i ~/Downloads/ssh-key-2026-05-17.key ubuntu@<公网ip>
```

结果依旧 `Connection closed`，排除本地配置问题。

## 转折点

换另一台 Ubuntu 机器连同一台服务器：

```bash
ssh ubuntu@<公网ip>
```

返回：

```text
Permission denied (publickey).
```

这说明：

1. 服务端 SSH 正常工作
2. 主机密钥协商正常
3. 连接已进入认证阶段

问题不在服务端，而在本机环境。

## 定位到 TUN 代理

检查本机代理状态：

```bash
scutil --proxy
```

```text
HTTPProxy : 127.0.0.1:7897
HTTPSProxy : 127.0.0.1:7897
SOCKSProxy : 127.0.0.1:7897
```

检查网卡：

```bash
ifconfig | grep -A2 utun
```

```text
utun4: ...
    inet 198.18.0.1 --> 198.18.0.1 netmask 0xfffffffc
```

看到 `198.18.0.1` 这个典型的 TUN 虚拟地址，基本确认：代理软件开了 TUN 模式，SSH 流量被接管。

## 原因

**Mac 上代理软件开启了 TUN 模式，SSH 到目标主机的连接被代理链路干扰，在握手阶段被提前断开。**

关闭 TUN 后，SSH 立刻恢复正常

## 为什么容易误判

表象特别像服务端问题：

- `ping` 正常
- `nc -vz` 端口通
- `ssh` 却被立刻断开
- 报错是 `Connection closed by remote host`

很容易让人去查安全组、`authorized_keys`、`sshd_config`。但只要另一台机器能走到 `Permission denied`，就说明服务端没问题，应该反过来查本机链路。

## 深入分析：Clash Verge TUN 模式为什么能影响 SSH

### TUN 模式做了什么

Clash Verge 底层使用 mihomo（原 Clash.Meta）内核，开启 TUN 模式时做了三件事：

1. **创建虚拟网卡**：在 macOS 上创建 `utunN` 接口，分配地址 `198.18.0.1/30`（IANA 保留的基准测试地址段）。这是一个 L3（网络层）设备，处理 IP 包。

2. **修改系统路由表**：注入默认路由，把 `0.0.0.0/0` 指向 utun 接口，等效于：
   ```
   default via 198.18.0.1 dev utun4
   ```
   所有出站 IP 包——浏览器 HTTPS、curl HTTP、SSH TCP 22——都被内核路由到 utun。

3. **用户态代理处理**：mihomo 在用户态读取 utun 上的原始 IP 包，解析协议、匹配规则，然后通过代理链路或直连重新发出。

### 为什么 SSH 会断

SSH 流量被 TUN 接管后，可能出现以下问题：

| 场景 | 后果 |
|------|------|
| 规则匹配到代理节点 | SSH 经代理中转，节点可能不支持长连接或干扰握手 |
| 代理节点超时/不稳定 | kex 包丢失或延迟过大，服务端主动断开 |
| fake-ip 模式 DNS 干扰 | DNS 返回虚假 IP（198.18.x.x），目标地址不一致 |
| 代理协议不透传二进制流 | 某些协议处理非 HTTP 二进制流时有 bug |

本次遇到的 `kex_exchange_identification: Connection closed` 说明连接在 SSH 版本字符串交换阶段就断了——代理链路无法正确转发 SSH 握手包，远端 sshd 等待客户端 banner 超时后主动关闭。

### 为什么普通代理模式不影响 SSH

普通系统代理只设置 `http_proxy` / `https_proxy` 和系统代理配置，SSH 客户端不读这些设置，直接走物理网卡。

TUN 模式工作在 L3，绕过了应用层的代理感知——不管应用是否支持代理配置，只要发 IP 包就会被路由到 utun。SSH 无法选择不走 TUN。


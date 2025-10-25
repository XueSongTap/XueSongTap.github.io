---
layout: article
title: 解决Docker容器代理访问问题：代理防火墙配置指南
tags: docker
---

实验室服务器上，docker一直走不了代理，proxy 也排查了，设置也走lan了，

clash换到v2ray，都不行

后来发现是防护墙屏蔽了docker的网

解决

```bash
# 开放docker内容器的访问权限
ufw allow from 172.17.0.1/24
# 刷新防火墙配置
ufw reload
```
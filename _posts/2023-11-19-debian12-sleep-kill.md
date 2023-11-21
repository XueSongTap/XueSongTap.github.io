---
layout: articles
title: 关闭debian休眠
tags: debian linux
---


安装debian12，自带桌面环境，但是自带休眠

禁止debian 休眠


执行：
```bash


sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
Created symlink /etc/systemd/system/sleep.target → /dev/null.
Created symlink /etc/systemd/system/suspend.target → /dev/null.
Created symlink /etc/systemd/system/hibernate.target → /dev/null.
Created symlink /etc/systemd/system/hybrid-sleep.target → /dev/null.
```
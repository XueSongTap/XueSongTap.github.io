---
layout: article
title:  Debian系统休眠功能完全禁用指南：提升主机稳定性
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
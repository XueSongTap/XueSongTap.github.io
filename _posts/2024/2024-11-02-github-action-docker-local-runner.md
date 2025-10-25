---
layout: article
title: CI/CD加速：搭建高性能本地Docker Runner实现GitHub Actions自动化部署
tags: docker github CI/CD runner action workflow
---

# 搭建本地 docker runner 

## github action的额度限制

github action 额度有限制，也有一些环境依赖特殊

选择本地runner，也让环境统一，打算跑在docker上

## docker runner 构建


### 调研
调研了几个方案，

https://github.com/myoung34/docker-github-actions-runner

这个用的人多，较为简单的 docker，虽然不支持横向扩展，但是也够了

支持横向扩展的多为k8s，参考 https://github.com/jonico/awesome-runners?tab=readme-ov-file

感觉有点重，目前没有k8s 环境，放弃


### 构建demo

fork 了 python-fire 仓库试了下

#### docker 构建

一个仓库一个docker

```bash
docker run -d --restart always --name github-runner-python-fire \
  -e REPO_URL="https://github.com/XueSongTap/python-fire" \
  -e RUNNER_NAME="runner-python-fire" \
  -e ACCESS_TOKEN="ghp_XXXXXXXX" \
  -e RUNNER_WORKDIR="/tmp/github-runner-python-fire" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/github-runner-python-fire:/tmp/github-runner-python-fire \
  myoung34/github-runner:latest
```


其中 REPOL_URL 为仓库名

ACCESS_TOKEN 为github 令牌：一般是ghp 开头


##### access_token 获取


GitHub —> Settings (设置) -> 底部，点击左侧菜单中的 Developer settings (开发者设置) ->  左侧 Personal access tokens -> 选择 Tokens (classic) ->  Generate new token (classic) 设置令牌：

Note (备注): 填写一个便于识别的名称，如 "GitHub Runner Token"

Expiration (有效期): 选择合适的有效期

勾选权限范围 (Scopes):

##### docker 构建效果


```bash
yxc@yxc-MS-7B89:~/code/2411/python-fire$ docker logs github-runner-python-fire
Runner reusage is disabled
Obtaining the token of the runner
Configuring

--------------------------------------------------------------------------------
|        ____ _ _   _   _       _          _        _   _                      |
|       / ___(_) |_| | | |_   _| |__      / \   ___| |_(_) ___  _ __  ___      |
|      | |  _| | __| |_| | | | | '_ \    / _ \ / __| __| |/ _ \| '_ \/ __|     |
|      | |_| | | |_|  _  | |_| | |_) |  / ___ \ (__| |_| | (_) | | | \__ \     |
|       \____|_|\__|_| |_|\__,_|_.__/  /_/   \_\___|\__|_|\___/|_| |_|___/     |
|                                                                              |
|                       Self-hosted runner registration                        |
|                                                                              |
--------------------------------------------------------------------------------

# Authentication
√ Connected to GitHub
# Runner Registration
√ Runner successfully added
√ Runner connection is good
# Runner settings
√ Settings Saved.
√ Connected to GitHub
Current runner version: '2.320.0'
2024-11-02 13:41:49Z: Listening for Jobs

```

有listening for jobs 即可

也就是说

1. runner 容器启动后会持续轮询 GitHub，询问是否有新的任务
2. 当有新任务时，runner 会自动下载并执行
3. 通信是单向的：runner 主动询问 GitHub，而不是 GitHub 主动联系 runner

#### workflkow 修改

需要同步编辑 .github/workflow  指定到self host

参考：https://github.com/google/python-fire/compare/master...XueSongTap:python-fire:master

主要是指定

```yaml
    runs-on: self-hosted
```

### 触发验证

```bash
yxc@yxc-MS-7B89:~/code/2411/python-fire$ docker logs github-runner-python-fire
Runner reusage is disabled
Obtaining the token of the runner
Configuring

--------------------------------------------------------------------------------
|        ____ _ _   _   _       _          _        _   _                      |
|       / ___(_) |_| | | |_   _| |__      / \   ___| |_(_) ___  _ __  ___      |
|      | |  _| | __| |_| | | | | '_ \    / _ \ / __| __| |/ _ \| '_ \/ __|     |
|      | |_| | | |_|  _  | |_| | |_) |  / ___ \ (__| |_| | (_) | | | \__ \     |
|       \____|_|\__|_| |_|\__,_|_.__/  /_/   \_\___|\__|_|\___/|_| |_|___/     |
|                                                                              |
|                       Self-hosted runner registration                        |
|                                                                              |
--------------------------------------------------------------------------------

# Authentication
√ Connected to GitHub
# Runner Registration
√ Runner successfully added
√ Runner connection is good
# Runner settings
√ Settings Saved.
√ Connected to GitHub
Current runner version: '2.320.0'
2024-11-02 13:41:49Z: Listening for Jobs
2024-11-02 13:45:58Z: Running job: build (3.10)
2024-11-02 13:49:00Z: Job build (3.10) completed with result: Succeeded
2024-11-02 13:49:06Z: Running job: build (3.11)
2024-11-02 13:51:25Z: Job build (3.11) completed with result: Succeeded
2024-11-02 13:51:30Z: Running job: build (3.12)
2024-11-02 13:54:06Z: Job build (3.12) completed with result: Succeeded
2024-11-02 13:54:14Z: Running job: build (3.13.0-rc.2)
2024-11-02 13:57:38Z: Job build (3.13.0-rc.2) completed with result: Succeeded
2024-11-02 13:57:43Z: Running job: build (3.8)
2024-11-02 13:59:54Z: Job build (3.8) completed with result: Succeeded
2024-11-02 14:00:02Z: Running job: build (3.9)
2024-11-02 14:02:28Z: Job build (3.9) comple
```

## TODO


目前这种构建手段docker 内是root 用户,pip 构建的时候会warning 

```bash
Error: WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

```

但是如果切到非root 用户，如果有apt install 这种脚本又无法跑CI，待解决

## 参考

https://docs.github.com/zh/actions/hosting-your-own-runners/managing-self-hosted-runners-with-actions-runner-controller/deploying-runner-scale-sets-with-actions-runner-controller


https://blog.webp.se/github-actions-hybrid-runner-zh/
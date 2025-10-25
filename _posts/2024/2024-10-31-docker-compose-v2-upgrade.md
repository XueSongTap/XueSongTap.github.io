---
layout: article
title: Docker Compose V2升级指南：新特性、兼容性变更与最佳实践
tags: docker compose
---

## docker compose

docker-compose 这种命令行是v1

docker compose 这种命令行是v2


有些docker-compose.yaml 语法 必须要升级到v2 构建

## 升级


docker compose v2 升级

mac/win 下安装最新docker desktop即可

ubuntu24 下默认apt 还是v1


因此需要手动下载


### 步骤


主要是要放到`~/.docker/cli-plugins` 下

```bash
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.18.1/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose

chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

```


### 验证
```bash
yxc@yxc-MS-7B89:~/immich-app$ docker compose

Usage:  docker compose [OPTIONS] COMMAND

Define and run multi-container applications with Docker.

Options:
      --ansi string                Control when to print ANSI control characters ("never"|"always"|"auto") (default "auto")
      --compatibility              Run compose in backward compatibility mode
      --dry-run                    Execute command in dry run mode
      --env-file stringArray       Specify an alternate environment file.
  -f, --file stringArray           Compose configuration files
      --parallel int               Control max parallelism, -1 for unlimited (default -1)
      --profile stringArray        Specify a profile to enable
      --project-directory string   Specify an alternate working directory
                                   (default: the path of the, first specified, Compose file)
  -p, --project-name string        Project name
```
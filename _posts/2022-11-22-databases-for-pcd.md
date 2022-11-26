---
layout: articles
title: point cloud点云数据存储方案
tags: pcd
---


## geomesa 

hbase 的方案，貌似只支持二维

https://www.geomesa.org/documentation/stable/tutorials/


## mongodb

bson 或者说json结构

什么都能存，三维数据也可以存，难的是如何检索

自带geosptial索引，但是好像是二维

https://www.mongodb.com/docs/manual/geospatial-queries/


## PostgreSQL

似乎支持n维

http://s3.cleverelephant.ca/foss4gna2013-pointcloud.pdf

有官方文档pointcloud存入postgre
https://pgpointcloud.github.io/pointcloud/tutorials/storing.html

This tutorial is a basic introduction to pgPointcloud to store points in a PostgreSQL database hosted on a Docker container.

通过pdal写进去

https://github.com/pgpointcloud/pointcloud

## Elasticsearch

有个geo系列？

https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-queries.html

ignore_z_value

If true (default) three dimension points will be accepted (stored in source) but only latitude and longitude values will be indexed; the third dimension is ignored. If false, geopoints containing any more than latitude and longitude (two dimensions) values throw an exception and reject the whole document. Note that this cannot be set if the script parameter is used.



可以存储，x, y 但是z 值不会被检索，大概是用在二维时空


## spark？

github有专门用于point cloud import进spark的库


https://github.com/IGNF/spark-iqmulus

A library for reading and writing Lidar point cloud collections in PLY, LAS and XYZ formats from Spark SQL.


GeoSpark(Sedona) spark的spatial组件 https://sedona.apache.org/

注：谷歌直接搜的geospark网站只是一个地理信息网站的

## potree

点云浏览器浏览

https://github.com/potree/potree


## LOPoCS

现成的light point cloud server

https://github.com/Oslandia/lopocs

https://gitlab.com/Oslandia/lopocs

provides a way to load Point Cloud from PostgreSQL

用pgpointcloud 连接postgre实现的现成的小server


## GREYHOUND
使用Entwine


## Entwine 
Entwine is a data organization library for massive point clouds


## 比较几种 geospark geotrellis geomesa geowave

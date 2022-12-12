---
layout: articles
title: point cloud点云数据存储方案
tags: pcd
---
# 存储方案

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



# 读写方案


## geotrellis
geotrellis可以把pdal写入hbase数据库
https://geotrellis.readthedocs.io/en/latest/tutorials/setup.html#using-scala

https://github.com/geotrellis/geotrellis-pointcloud


只是使用了pdal 的pipeline思想？


filters.chipper


https://github.com/geotrellis/geotrellis-pointcloud

## geowave ？


## geoserver
https://www.cnblogs.com/shoufengwei/p/5619419.html

## tiledb

https://tiledb.com/

https://docs.tiledb.com/geospatial/pdal


作者：抱紧你的我
链接：https://www.zhihu.com/question/357231642/answer/1373440760
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

①PostGis为PostgreSql提供了空间数据类型，空间函数，空间索引，作为插件使PostGreSql成为了一个空间关系型数据库。 

②Ganos同理为Hbase提供空间能力，使得Hbase成为了一个空间分布式数据库，但是不仅仅如此，数据库仅仅是存储，而Ganos还通过Spark进行数据分析。 

③PolarDb底层通过存储集群的方式实现了按存储容量收费，避免Mysql按照2T的实例购买。首先抛开原理谈优点，相比Mysql实现了容量大，高性价比，分钟级弹性，读一致性，毫秒级延迟，无锁备份，复杂Sql查询加速。说白了是对标于Mysql的竞争产品。 

④GeoHash通过一定的规则把经纬度二维坐标转换成一维编码，从而更加高效，同时在个别场景下有更好的用途。实质是个算法。 

⑤GeoMesa被阿里进一步封装，形成了Ganos。GeoMesa同样是基于Hbase或者其他分布式数据库进行存储，基于Spark进行时空数据分析，两者说白了就是使用一定的方法（添加时空数据类型，添加时空函数等等）把时空数据与大数据组件如存储Hbase，计算Spark结合起来，使之成为能处理，存储，分析时空的大数据平台。 

⑥在上面的场景下，GeoMesa不能处理时空栅格数据，那么就引入了GeoTrellis来处理时空栅格数据。可以说是对GeoMesa的补充。 

⑦GeoMesa中空间处理部分Spark核心部分是GeoSpark算法的优化和改进。 

⑧GeoWave作为Apache顶级项目Accumulo键值对数据库的spatial index，为Accumulo键值对数据库提供了空间能力的同时也提供了空间数据处理能力。 

⑨lgnite可以看作分布式内存网格的一种实现，提供分布式计算的同时提供分布式内存存储（可以看作redis） ⑩Sphinx使用C++语言编写，作为一种基于Sql的全文检索引擎。与其对标的典型有 Lucene,Solr,ElstaticSearch。了解浅薄，如有错欢迎指出。
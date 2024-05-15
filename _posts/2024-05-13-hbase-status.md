---
layout: articles
title: hbase 查询status
tags: hbase nosql hadoop
---
hbase shell
```bash
hbase(main):004:0> status
1 active master, 0 backup masters, 8 servers, 0 dead, 179.2500 average load
```

```bash
hbase(main):001:0>  status 'simple'
active master:  172.17.128.217:16000 1713846984224
0 backup masters
8 live servers
    172.17.129.68:16020 1714282894493
        requestsPerSecond=187.0, numberOfOnlineRegions=176, usedHeapMB=1224, maxHeapMB=1472, numberOfStores=176, numberOfStorefiles=556, storefileUncompressedSizeMB=499341, storefileSizeMB=467167, compressionRatio=0.9356, memstoreSizeMB=426, storefileIndexSizeMB=0, readRequestsCount=499919762, writeRequestsCount=1381090394, rootIndexSizeKB=13106, totalStaticIndexSizeKB=419348, totalStaticBloomSizeKB=476542, totalCompactingKVs=993105057, currentCompactedKVs=993105057, compactionProgressPct=1.0, coprocessors=[GroupedAggregateRegionObserver, Indexer, ScanRegionObserver, ServerCachingEndpointImpl, UngroupedAggregateRegionObserver]
    172.17.139.18:16020 1715237164426
        requestsPerSecond=193.0, numberOfOnlineRegions=176, usedHeapMB=1293, maxHeapMB=1472, numberOfStores=176, numberOfStorefiles=602, storefileUncompressedSizeMB=545886, storefileSizeMB=521185, compressionRatio=0.9548, memstoreSizeMB=409, storefileIndexSizeMB=0, readRequestsCount=331155597, writeRequestsCount=811450281, rootIndexSizeKB=13491, totalStaticIndexSizeKB=449955, totalStaticBloomSizeKB=549775, totalCompactingKVs=639538123, currentCompactedKVs=639538123, compactionProgressPct=1.0, coprocessors=[GroupedAggregateRegionObserver, Indexer, MetaDataEndpointImpl, MetaDataRegionObserver, MultiRowMutationEndpoint, ScanRegionObserver, SequenceRegionObserver, ServerCachingEndpointImpl, UngroupedAggregateRegionObserver]
    172.17.128.219:16020 1715237192000
        requestsPerSecond=179.0, numberOfOnlineRegions=170, usedHeapMB=1255, maxHeapMB=1472, numberOfStores=170, numberOfStorefiles=597, storefileUncompressedSizeMB=524008, storefileSizeMB=504938, compressionRatio=0.9636, memstoreSizeMB=427, storefileIndexSizeMB=0, readRequestsCount=308184181, writeRequestsCount=705004038, rootIndexSizeKB=13316, totalStaticIndexSizeKB=421922, totalStaticBloomSizeKB=496133, totalCompactingKVs=305806053, currentCompactedKVs=305806053, compactionProgressPct=1.0, coprocessors=[GroupedAggregateRegionObserver, Indexer, MetaDataEndpointImpl, MultiRowMutationEndpoint, ScanRegionObserver, ServerCachingEndpointImpl, UngroupedAggregateRegionObserver]
    172.17.139.17:16020 1715237159277
        requestsPerSecond=181.0, numberOfOnlineRegions=171, usedHeapMB=1288, maxHeapMB=1472, numberOfStores=171, numberOfStorefiles=580, storefileUncompressedSizeMB=506305, storefileSizeMB=495094, compressionRatio=0.9779, memstoreSizeMB=416, storefileIndexSizeMB=0, readRequestsCount=301846073, writeRequestsCount=579317723, rootIndexSizeKB=13573, totalStaticIndexSizeKB=414135, totalStaticBloomSizeKB=519861, totalCompactingKVs=456262817, currentCompactedKVs=456262817, compactionProgressPct=1.0, coprocessors=[GroupedAggregateRegionObserver, Indexer, ScanRegionObserver, ServerCachingEndpointImpl, UngroupedAggregateRegionObserver]
    172.17.128.218:16020 1715237175137
        requestsPerSecond=187.0, numberOfOnlineRegions=171, usedHeapMB=1262, maxHeapMB=1472, numberOfStores=171, numberOfStorefiles=576, storefileUncompressedSizeMB=524333, storefileSizeMB=484345, compressionRatio=0.9237, memstoreSizeMB=412, storefileIndexSizeMB=0, readRequestsCount=286597477, writeRequestsCount=867740838, rootIndexSizeKB=12314, totalStaticIndexSizeKB=440021, totalStaticBloomSizeKB=509388, totalCompactingKVs=337719606, currentCompactedKVs=337719606, compactionProgressPct=1.0, coprocessors=[GroupedAggregateRegionObserver, Indexer, ScanRegionObserver, ServerCachingEndpointImpl, UngroupedAggregateRegionObserver]
    172.17.139.16:16020 1714282838975
        requestsPerSecond=180.0, numberOfOnlineRegions=175, usedHeapMB=1224, maxHeapMB=1472, numberOfStores=179, numberOfStorefiles=586, storefileUncompressedSizeMB=533029, storefileSizeMB=497954, compressionRatio=0.9342, memstoreSizeMB=399, storefileIndexSizeMB=0, readRequestsCount=485961723, writeRequestsCount=1438226489, rootIndexSizeKB=13076, totalStaticIndexSizeKB=445381, totalStaticBloomSizeKB=522555, totalCompactingKVs=567920206, currentCompactedKVs=567920206, compactionProgressPct=1.0, coprocessors=[GroupedAggregateRegionObserver, Indexer, MultiRowMutationEndpoint, ScanRegionObserver, ServerCachingEndpointImpl, UngroupedAggregateRegionObserver]
    172.17.129.67:16020 1715237167108
        requestsPerSecond=193.0, numberOfOnlineRegions=179, usedHeapMB=1306, maxHeapMB=1472, numberOfStores=179, numberOfStorefiles=575, storefileUncompressedSizeMB=487457, storefileSizeMB=471678, compressionRatio=0.9676, memstoreSizeMB=420, storefileIndexSizeMB=0, readRequestsCount=319824714, writeRequestsCount=668864554, rootIndexSizeKB=13370, totalStaticIndexSizeKB=397348, totalStaticBloomSizeKB=486443, totalCompactingKVs=309612661, currentCompactedKVs=309612661, compactionProgressPct=1.0, coprocessors=[GroupedAggregateRegionObserver, Indexer, ScanRegionObserver, ServerCachingEndpointImpl, UngroupedAggregateRegionObserver]
    172.17.139.15:16020 1715237158696
        requestsPerSecond=367.0, numberOfOnlineRegions=217, usedHeapMB=1331, maxHeapMB=1472, numberOfStores=217, numberOfStorefiles=832, storefileUncompressedSizeMB=737978, storefileSizeMB=714765, compressionRatio=0.9685, memstoreSizeMB=428, storefileIndexSizeMB=0, readRequestsCount=731122769, writeRequestsCount=1130166783, rootIndexSizeKB=18659, totalStaticIndexSizeKB=658979, totalStaticBloomSizeKB=729119, totalCompactingKVs=650897234, currentCompactedKVs=647327715, compactionProgressPct=0.9945161, coprocessors=[GroupedAggregateRegionObserver, Indexer, ScanRegionObserver, ServerCachingEndpointImpl, UngroupedAggregateRegionObserver]
0 dead servers
Aggregate load: 1667, regions: 1435

```

根据你提供的 `status 'simple'` 命令输出，HBase 集群目前的状态如下：

- **8 live servers**: 有8个区域服务器在运行。
- **0 dead servers**: 没有区域服务器宕机。
- **Aggregate load: 202, regions: 1434**: 整体负载为202，共有1434个区域。

每个区域服务器的情况如下（以其中几个为例）：

- **requestsPerSecond**: 每个服务器的请求处理速率大约在24至29之间。这表示每秒钟每个服务器处理的请求数量。
- **numberOfOnlineRegions**: 每个服务器在线管理的区域数量大致在170至216之间，这影响服务器的工作负荷。
- **usedHeapMB 和 maxHeapMB**: 已使用的堆内存与堆内存上限。大部分服务器的已用堆内存超过1000 MB，接近其上限1472 MB。
- **storefileSizeMB 和 memstoreSizeMB**: 存储文件大小和内存存储的大小，两者都显示出较高的使用量，这也是压力的一个指标。

这些指标表明，尽管没有服务器宕机，但区域服务器的内存使用接近其上限，处理的区域数量较多，并且处理的请求频率较高。这些因素表明集群负载是相对较高的。如果你经常见到内存接近上限或请求处理速率持续较高，可能需要考虑优化HBase的配置，例如增加区域服务器的数量或优化数据模型和访问模式，以降低单个服务器的压力。
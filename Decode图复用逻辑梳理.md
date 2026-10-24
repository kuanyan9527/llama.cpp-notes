# Decode图复用逻辑梳理
基于llama.cpp的tag:b10631代码阅读整理
[源码地址](https://github.com/ggml-org/llama.cpp/tree/b10631)

## 整个过程的伪代码
```cpp
process_ubatch(const llama_ubatch & ubatch) {
    const auto gparams = graph_params(ubatch);
    if (!can_reuse(gparams)) {
        build_graph(gparams);
    }
    graph_compute(graph);
}

graph_compute(ggml_cgraph * gf) {
    ggml_backend_sched_graph_compute_async(sched, gf);
}

ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    ggml_backend_sched_alloc_graph(sched, graph);
    ggml_backend_sched_compute_splits(sched);
}

ggml_backend_sched_compute_splits {
    ggml_backend_graph_compute_async(backend, graph);
}

ggml_backend_graph_compute_async(ggml_backend_sched_t sched) {
    backend->iface->graph_compute(backend, graph);
}
// 这里做了一层抽象，用一个meta backend做为调度者把图分发给各个backend执行计算
static const ggml_backend_i ggml_backend_meta_i = {
    ...
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    ...
};

ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    const bool needs_rebuild = graph->uid != backend_ctx->uid;
    if (need_rebuild) {
        // 将新传过来的计算图按照边界切分成小图
        // 做了边界延迟的优化，使切分出来的计算图尽可能的大
    }
    for (auto &subgraph : subgraphs) {
        for (auto &backend : backends) {
            ggml_backend_graph_compute_async(backend, subgraph);
        }
        comm_allreduce();
    }
}
// 这里才把图发给真正的backend计算，以rpc backend为例继续
static ggml_backend_i ggml_backend_rpc_interface = {
    ...
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    ...
}

ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    bool reuse = rpc_dev_ctx->last_graph_uid == cgraph->uid;
    if (reuse) {
        send_rpc_cmd(RPC_CMD_GRAPH_RECOMPUTE);
    } else {
        send_rpc_cmd(RPC_CMD_GRAPH_COMPUTE);
    }
}

// rpc server收到处理
switch cmd:
    case RPC_CMD_GRAPH_RECOMPUTE: {
        graph = stored_graphs[device].graph;
        ggml_backend_graph_compute(backends[device], graph);
        break
    }
    case RPC_CMD_GRAPH_COMPUTE: {
        // 反序列化构建出计算图
        ggml_backend_graph_compute(backends[device], graph);
        stored_graphs[device].graph = graph;
    }
        
```

## 分析说明 
1. 图复用发生在`process_ubatch`函数中，当`can_reuse`函数判断为`false`时会调用`build_graph`函数重建计算图并分配一个唯一的计算图uid；
2. 在`meta backend`中比较传进来的图uid和上一次使用的图uid，uid不一致则需要执行切分子图的逻辑；
3. 切分子图时会做延迟优化，让切出来的子图足够大，并给子图也分配一个全局的uid；
4. 将切除来的每个子图发送给对应的`backend`执行计算，**等待**每个子图在左右`backend`上执行完后要做一次`allreduce`;
5. `rpc backend`收到子图会比对上一次使用的图uid，一致则给`rpc server`发`RPC_CMD_GRAPH_RECOMPUTE`反之则发`RPC_CMD_GRAPH_COMPUTE`;
6. `rpc server`端收到`RPC_CMD_GRAPH_RECOMPUTE`命令会拿传过来的图uid去缓存里取出计算图执行计算，收到`RPC_CMD_GRAPH_COMPUTE`命令会把图反序列化执行计算并缓存起来；
7. 整个过程中只要`process_ubatch`中没有新建图则后续步骤中图复用一直生效。
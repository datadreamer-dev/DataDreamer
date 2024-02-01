Training Models on Multiple GPUs
#######################################################

DataDreamer makes training on multiple GPUs with :py:class:`~datadreamer.trainers.Trainer` objects extremely
simple and straightforward. All you need to do is pass in a list of devices to the ``device`` parameter of 
:py:class:`~datadreamer.trainers.Trainer` at construction instead of a single device. That's it.

Distributed Training Modes
==========================================

There are two distributed training models supported by DataDreamer:

1. FSDP (default)
2. DDP

.. dropdown:: FSDP (Fully-Sharded Data Parallel)

    By default, when training on multiple GPUs, DataDreamer uses PyTorch's new `FSDP (Fully-Sharded Data Parallel) <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_
    implementation. It is useful for training large models that don't fit on a single GPU. FSDP shards the model parameters across all GPUs and
    only loads a partial slice of the model on each GPU. This allows you to train models that are larger than the memory of a single GPU.

    Your effective batch size is the ``batch_size`` you supply multiplied by the number of GPUs as each GPU will process a batch
    independently and perform a synchronized weight update at the end.

.. dropdown:: DDP (Distributed Data Parallel)

    In Distributed Data Parallel, your model is *not* sharded across multiple GPUs. Instead, each GPU has a full copy of the model. This means that
    your model must fit on a single GPU. DDP is primarily useful for scaling up the effective batch size during training and can help train models
    faster, but is not useful for training models that don't fit on a single GPU. DDP is not used by default by DataDreamer. If you want to use DDP instead, you can pass
    ``fsdp=False`` to :py:class:`~datadreamer.trainers.Trainer` at construction. This will disable FSDP and switch to DDP.

    Your effective batch size is the ``batch_size`` you supply multiplied by the number of GPUs as each GPU will process a batch
    independently and perform a synchronized weight update at the end.

Monitoring GPU Memory Usage
===========================

You can easily monitor the GPU memory usage of your final multi-GPU training setup by passing the ``verbose`` parameter
to :py:class:`~datadreamer.trainers.Trainer` at construction. This will log device memory usage at
the beginning and end of training and after each epoch.


Multi-Node Multi-GPU Training
=============================

If you want to train on multiple nodes (servers or machines) each with multiple GPUs, you can do so by running the same DataDreamer
training script on each node and passing a dictionary to the ``distributed_config`` parameter of 
:py:class:`~datadreamer.trainers.Trainer`. It should look like:

.. code-block:: python

    {
        "master_addr": "<IP address of master node>",
        "master_port": "<free and accessible port on master node>",
        "nnodes": total_number_of_nodes_in_cluster,
        "node_rank": the_rank_of_this_node_in_the_cluster,
    }

.. dropdown:: Setting up ``distributed_config`` and ``device`` on each node

    .. important::
        You should run the same DataDreamer training script on each node. The only difference should be the ``node_rank`` in the
        ``distributed_config`` parameter and ``device`` parameter passed to :py:class:`~datadreamer.trainers.Trainer` at
        construction.

    The ``master_addr`` should be the IP address of the master node amongst all of your nodes. This is also the only node that should log and
    save the model to disk and continue execution after training is complete, the rest of the nodes will exit after training is complete.
    The ``master_port`` should be a free port on the master node that will be used for communication between the nodes and the master node.
    The ``nnodes`` should be the total number of nodes in the cluster. On each node, the ``node_rank`` in the ``distributed_config`` config should be different
    (to specify which # node it is out of the ``nnodes`` total nodes). 


    For each node, you should pass a list of GPUs to the ``device`` parameter of :py:class:`~datadreamer.trainers.Trainer` at
    construction that contains all of the GPUs on the current node that you wish to utilize.

.. tip::
    If you are training on a dataset that is the output of steps in your DataDreamer session, all of those steps must execute on each node.
    If you want to only compute these steps once, you can compute them on the master node and then copy the output folder of the
    :py:class:`~datadreamer.DataDreamer` session to the other nodes before training. This will ensure the other nodes don't recompute
    these steps and instead load the cached outputs from disk.


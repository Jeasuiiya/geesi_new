from cupy.cuda import Device
import time
import sys
from typing import Any, List, Union, Sequence, Tuple, Optional
import ray
import re
import numpy as np
import logging
import threading
from jax.lib import xla_client
from jax.core import gensym
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src import prng
from jax._src import random
import socket
import os
import time
from ray.util.placement_group import get_current_placement_group,PlacementGroup
from geesibling.adapters.jax.model_parallelism import MakeScheduleContext

from jax.tree_util import tree_flatten
from geesibling.adapters.jax import  DeviceType
from geesibling.adapters.jax.pipeline.instructions import PipelineInstType
import jax
import cupy
import ray.util.collective as col
from geesibling.core.lib._graph import Device
from ray.util.placement_group import remove_placement_group
from geesibling.adapters.jax.pipeline.stage_construction import compile_executable
jax.config.update("jax_enable_x64", True)
from geesibling.adapters.jax.shard_parallel.shard_parallel import shard_parallel
def device_config(attrs):
        d = []
        for k, v in attrs.items():
            d.append(Device(v["type"], k, v["memory"], v["free_memory"], v["execute_time"]))
        return d

def try_import_ray_state(error: bool = False):
    try:
        if hasattr(ray.state, "_real_worker"):
            if error:
                raise ImportError("Could not import `ray.state`!"
                                  "You might use the ray-nightly "
                                  "and `ray.state` is deprecated there"
                                  "`pip install ray>=1.13.0`.")
            return ray.state._real_worker  # pylint: disable=protected-access
        else:
            return ray.state
    except ModuleNotFoundError:
        return ray._private.state  # pylint: disable=protected-access

def get_bundle2ip(pg: PlacementGroup = None):
    """get the ip address list from placement group

    The ordering of the ip address are aligned with each bundle index.
    """

    if pg:
        pg_id = pg.id.hex()
    # dictionary: bundle_group to node_ip
    dict_bg2ip = {}

    ray_state = try_import_ray_state()
    resources_list = ray_state.state._available_resources_per_node(  # pylint: disable=protected-access
    ).values()

    for resource in resources_list:
        resource_name_list = resource.keys()

        node_ip = None
        bundle_index_list = []
        for resource_name in resource_name_list:
            if pg:
                try_bundle_index = re.findall(rf"bundle_group_(\d+)_{pg_id}",
                                              resource_name)
            else:
                try_bundle_index = re.findall(r"bundle_group_(\d+)_.*",
                                              resource_name)
            try_node_ip = re.findall(
                r"^node:(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$)", resource_name)

            if try_node_ip:
                node_ip = try_node_ip[0]

            if try_bundle_index:
                bundle_index_list.append(try_bundle_index[0])

        dict_bg2ip.update(
            **dict(zip(bundle_index_list, [node_ip] * len(bundle_index_list))))

    ip_list = []
    for i in range(len(dict_bg2ip)):
        ip_list.append(dict_bg2ip[str(i)])
    return ip_list

def get_bundle_idx(placement_group: PlacementGroup, node_ips: List[str]):

    bundle_ips = get_bundle2ip(placement_group)
    bundle_specs = placement_group.bundle_specs
    node_bundle_idx_list = [
        i for i, bundle_spec in enumerate(bundle_specs)
        if bundle_spec.get("GPU", 0) > 0
    ]
#    print(bundle_ips,bundle_specs)
    if len(node_bundle_idx_list) < len(node_ips):
        raise ValueError("The number of bundles with GPU resources "
                         "is less than the number of node IPs.")

    bundle_ip2idx = {bundle_ips[i]: i for i in node_bundle_idx_list}

    sorted_bundle_idx = [bundle_ip2idx[ip] for ip in node_ips]

    return sorted_bundle_idx


def env_integer(key, default):
    if key in os.environ:
        value = os.environ[key]
        if value.isdigit():
            return int(os.environ[key])

        logger.debug(f"Found {key} in environment, but value must "
                     f"be an integer. Got: {value}. Returning "
                     f"provided default {default}.")
        return default
    return default

def try_import_ray_worker(error: bool = False):
    # In the ray-nightly version,
    # worker = _DeprecationWrapper("worker", ray._private.worker)
    # `_DeprecationWrapper` has attributes of `_real_worker`
    try:
        if hasattr(ray.worker, "_real_worker"):
            if error:
                raise ImportError("Could not import `ray.worker`!"
                                  "You might use the ray-nightly "
                                  "and `ray.worker` is deprecated there"
                                  "`pip install ray==1.13.0`.")
            return ray.worker._real_worker  # pylint: disable=protected-access
        else:
            return ray.worker
    except ModuleNotFoundError:
        return ray._private.worker  # pylint: disable=protected-access

def create_placement_group(num_hosts,
                           host_num_devices,
                           name,
                           additional_resources_per_host=None):
    current_placement_group = get_current_placement_group()
    ray_worker = try_import_ray_worker()
    worker = ray_worker.global_worker  # pylint: disable=protected-access
    should_capture_child_tasks_in_placement_group = (
        worker.should_capture_child_tasks_in_placement_group)
    should_create_placement_group = (
        current_placement_group is None or
        not should_capture_child_tasks_in_placement_group)

    if should_create_placement_group:
        additional_resources_per_host = (additional_resources_per_host or {})

        bundles = [{
            "CPU": 4,
            "GPU": host_num_devices[i],
            **additional_resources_per_host
        } for i in range(num_hosts)]
        strategy = "SPREAD"
        placement_group = ray.util.placement_group(bundles,
                                                   strategy=strategy,
                                                   name=name or "")
        logging.debug("Waiting for placement group to start.")
        timeout = env_integer("ALPA_PLACEMENT_GROUP_TIMEOUT_S_ENV", 100)
        ready, _ = ray.wait([placement_group.ready()], timeout=timeout)
        if ready:
            logging.debug("Placement group has started.")
        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure your "
                "cluster either has enough resources or use an "
                "autoscaling cluster. If you are running on a cluster, "
                "make sure you specify an address in `ray.init()`, for example,"
                ' `ray.init("auto")`. You can also increase the timeout by '
                "setting the ALPA_PLACEMENT_GROUP_TIMEOUT_S environment "
                "variable. Current resources available: "
                f"{ray.available_resources()}, resources requested by "
                f"the placement group: {placement_group.bundle_specs}")
        return placement_group
    else:
        return current_placement_group


def is_ray_node_resource(resource_key):
    """Check if the current resource is the host ip."""
    ishost_regex = re.compile(r"^node:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    return ishost_regex.match(resource_key)

def device_id_to_str(host_ip, device_id, device_type="gpu"):
    """Convert device id (int) to a canonical device string."""
    return f"{host_ip}:{device_type}:{device_id}"

def check_server_port(address, port):
    """Checking Port Opening Status """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((address, port))
            return True
        except socket.error:
            return False

def get_global_cluster():
    return global_cluster

def retrieve_placement_group():
    """retrieve the placement group to support node affinity scheduling

    If already inside the placement group, retrieve the current placement
    group (case I). Then, if the placement group is detected globally in
    alpa, retrieve the global placement group (case II).

    """
    # case 1:
    # Get the current placement group which a task or actor is using
    current_placement_group = get_current_placement_group()
    if current_placement_group:
        return current_placement_group
    # case 2:
    # Get the placement group created when alpa.init('ray')
    global_cluster = get_global_cluster()
    if global_cluster and global_cluster.placement_group:
        alpa_placement_group = global_cluster.placement_group
        return alpa_placement_group


class MeshHostWorker:
    """
    A ray actor that manages the xla computation and buffers on a single host.
    """

    def __init__(self, server_address: str, num_hosts: int, host_id: int,
                 mesh_id: int,devices,node_ips
                 ):
        self.num_hosts = num_hosts
        self.host_id = host_id
        self.mesh_id = mesh_id
        self.devices=devices
        self.node_ips=node_ips
        self.distributed_client = (
            xla_client._xla.get_distributed_runtime_client(
                server_address, host_id))
        self.distributed_client.connect()
        self.backend = xla_client.make_gpu_client(self.distributed_client,
                                                      node_id=host_id)
        self.local_devices = self.backend.local_devices()
        self.num_devices = len(self.local_devices)
        self.instructions=[]
        self.jax_all_stages=[]
        self.func=[]
        self.out_tree=[]
        self.policy=[]
        self.method=[]
        self.args=[]
        self.kwargs=[]
        self.global_outvars=[]
        self.global_invars=[]
        self.outvars_map=[]
        self.result=[]
        self.buffers = [{}]  # Dict[uuid -> Sequence[DeviceArray]]
        self.executables = {}  # Dict[uud -> MeshWorkerExecutable]
        self.send_tasks = {}  # Dict[uuid -> ReshardingSendTask]
        self.recv_tasks = {}  # Dict[uuid -> ReshardingRecvTask]
        self.broadcast_tasks = {}  # Dict[uuid -> BroadcastTask]
        self.broadcast_communicators = {}
        self.data_loaders = {}  # Dict[uuid -> MeshWorkerDataLoader]
        self.data_loader_iters = {}  # Dict[uuid -> iterator]
        self.reduction_vector=[]
        self.flag=[]


    def get_stages_to_run(self,f,policy, method, batch_invars, num_microbatch,args_flat,out_tree,layer_num,*abstract_args):
        self.func=f
        self.policy=policy
        self.method=method
        self.layer_num=layer_num
        self.out_tree=out_tree

        (self.jax_all_stages,instructions,self.global_invars,self.global_outvars,self.outvars_map,self.reduction_vector,self.flag)=compile_executable(f, batch_invars, num_microbatch,layer_num,*abstract_args)
        self.instructions=instructions[self.mesh_id]
        #data_spilt
        self.data_put_buffers(args_flat,self.global_invars,num_microbatch,self.reduction_vector,self.flag)
        return self.instructions

    def get_data_to_split(self,args_flat,num_microbatch):
        self.data_put_buffers(args_flat,self.global_invars,num_microbatch,self.reduction_vector,self.flag)
        return self.instructions

    def data_put_buffers(self,flat_args,global_invars,micro_batch_id,reduction_vector,flags):
        self.buffers=[{} for i in range(micro_batch_id+1)]
        for i,var in enumerate(global_invars):
            if reduction_vector[i]:
                datas=np.split(flat_args[i],micro_batch_id,0)
                for j in range(micro_batch_id):
                    self.buffers[j][var]=datas[j]
            elif i>=flags:
                self.buffers[-1][var]=np.zeros(var.aval.shape, dtype=var.aval.dtype)
            else:
                self.buffers[-1][var]=flat_args[i]

    def run_executable(self,num):
        instruction=self.instructions[num]
        if instruction.opcode == PipelineInstType.RUN:
            self.run_model_parallelism(instruction.stage_id,
                                    instruction.micro_batch_id,
                                    instruction.input_vars,
                                    instruction.output_vars
                                    )
        elif instruction.opcode == PipelineInstType.SEND:
            self.do_send(instruction.micro_batch_id,
                        instruction.output_vars,
                        instruction.dst_rank,
                        instruction.groupname,
                        )
        elif instruction.opcode == PipelineInstType.RECV:
            self.do_recv(instruction.micro_batch_id,
                        instruction.input_vars,
                        instruction.src_rank,
                        instruction.groupname,
                        )

    def run_shard_parallelism(self,num,tp_num):
        if  tp_num > len(self.devices) :
            raise ValueError("tp_num should be less than the number of devices")
        instruction=self.instructions[num]
        stage_id=instruction.stage_id
        micro_batch_id=instruction.micro_batch_id
        input_vars=instruction.input_vars
        output_vars=instruction.output_vars
        flat_args=[]
        for var in input_vars:
            if var in self.buffers[-1]:
                flat_args.append(self.buffers[-1][var])
            else:
                flat_args.append(self.buffers[micro_batch_id][var])

        result = shard_parallel(self.jax_all_stages[stage_id], flat_args, tp_num, self.out_tree)

        for var, val in zip(output_vars,result):
             self.buffers[micro_batch_id][var] = val
        print(stage_id,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
    def run_model_parallelism(self,num):

        instruction=self.instructions[num]
        stage_id=instruction.stage_id
        micro_batch_id=instruction.micro_batch_id
        input_vars=instruction.input_vars
        output_vars=instruction.output_vars
        flat_args=[]
        for var in input_vars:
            if var in self.buffers[-1]:
                flat_args.append(self.buffers[-1][var])
            else:
                flat_args.append(self.buffers[micro_batch_id][var])
        make_ctx = MakeScheduleContext(self.func ,"", self.policy or "fddps", self.method or "")#用于构建调度上下文
        #make_ctx.args, make_ctx.kwargs = self.args, self.kwargs
        bundle_ips = get_bundle2ip(retrieve_placement_group())
        ip=bundle_ips[self.host_id]
        device_info={}
#len(self.devices)
        for i in range(len(self.devices)):
            device_info["gpu:"+str(i)]={
                    "type": DeviceType.gpu,
                    "memory": 3 * 1024 * 1024* 1024 ,
                    "free_memory":3 * 1024 * 1024* 1024 ,
                    "execute_time": 0,
                }
        devices=device_config(device_info)
        ctx = make_ctx([self.jax_all_stages[stage_id],devices]) #(ShapedArray(int32[], weak_type=True), ShapedArray(int32[], weak_type=True))
        result = make_ctx.get_model_parallelism_result(ctx, flat_args, self.out_tree)
        for var, val in zip(output_vars,result):
            self.buffers[micro_batch_id][var] = val
        print(stage_id,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")

    def do_send_data(self, num):
        instruction = self.instructions[num]
        micro_batch_id = instruction.micro_batch_id
        output_vars = instruction.output_vars
        dst_rank = instruction.dst_rank
        group_name = instruction.groupname
        dst_gpu_idx = 0
        send_buffers = []
        for var in output_vars:
            if isinstance(var.aval.dtype, np.dtype):
                send_buffer = self.buffers[micro_batch_id][var]
                send_buffer = send_buffer.astype(np.float64)
                send_buffer=send_buffer.flatten()
                send_buffers.append(send_buffer)
            else:
                send_buffer = self.buffers[micro_batch_id][var]
                send_buffer = prng.random_unwrap(send_buffer)
                send_buffer = send_buffer.astype(np.float64)
                send_buffer = cupy.array(send_buffer)
        concatenated_send_buffer = np.concatenate([cupy.asnumpy(buf) for buf in send_buffers])
        with cupy.cuda.Device(0):
            concatenated_send_buffer = cupy.array(concatenated_send_buffer)
        col.send_multigpu(concatenated_send_buffer, dst_rank, dst_gpu_idx, group_name)
        cupy.cuda.Device(0).synchronize()


    def do_recv_data(self, num):
        instruction=self.instructions[num]
        micro_batch_id=instruction.micro_batch_id
        input_vars=instruction.input_vars
        src_rank=instruction.src_rank
        group_name=instruction.groupname
        src_gpu_idx = 0
        total_size = 0
        for var in input_vars:
            if isinstance(var.aval.dtype, np.dtype):
                total_size += np.prod(var.aval.shape)
            else:
                total_size+=2
        with cupy.cuda.Device(0):
            concatenated_recv_buffer = cupy.zeros(int(total_size), dtype=np.float64)
        col.recv_multigpu(concatenated_recv_buffer, src_rank, src_gpu_idx, group_name)
        cupy.cuda.Device(0).synchronize()
        concatenated_recv_buffer = concatenated_recv_buffer.get()
        start_idx = 0
        for i, var in enumerate(input_vars):
            if isinstance(var.aval.dtype, np.dtype):
                size = int(np.prod(var.aval.shape))
                recv_buffer = concatenated_recv_buffer[int(start_idx):int(start_idx) + int(size)]
                recv_buffer = recv_buffer.reshape(var.aval.shape)
                start_idx += size
                recv_buffer = recv_buffer.astype(var.aval.dtype)
            else:
                recv_buffer = concatenated_recv_buffer[start_idx:start_idx + 2]
                start_idx += 2
                recv_buffer = recv_buffer.astype(np.uint32)
                recv_buffer = prng.random_wrap(recv_buffer, impl=random.default_prng_impl())
            val = jax.device_put(recv_buffer)
            if var in self.buffers[-1]:
                self.buffers[-1][var] = val
            else:
                self.buffers[micro_batch_id][var] = val

    def do_send_1(self, num):
        instruction = self.instructions[num]
        micro_batch_id = instruction.micro_batch_id
        output_vars = instruction.output_vars
        dst_rank = instruction.dst_rank
        group_name = instruction.groupname
        dst_gpu_idx = 0
        with cupy.cuda.Device(0):
            for var in output_vars:
                if isinstance(var.aval.dtype, np.dtype):
                    send_buffer = self.buffers[micro_batch_id][var]
                    if var.aval.dtype == np.bool_:
                        send_buffer = send_buffer.astype(np.uint32)
                    with cupy.cuda.Device(0):
                        send_buffer = cupy.array(send_buffer)
                else:
                    send_buffer = self.buffers[micro_batch_id][var]
                    send_buffer = prng.random_unwrap(send_buffer)
                    send_buffer = cupy.array(send_buffer)
                col.send_multigpu(send_buffer, dst_rank, dst_gpu_idx, group_name)
                cupy.cuda.Device(0).synchronize()

    def do_recv_1(self, num):
        instruction=self.instructions[num]
        micro_batch_id=instruction.micro_batch_id
        input_vars=instruction.input_vars
        src_rank=instruction.src_rank
        group_name=instruction.groupname
        src_gpu_idx = 0
        with cupy.cuda.Device(0):
            for var in input_vars:
                if isinstance(var.aval.dtype, np.dtype):
                    if var.aval.dtype==np.bool_:
                        recv_buffer = cupy.zeros(var.aval.shape,dtype=np.uint32)
                    else:
                        recv_buffer = cupy.zeros(var.aval.shape,dtype=var.aval.dtype)
                    col.recv_multigpu(recv_buffer, src_rank,src_gpu_idx, group_name)
                    cupy.cuda.Device(0).synchronize()
                    recv_buffer = recv_buffer.get()
                    if var.aval.dtype==np.bool_:
                        recv_buffer = recv_buffer.astype(np.bool_)
                else:
                    recv_buffer = cupy.zeros((2,), dtype=cupy.uint32)
                    col.recv_multigpu(recv_buffer, src_rank,src_gpu_idx, group_name)
                    cupy.cuda.Device(0).synchronize()
                    recv_buffer = recv_buffer.get()
                    recv_buffer = prng.random_wrap(recv_buffer, impl=random.default_prng_impl())
                val = jax.device_put(recv_buffer)
                if var in self.buffers[-1]:
                    self.buffers[-1][var] = val
                else:
                    self.buffers[micro_batch_id][var] = val

    def shutdown(self):
        self.distributed_client.shutdown()

    def free_buffers(self):
        self.buffers=[{}]
        self.result=[]

    def return_result(self):
        self.result=[]
        for var in self.global_outvars:
            if var in self.buffers[-1]:
                self.result.append(self.buffers[-1][var])
            else:
                self.result.append(self.buffers[-2][var])
        _, out_tree = tree_flatten(self.out_tree)
        return tree_unflatten(out_tree, self.result)
    def read_result(self):
        return self.result

    def read_buffers(self):
        return self.buffers
    def read_global_outvars(self):
        return self.global_outvars


    def read_name(self):
        return (self.mesh_id,self.host_id)
    def sync_all(self):
        for device in self.local_devices:
            device.synchronize()

class DeviceCluster:
    def __init__(self,
                 num_nodes: int = None,
                 num_devices_per_node: int = None,
                 namespace: Optional[str] = None):

        #get ray_global_node,head_info,all_host_info,all_host_ips
        ray_global_node = ray.worker._global_node
        try:
            self.head_info = ray_global_node.address_info
        except AttributeError as ae:
            raise RuntimeError(
                "Cannot access ray global node. Did you call ray.init?") \
                from ae
        all_host_info = []
        all_host_ips = []
        for node in ray.nodes():
            for key in node["Resources"]:
                if (is_ray_node_resource(key) and
                        "GPU" in node["Resources"]):
                    all_host_info.append(node)
                    all_host_ips.append(key.split("node:")[-1])

        # get num_hosts,num_devices_per_node
        all_host_num_devices = []
        for host_info in all_host_info:
            number = host_info["Resources"]["GPU"]
            all_host_num_devices.append(int(number))
        num_hosts = len(all_host_info)
        self.host_num_devices = all_host_num_devices
        self.host_info = all_host_info
        self.host_ips = all_host_ips


        # Create placement group
        self.namespace = namespace
        if namespace:
            pg_name = namespace + "_pg"
            try:
                pg = ray.util.get_placement_group(pg_name)
            except ValueError:
                pg = None
        else:
            pg_name = pg = None

        if pg:
            self.placement_group = pg
        else:
            self.placement_group = create_placement_group(
                num_hosts, self.host_num_devices, pg_name)

        # Update the Device Cluster info from placement group
        if num_devices_per_node or num_nodes:
            # map: host ip to host info
            host_ip2info = dict(zip(all_host_ips, all_host_info))

            # get bundle's ip address
            ips = get_bundle2ip(self.placement_group)
            bundle_specs = self.placement_group.bundle_specs

            # filter out the bundle index with device (GPUs)
            device_bundle_idx_list = [
                i for i, bundle_spec in enumerate(bundle_specs)
                if bundle_spec.get("GPU", 0) > 0
            ]

            # filter nodes according to the placement group
            self.host_info = [host_ip2info[ip] for ip in ips]
            self.host_ips = [
                ips[bundle_idx] for bundle_idx in device_bundle_idx_list
            ]
        else:
            self.host_info = all_host_info
            self.host_ips = all_host_ips





    @property
    def num_cpus(self):
        return sum(
            map(lambda info: int(info["Resources"]["CPU"]), self.host_info))

    @property
    def num_devices(self):
        return sum(self.host_num_devices)

    @property
    def num_hosts(self):
        return len(self.host_info)


    def get_virtual_physical_mesh(self,
                                  host_ids: Sequence[int] = None,
                                  num_devices_per_host: int = None):
        host_ids = host_ids or np.arange(len(self.host_info))
        host_info = [self.host_info[x] for x in host_ids]
        num_devices_per_host = num_devices_per_host or self.host_num_devices[
            host_ids[0]]
        for host_id in host_ids:
            assert self.host_num_devices[host_id] >= num_devices_per_host
        return VirtualPhysicalMesh(host_ids=host_ids,
                                   host_info=host_info,
                                   num_devices_per_host=num_devices_per_host,
                                   parent=self,namespace=self.namespace)



    def delete_placement_group(self):
        """remove the placement group for the current device cluster."""
        remove_placement_group(self.placement_group)
        self.placement_group = None




class VirtualPhysicalMesh:
    def __init__(self,
                 host_ids: Sequence[int],
                 host_info: Sequence[dict],
                 num_devices_per_host,
                 namespace,
                 parent: "VirtualPhysicalMesh" = None,
                 devices: Sequence[Sequence[int]] = None):
        # host_ids are the indices of hosts in the global DeviceCluster
        self.host_ids = host_ids
        self.host_info = host_info
        self.num_devices_per_host = num_devices_per_host
        self.parent = parent
        self.namespace=namespace
        self.launched_physical_mesh = None
        self.launched_physical_mesh_group = None
        if devices is not None:
            if len(devices) != len(host_ids):
                raise RuntimeError(
                    "Please specify the gpu IDs used on each host.")
            if not all(len(ids) == num_devices_per_host for ids in devices):
                raise RuntimeError(
                    "Device IDs specified for each host does not align "
                    "with `num_devices_per_host`.")
        else:
            devices = [list(range(num_devices_per_host)) for _ in host_ids]

        self.devices = devices
        # Depending on gpu_ids, generate device strs and ask Ray to allocate.
        self.device_strs = []
        for i in range(self.num_hosts):
            ip = self.host_info[i]["NodeManagerAddress"]
            self.device_strs.extend(
                [device_id_to_str(ip, j) for j in devices[i]])
    @property
    def shape(self):
        return (len(self.host_ids), self.num_devices_per_host)

    @property
    def num_devices(self):
        """Return the total number of GPUs on this mesh."""
        return len(self.host_ids) * self.num_devices_per_host

    @property
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        return len(self.host_ids)

    def slice_2d(self, host_indices, device_indices):
        # [0] [(0, 1, 2, 3)]
        host_ids = [self.host_ids[x] for x in host_indices]
        host_info = [self.host_info[x] for x in host_indices]

        # Check the validity of device_indices
        for i in range(len(device_indices)):
            for x in device_indices[i]:
                assert x in self.devices[i]

        return VirtualPhysicalMesh(host_ids=host_ids,
                                   host_info=host_info,
                                   num_devices_per_host=len(device_indices[0]),
                                   parent=self,
                                   devices=device_indices,
                                    namespace=self.namespace)


    def get_physical_mesh_group(self, sliced_virtual_meshes):
        """Launch a physical mesh group (which will request resources from
        Ray)."""
        assert self.launched_physical_mesh_group is None, \
            "Physical mesh group can only be launched once."

        # Launch physical meshes in parallel
        physical_meshes = [None] * len(sliced_virtual_meshes)

        def launch_func(i):
            physical_meshes[i] = sliced_virtual_meshes[i].get_physical_mesh(i)

        threads = []
        for i in range(len(sliced_virtual_meshes)):
            t = threading.Thread(target=launch_func, args=(i,))
            t.start()
            threads.append(t)
        for i in range(len(sliced_virtual_meshes)):
            threads[i].join()

        self.launched_physical_mesh_group = (PhysicalDeviceMeshGroup(
            physical_meshes, self))
        return self.launched_physical_mesh_group

    def get_physical_mesh(self, mesh_id: int = 0):
        """Launch a physical mesh (which will request resources from Ray)."""
        assert self.launched_physical_mesh is None, \
            "Physical mesh can only be launched once."

        self.launched_physical_mesh = DistributedPhysicalDeviceMesh(
            host_ids=self.host_ids,
            host_info=self.host_info,
            num_devices_per_host=self.num_devices_per_host,
            parent=self,
            devices=self.devices,
            mesh_id=mesh_id)
        return self.launched_physical_mesh





def get_sliced_virtual_submeshes(virtual_mesh,submesh_shapes):
    """Slice the origin mesh into submeshes given submesh shapes."""
    num_hosts = virtual_mesh.num_hosts
    num_devices_per_host = virtual_mesh.num_devices_per_host
    submesh_sizes = [np.prod(submesh) for submesh in submesh_shapes]
    virtual_submeshes = [None] * len(submesh_shapes)
    assert sum(submesh_sizes) == virtual_mesh.num_devices
    sorted_submesh_indices = np.argsort(submesh_sizes, kind="stable")
    current_host_id = 0
    current_device_id = 0
    for i in reversed(sorted_submesh_indices):
        required_num_hosts, required_num_devices = submesh_shapes[i]
        if required_num_devices == num_devices_per_host:
            assert current_device_id == 0
            assert current_host_id + required_num_hosts <= num_hosts, (
                "Do not have enough hosts for the solution.")
            virtual_submeshes[i] = virtual_mesh.slice_2d(
                tuple(
                    range(current_host_id,
                          current_host_id + required_num_hosts)),
                (tuple(range(num_devices_per_host)),) * required_num_hosts)
            current_host_id += required_num_hosts
        else:
            assert required_num_hosts == 1
            assert required_num_devices < num_devices_per_host
            assert (current_device_id + required_num_devices <=
                    num_devices_per_host), (
                        "Do not have enough devices in a host for the solution")
            virtual_submeshes[i] = virtual_mesh.slice_2d([current_host_id], [
                tuple(
                    range(current_device_id,
                          current_device_id + required_num_devices))
            ])
            current_device_id += required_num_devices
            if current_device_id == num_devices_per_host:
                current_host_id += 1
                current_device_id = 0
    assert current_host_id == num_hosts
    assert current_device_id == 0
    return virtual_submeshes

class PhysicalDeviceMesh:
    def _init_(slef):
        pass
used_port_set = set((None,))

class DistributedPhysicalDeviceMesh():
    """
    A multi-host physical device mesh to run computation distributedly.
    It uses ray actors and the distributed XLA runtime.
    """

    def __init__(self,
                 host_ids: Sequence[int],
                 host_info: Sequence[dict],
                 num_devices_per_host: int,
                 parent: Optional["VirtualPhysicalMesh"] = None,
                 devices: Optional[Sequence[Sequence[int]]] = None,
                 mesh_id: Optional[int] = None,
                 namespace: Optional[str] = None):
        # host_ids are the indices of hosts in the global DeviceCluster
        self.host_ids = host_ids
        self.host_info = host_info
        self.num_hosts = len(host_ids)
        self.num_devices_per_host = num_devices_per_host
        self.parent = parent
        self.mesh_id = mesh_id
        self.workers = None
        self.service_server = None
        self.operation_executables = {}
        self.one_replica_ids = {}
        self.namespace = namespace

        if devices is not None:
            if len(devices) != len(host_ids):
                raise RuntimeError(
                    "Please specify the gpu IDs used on each host.")
            if not all(len(ids) == num_devices_per_host for ids in devices):
                raise RuntimeError(
                    "Devices specified for each host does not align "
                    "with `num_devices_per_host`.")
        else:
            devices = [list(range(num_devices_per_host)) for _ in host_ids]

        self.devices = devices
        self.device_strs = []
        self.node_ips = []
        for i in range(self.num_hosts):
            ip = self.host_info[i]["NodeManagerAddress"]
            self.device_strs.extend(
                [device_id_to_str(ip, j) for j in devices[i]])
            self.node_ips.append(ip)
        found_existing_workers = False
        if self.namespace:
            try:
                ray.get_actor(self.get_host_worker_name(0))
                found_existing_workers = True
            except ValueError:
                pass
        if found_existing_workers:
            self.service_server = None
            self.workers = self.connect_to_existing_workers()
            self.launched = False
        else:
            self.service_server, self.workers = self.launch_xla_servers()
            self.launched = True

        self.to_delete_remote_refs = []
        self.to_delete_remote_ref_ct = 0
    def get_host_worker_name(self, host_id):
        if self.namespace:
            return f"mesh_{self.mesh_id}_host_{host_id}"
        else:
            return None

    def shutdown(self):
        if self.service_server:
            self.service_server.shutdown()
            self.service_server = None
        self.launched = False

    def launch_xla_servers(self):
        port = None
        while port in used_port_set:
            port = np.random.randint(20000,25000)
            if check_server_port(ray.util.get_node_ip_address(), port):
                port = None
        used_port_set.add(port)

        server_address = f"{ray.util.get_node_ip_address()}:{port}"
        service_server = xla_client._xla.get_distributed_runtime_service(
            server_address, self.num_hosts)
        time.sleep(0.4)

        workers = []
        placement_group = retrieve_placement_group()
        device_bundle_idx_list = get_bundle_idx(placement_group, self.node_ips)
        for i in range(self.num_hosts):
            # Set XLA environment variables
            env_vars = {}
            bundle_index = device_bundle_idx_list[i]

            host_worker_name = self.get_host_worker_name(i)
            # Launch the MeshHostWorker
            cls = ray.remote(num_cpus=0,
                             num_gpus=self.num_devices_per_host)(MeshHostWorker)
            worker = cls.options(name=host_worker_name,placement_group=placement_group,
                                 placement_group_bundle_index=bundle_index,
                                 runtime_env={
                                     "env_vars": env_vars
                                 }).remote(server_address, self.num_hosts, i,
                                           self.mesh_id,self.devices[i],self.node_ips[i]
                                           )
            workers.append(worker)
        return service_server, workers

class PhysicalDeviceMeshGroup:
    """A list of physical devices that forms a pipeline."""

    def __init__(self, meshes: Sequence[DistributedPhysicalDeviceMesh],
                 parent: VirtualPhysicalMesh):
        self.meshes = list(meshes)
        self.parent = parent
        self.collective_groups: List[List[Any]] = [
            [None for _ in range(len(self))] for _ in range(len(self))
        ]

    def __getitem__(self, index):
        return self.meshes[index]

    def __len__(self):
        return len(self.meshes)

    def index(self, *args, **kwargs):
        return self.meshes.index(*args, **kwargs)

def get_submesh_shapes(layer_num,id_num,ip_num):
    all_gpu=id_num*ip_num
    if all_gpu%layer_num==0 and ip_num%layer_num==0:
        return [(1,int(all_gpu/layer_num))]*layer_num
    else:
        print("error")

def get_global_virtual_physical_mesh():
    return global_virtual_physical_mesh

def init_global_cluster(cluster: str,
                        cluster_address: Optional[str] = None,
                        num_nodes: Optional[int] = None,
                        num_devices_per_node: Optional[int] = None,
                        namespace: Optional[str] = "gee_default_space",
                        layer_num:Optional[int]=None):
    global global_cluster, global_physical_mesh, global_virtual_physical_mesh


    if cluster == "local":
#        global_physical_mesh = LocalPhysicalDeviceMesh()
        print("Local_devie")
    elif cluster == "no":
        pass
    elif cluster == "ray":
        ray_addr = cluster_address if cluster_address else "auto"
        ray.init(address=ray_addr,ignore_reinit_error=True,include_dashboard=True)
        global_cluster = DeviceCluster(num_nodes, num_devices_per_node,namespace="gee_default_space")
        global_virtual_physical_mesh = (global_cluster.get_virtual_physical_mesh())
        virtual_mesh=get_global_virtual_physical_mesh()
        submesh_shapes=get_submesh_shapes(layer_num,global_cluster.num_hosts,global_cluster.num_devices//global_cluster.num_hosts)
        print("submesh",submesh_shapes)
        sliced_virtual_meshes = get_sliced_virtual_submeshes(virtual_mesh,submesh_shapes)
        virtual_mesh.get_physical_mesh_group(sliced_virtual_meshes)
        for i in range(len(virtual_mesh.launched_physical_mesh_group)):
            for j in range(i+1,len(virtual_mesh.launched_physical_mesh_group)):
                _options = {
                        "group_name": str(i)+"_"+str(j),
                        "world_size": 2,
                        "ranks": [0, 1],
                        "backend": "nccl",
                    }
                col.create_collective_group([virtual_mesh.launched_physical_mesh_group[i].workers[0],virtual_mesh.launched_physical_mesh_group[j].workers[0]], **_options)


def shutdown_global_cluster():
    global global_cluster, global_physical_mesh, global_virtual_physical_mesh
    global_cluster.delete_placement_group()
    global_cluster = None


global_cluster: DeviceCluster = None
global_physical_mesh: PhysicalDeviceMesh = None
global_virtual_physical_mesh: VirtualPhysicalMesh = None

#init_global_cluster(cluster="ray")
#submesh_shapes=[(1, 4), (1, 4), (1, 4), (1, 4)]      #层划分时获得
#virtual_mesh=get_global_virtual_physical_mesh()
#sliced_virtual_meshes = get_sliced_virtual_submeshes(virtual_mesh,submesh_shapes)
#virtual_mesh.get_physical_mesh_group(sliced_virtual_meshes)

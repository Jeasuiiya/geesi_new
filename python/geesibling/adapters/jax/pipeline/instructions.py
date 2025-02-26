from collections import namedtuple, defaultdict
from dataclasses import dataclass
import enum
from typing import Optional, Sequence
from jax.core import ClosedJaxpr, Var
import jax
from geesibling.adapters.jax.pipeline.pipeline_schedules import PipelineSchedule

mesh_executable_counter = -1
PartialGradWorkerExecutableConfig = namedtuple(
    "PartialGradWorkerExecutableConfig",
    ["exec_uuid", "pr"])

class PipelineInstType(enum.IntEnum):
    """Enum class for pipeline instruction types."""

    # Run an XLA executable
    RUN = 0
    # Run a sending task
    SEND = 1
    # Run a receiving task
    RECV = 2

@dataclass
class PipelineInstruction:
    """Base class for pipeline instructions."""

    opcode: PipelineInstType
    stage_id: int
    micro_batch_id: int
    input_vars: Optional[Var]
    output_vars: Optional[Var]
    src_rank: int
    dst_rank: int
    groupname: str
    info: str

    @classmethod
    def run(cls, stage_id, micro_batch_id, input_vars, output_vars, info=""):  # noqa
        return cls(opcode=PipelineInstType.RUN,
                   stage_id = stage_id,
                   micro_batch_id = micro_batch_id,
                   input_vars = input_vars,
                   output_vars = output_vars,
                   src_rank = None,
                   dst_rank = None,
                   groupname = None,
                   info=info)

    @classmethod
    def send(cls, micro_batch_id, output_vars, dst_rank, groupname, info=""):  # noqa
        return cls(opcode=PipelineInstType.SEND,
                   stage_id = None,
                   micro_batch_id = micro_batch_id,
                   input_vars = None,
                   output_vars = output_vars,
                   src_rank = None,
                   dst_rank = dst_rank,
                   groupname = groupname,
                   info=info)

    @classmethod
    def recv(cls, micro_batch_id, input_vars, src_rank, groupname, info=""):  # noqa
        return cls(opcode=PipelineInstType.RECV,
                   stage_id = None,
                   micro_batch_id = micro_batch_id,
                   input_vars=input_vars,
                   output_vars=None,
                   src_rank = src_rank,
                   dst_rank = None,
                   groupname = groupname,
                   info=info)


class PipelineInstEmitter:
    """Pipeline Instruction Emitter."""
    def __init__(self,*,
                 jax_all_stages: Sequence[ClosedJaxpr],
                 global_invars: Sequence[Var],
                 global_outvars: Sequence[Var],
                 mesh_group,
                 schedule: PipelineSchedule,
                 num_microbatch: int,
                 stage_to_mesh,
                 outvars_map):
        ##### Input arguments #####
        self.jax_all_stages = jax_all_stages
        self.global_invars = global_invars
        self.global_outvars = global_outvars
        self.mesh_group = mesh_group
        self.num_mesh = len(mesh_group)
        self.schedule = schedule
        self.num_microbatch = num_microbatch
        self.stage_to_mesh=stage_to_mesh
        self.out_to_meshid=defaultdict(list)  # Dict[worker -> List[PipelineInstruction]]
        self.outvars_map=outvars_map
        self.env={}

    def compile(self):
        """Compile pipeline instructions and executables for workers."""
        self.alloutvars()
#        (executable_uuids,executable_config_lists) = self._compile_computation_executables()
        instruction_lists = defaultdict(
            list)  # Dict[worker -> List[PipelineInstruction]]
        for num, sched in enumerate(self.schedule.schedules):
            if num<len(self.schedule.schedules)-1:
                self._compile_exec_one_tick(sched,instruction_lists)
        self._compile_exec_one_tick_apply(self.schedule.schedules[-1],instruction_lists)
        batch_idx=self.num_microbatch-1
        out_from_id=[]
        out_from_var={}
        for var in self.global_outvars:
            if var in self.outvars_map:
                var=self.outvars_map[var]
            if 0 not in self.env[(var,batch_idx)]:
                if self.env[(var,batch_idx)][0] not in out_from_id:
                    out_from_id.append(self.env[(var,batch_idx)][0])
                if self.env[(var,batch_idx)][0] not in out_from_var:
                    out_from_var[self.env[(var,batch_idx)][0]]=[var]
                else:
                    out_from_var[self.env[(var,batch_idx)][0]].append(var)
        dst_rank = 0
        src_rank = 1 
        for mesh_id in out_from_id:
            instruction_lists[mesh_id].append(
                PipelineInstruction.send(batch_idx, 
                                        out_from_var[mesh_id],
                                        dst_rank,
                                        "0_"+str(mesh_id),
                                        info=f"mesh {mesh_id} global outvars to mesh 0"))

            instruction_lists[0].append(
                PipelineInstruction.recv(batch_idx, 
                                        out_from_var[mesh_id], 
                                        src_rank,
                                        "0_"+str(mesh_id),
                                        info=f"mesh 0 recv global outvars from mesh {mesh_id}"))

        return instruction_lists

    def alloutvars(self):
        for num,pr in enumerate(self.jax_all_stages):
            mesh_id=self.stage_to_mesh[num]
            if mesh_id not in self.out_to_meshid:
                self.out_to_meshid[mesh_id]=[i for i in pr.jaxpr.outvars]
            else:
                self.out_to_meshid[mesh_id]+=[i for i in  pr.jaxpr.outvars]
            '''if num<len(self.outvars_map):
                if self.outvars_map[num]!={}:
                    for src_var,dis_var in self.outvars_map[num].items():
                       var_index=self.out_to_meshid[mesh_id].index(src_var)
                       self.out_to_meshid[mesh_id].remove(src_var)
                       self.out_to_meshid[mesh_id].insert(var_index,dis_var)'''

    def next_mesh_executable_uuid(self):
        global mesh_executable_counter
        mesh_executable_counter = (mesh_executable_counter + 1) % (1 << 60)
        return mesh_executable_counter

    def _compile_computation_executables(self):
        """Compile executables for forward, backward, and apply_grad
        compuations."""
        #配置运行执行所需要的信息
        executable_uuids = []  # List[stage_idx -> executable_uuids]
        executable_config_lists = defaultdict(list)  # Dict[worker -> List[ExecutableConfig]]

        for stage_idx, stage in enumerate(self.jax_all_stages):
            exec_uuid = self.next_mesh_executable_uuid()
            executable_uuids.append(exec_uuid)
            mesh_idx = self.schedule.stage_placement(stage_idx)
            assert len(mesh_idx) == 1
            mesh_idx = list(mesh_idx)[0]
            pr =self.jax_all_stages[stage_idx]
            exec_config = PartialGradWorkerExecutableConfig(
                exec_uuid, pr)

            executable_config_lists[self.mesh_group[mesh_idx]].append(exec_config)
        return executable_uuids, executable_config_lists


    def _compile_exec_one_tick_apply(self, sched, instruction_lists):
        worker_tmp_instructions = {}
        for mesh in self.mesh_group:
            worker_tmp_instructions[mesh] = []
            #先判断运行指令的输入参数是否在自身的网格中（env存储该值)，来增加recv指令

        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            batch_idx, stage_idx = task
            stage = self.jax_all_stages[stage_idx]
            to_reshard_vars = []
            src_idx={}
            in_var=[]
            out_var=[]
            for invar in stage.jaxpr.invars:
                if invar in self.outvars_map:
                    invar=self.outvars_map[invar]
                in_var.append(invar)
                if invar in self.global_invars:
                    continue
                if self.env_var_at(invar, batch_idx, mesh_idx):
                    continue
                if self.env_get_var_meshes(invar, batch_idx) not in src_idx:
                    src_idx[self.env_get_var_meshes(invar, batch_idx)]=[invar]
                else:
                    src_idx[self.env_get_var_meshes(invar, batch_idx)].append(invar)
            for outvar in stage.jaxpr.outvars:
                #if stage_idx<len(self.outvars_map) and outvar in self.outvars_map[stage_idx]:
                #    outvar=self.outvars_map[stage_idx][outvar]
                out_var.append(outvar)
                self.env_var_at(outvar, batch_idx, mesh_idx)
            #增加三种指令：目的主机的变量缓存指令，源主机的发送指令，目的主机的接受指令
            for src_idx, to_reshard_vars in src_idx.items():
                self._compile_get_vars_from_mesh(to_reshard_vars,
                                                src_idx, mesh_idx,
                                                batch_idx, 
                                                instruction_lists)

            # execute
            self._compile_exec_one_mesh(mesh_idx, stage_idx, batch_idx, in_var, out_var, instruction_lists)
    def _compile_exec_one_tick(self, sched, instruction_lists):
        worker_tmp_instructions = {}
        for mesh in self.mesh_group:
            worker_tmp_instructions[mesh] = []
            #先判断运行指令的输入参数是否在自身的网格中（env存储该值)，来增加recv指令
        for mesh_idx, task in enumerate(sched):
            if not task:
                continue
            batch_idx, stage_idx = task
            stage = self.jax_all_stages[stage_idx]
            to_reshard_vars = []
            src_idx={}
            in_var=[]
            out_var=[]
            for invar in stage.jaxpr.invars:
                if invar in self.outvars_map:
                    invar=self.outvars_map[invar]
                in_var.append(invar)
                if invar in self.global_invars:
                    continue
                if self.env_var_at(invar, batch_idx, mesh_idx):
                    continue
                if self.env_get_var_meshes(invar, batch_idx) not in src_idx:
                    src_idx[self.env_get_var_meshes(invar, batch_idx)]=[invar]
                else:
                    src_idx[self.env_get_var_meshes(invar, batch_idx)].append(invar)
            for outvar in stage.jaxpr.outvars:
                #if stage_idx<len(self.outvars_map) and outvar in self.outvars_map[stage_idx]:
                #    outvar=self.outvars_map[stage_idx][outvar]
                out_var.append(outvar)
                self.env_var_at(outvar, batch_idx, mesh_idx)
            #增加三种指令：目的主机的变量缓存指令，源主机的发送指令，目的主机的接受指令

            for src_idx, to_reshard_vars in src_idx.items():
                self._compile_get_vars_from_mesh(to_reshard_vars,
                                                src_idx, mesh_idx,
                                                batch_idx, 
                                                instruction_lists)
            # execute
            self._compile_exec_one_mesh(mesh_idx, stage_idx, batch_idx, in_var, out_var, worker_tmp_instructions)

        for worker, worker_instruction in worker_tmp_instructions.items():
            instruction_lists[worker].extend(worker_instruction)


   
    def _compile_get_vars_from_mesh(self, invars, src_mesh_idx, dst_mesh_idx,
                                    batch_idx, instruction_lists, 
                                    ):
        '''创建通信器时，都是小的mesh_id的worker放在前面，例如mesh0的worker为A，mesh1的worker为B
        那么创建通信器的时候就worker列表顺序如下[A，B],那么A对应的rank为0，B对应的rank为1
        执行send指令的时候，需要判断src_mesh_id是否小于dst_mesh_id，如果小于，那么 dst_rank=1；如果大于，那么dst_rank=0;
        recv指令则为src < dst时 src_rank = 0;src > dts时 src_rank = 1'''
        if len(invars) == 0:
            return
        if src_mesh_idx < dst_mesh_idx:
            groupname = f"{src_mesh_idx}_{dst_mesh_idx}"
            dst_rank = 1
            src_rank = 0
        else:
            groupname = f"{dst_mesh_idx}_{src_mesh_idx}"
            dst_rank = 0
            src_rank = 1
        
        instruction_lists[src_mesh_idx].append(
            PipelineInstruction.send(batch_idx, 
                                     invars,
                                     dst_rank,
                                     groupname,
                                     info=f"mesh {src_mesh_idx} send vars to mesh {dst_mesh_idx}"))
        
        instruction_lists[dst_mesh_idx].append(
            PipelineInstruction.recv(batch_idx, 
                                     invars, 
                                     src_rank,
                                     groupname,
                                     info=f"mesh {dst_mesh_idx} recv vars from mesh {src_mesh_idx}"))

    def _compile_exec_one_mesh(self, mesh_idx, stage_idx, batch_idx, 
                               in_var, out_var, 
                               instruction_lists):
        instruction_lists[mesh_idx].append(
            PipelineInstruction.run(stage_idx, 
                                    batch_idx, 
                                    in_var, 
                                    out_var,
                                    info=f"micro_batch {batch_idx} stage {stage_idx}"))

    def env_var_at(self,invar, batch_idx, mesh_idx):
        if (invar,batch_idx) in self.env:
            if mesh_idx in self.env[(invar,batch_idx)]:
                return True
            else:
                self.env[(invar,batch_idx)].append(mesh_idx)
                return False
        else:
            self.env[(invar,batch_idx)]=[mesh_idx]
            return False

    def env_get_var_meshes(self,invar, batch_idx):
        for mesh_id,vars in self.out_to_meshid.items():
            if invar in vars:
                return mesh_id

from geesibling.adapters.jax.pipeline.devicecontext import init_global_cluster,get_global_virtual_physical_mesh,get_sliced_virtual_submeshes,shutdown_global_cluster
from geesibling.adapters.jax.pipeline.layer_construction import layer_level_transformation
from geesibling.adapters.jax.pipeline.primitive_def import mark_gradient
from jax._src import api
class ShardParallel():

    def __init__(self,devices,policy):
        self.method = "ShardParallel"
        self.devices=devices
        self.policy=policy


    def grad(self,*args, **kwargs):
        """This is the same as jax.grad, except that alpa inserts a
        gradient marker after the gradient computation.

        This function annotates all gradient tensors. This information is used to
        perform gradient accumulation transformation.
        If any auxiliary tensors are returned, they are averaged over mini batches
        in the same way as how the gradients are averaged.
        """

        def ret(*call_args, **call_kwargs):
            # Apply transformations (e.g., layer construction, rematerialization)
            # to the forward func
            arg_list = list(args)
            grad_func = api.grad(*arg_list, **kwargs)
            grads = grad_func(*call_args, **call_kwargs)
            return grads

        return ret


    def value_and_grad(self,*args, **kwargs):
        """This is the same as jax.value_and_grad, except that alpa inserts a
        gradient marker after the gradient computation.


        This function annotates all gradient tensors. This information is used to
        perform gradient accumulation transformation.
        If any auxiliary tensors are returned, they are averaged over mini batches
        in the same way as how the gradients are averaged.
        """

        def ret(*call_args, **call_kwargs):
            # Apply transformations (e.g., layer construction, rematerialization)
            # to the forward func
            arg_list = list(args)
            grad_func = api.value_and_grad(*arg_list, **kwargs)
            val, grads = grad_func(*call_args, **call_kwargs)
            return (val, grads)

        return ret

class PipeshardParallel():

    def __init__(self,policy,num_microbatch,layer_method,if_ray):
        self.method = "PipeshardParallel"
        self.policy=policy
        self.layer_method=layer_method
        self.num_microbatch=num_microbatch
        self.if_ray=if_ray
        self.layer_num=4
        self.flag = False
        if if_ray:
          init_global_cluster(cluster="ray",layer_num=self.layer_num)


    def grad(self,*args, **kwargs):
        """This is the same as jax.grad, except that alpa inserts a
        gradient marker after the gradient computation.

        This function annotates all gradient tensors. This information is used to
        perform gradient accumulation transformation.
        If any auxiliary tensors are returned, they are averaged over mini batches
        in the same way as how the gradients are averaged.
        """

        def ret(*call_args, **call_kwargs):
            # Apply transformations (e.g., layer construction, rematerialization)
            # to the forward func
            arg_list = list(args)
            arg_list[0] = layer_level_transformation(arg_list[0],layer_num=self.layer_num)
            grad_func = api.grad(*arg_list, **kwargs)
            grads = grad_func(*call_args, **call_kwargs)
            return mark_gradient(grads)

        return ret


    def value_and_grad(self,*args, **kwargs):
        """This is the same as jax.value_and_grad, except that alpa inserts a
        gradient marker after the gradient computation.


        This function annotates all gradient tensors. This information is used to
        perform gradient accumulation transformation.
        If any auxiliary tensors are returned, they are averaged over mini batches
        in the same way as how the gradients are averaged.
        """

        def ret(*call_args, **call_kwargs):
            # Apply transformations (e.g., layer construction, rematerialization)
            # to the forward func
            arg_list = list(args)
            arg_list[0] = layer_level_transformation(arg_list[0],layer_num=self.layer_num)
            grad_func = api.value_and_grad(*arg_list, **kwargs)
            val, grads = grad_func(*call_args, **call_kwargs)
            return mark_gradient((val, grads))

        return ret

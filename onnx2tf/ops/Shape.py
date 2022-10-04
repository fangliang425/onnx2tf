import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Shape

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    start = graph_node.attrs.get('start', 0)

    if start < 0:
        start += input_tensor_rank
        # Clip if start is still < 0
        start = 0 if start < 0 else start

    end = graph_node.attrs.get('end', input_tensor_rank)
    if end < 0:
        end += input_tensor_rank
        # Clip if end is still < 0
        end = 0 if end < 0 else end

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.slice(
            tf.shape(
                input=input_tensor,
                out_type=dtype,
                name=graph_node.name,
            ),
            [start],
            [end - start],
        )
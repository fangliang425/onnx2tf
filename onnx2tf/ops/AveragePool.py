import math
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from tensorflow.python.keras.layers import (
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
)
from onnx2tf.utils.colors import Color
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    calc_pads_same_pooling,
    pad_input,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """AveragePool

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
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    # 0: False, 1: True
    ceil_mode = bool(graph_node.attrs.get('ceil_mode', 0))
    # 0: False, 1: True
    count_include_pad = bool(graph_node.attrs.get('count_include_pad', 0))
    kernel_shape = graph_node.attrs['kernel_shape']
    spatial_size = len(kernel_shape)
    x_rank = spatial_size + 2
    strides = graph_node.attrs.get('strides', [1] * spatial_size)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    input_tensor_shape = input_tensor.shape
    is_known_shape = None not in input_tensor_shape

    pads = graph_node.attrs.get('auto_pad', 'NOTSET')
    if pads == 'NOTSET':
        pads = graph_node.attrs.get('pads', [0] * spatial_size * 2)
        if is_known_shape and pads != [0] * spatial_size * 2:
            in_shape = input_tensor.get_shape()
            same_paddings = calc_pads_same_pooling(
                in_spatial_shape=in_shape[1:x_rank - 1],
                kernel_shape=kernel_shape,
                strides=strides,
                dilations=dilations,
                padding='SAME_UPPER',
                is_known_shape=is_known_shape,
            )
            if pads == same_paddings:
                pads = 'SAME_UPPER'

    is_explicit_padding = type(pads) is list
    padding_ = ''

    if is_explicit_padding or pads == 'SAME_LOWER' or (pads == 'SAME_UPPER' and count_include_pad):
        # pad the input
        padded_tensor = pad_input(
            input_tensor=input_tensor,
            is_known_shape=is_known_shape,
            kernel_shape=kernel_shape,
            ceil_mode=ceil_mode,
            spatial_size=spatial_size,
            strides=strides,
            dilations=dilations,
            padding=pads,
            padding_constant=0,
        )
        padding_ = 'valid'

    elif pads == 'SAME_UPPER':
        padded_tensor = input_tensor
        padding_ = 'same'

    else:
        padded_tensor = input_tensor
        padding_ = 'same'

    # Workaround pads
    # Thanks, MPolaris/onnx2tflite
    # https://github.com/MPolaris/onnx2tflite/blob/abbec2606b5767de7c9e348d1a24fbcd0d013564/layers/common_layers.py#L107-L113
    calc_pads = graph_node.attrs.get('pads', [0] * spatial_size * 2)
    func = math.floor if ceil_mode == 0 else math.ceil
    for i in range(spatial_size):
        pad_shape = calc_pads[i] + calc_pads[i+spatial_size]
        output_shape_raw = (input_tensor_shape[1+i]+pad_shape-((kernel_shape[i]-1)*dilations[i]+1))/strides[i]+1
        if func(output_shape_raw) != input_tensor_shape[1+i]:
            padding_ = "valid"
            break

    tmp_pad = None
    if padding_ == "valid" and calc_pads is not None and np.sum(calc_pads) > 0:
        tmp_pad = \
            [[0,0]] + \
            [
                [pad_begin, pad_end] \
                    for pad_begin, pad_end in zip(calc_pads[0:spatial_size], calc_pads[spatial_size:len(calc_pads)])
            ] + \
            [[0,0]]
        padded_tensor = tf.pad(
            tensor=input_tensor,
            paddings=tmp_pad,
            mode='CONSTANT',
            constant_values=-np.inf,
        )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_op_type = None
    if len(kernel_shape) == 1:
        pooled_tensor = AveragePooling1D(
            pool_size=kernel_shape,
            strides=strides,
            padding=padding_.upper(),
        )(padded_tensor)
        tf_op_type = AveragePooling1D

    elif len(kernel_shape) == 2:
        if tmp_pad is None:
            pooled_tensor = AveragePooling2D(
                pool_size=kernel_shape,
                strides=strides,
                padding=padding_.upper(),
            )(padded_tensor)
            tf_op_type = AveragePooling2D
        else:
            def avg_pool(x):
                n_channel = x.shape[-1]
                patches = tf.image.extract_patches(
                    images=x,
                    sizes=[1,kernel_shape[0],kernel_shape[1],1],
                    strides=[1,strides[0],strides[1],1],
                    rates=[1,1,1,1],
                    padding='VALID',
                )
                mask = tf.math.not_equal(patches[:,:,:,0::n_channel], tf.constant(-np.inf, dtype=patches.dtype))
                mn,mh,mw,mc = mask.shape
                channel_avg_pool = []
                for c in range(n_channel):
                    patch = patches[:,:,:,c::n_channel]
                    """
                    [           -inf,            -inf,            -inf,
                                -inf,  1.76405235e+00, -1.61389785e+00,
                                -inf, -6.36614888e-02, -8.74662522e-01
                    ],
                    [           -inf,            -inf,            -inf,
                                -1.61389785e+00,  1.05000207e-02,  2.38314477e+00,
                                -8.74662522e-01, -2.21574398e-01, -3.21258937e-01
                    ],
                    [           -inf,            -inf,            -inf,
                                2.38314477e+00, -3.92828182e-02, -6.37437026e-01,
                                -3.21258937e-01,  4.61468877e-01,  1.06160017e+00
                    ],
                    :
                    """
                    patch_means_all = []
                    for i in range(mn):
                        patch_means_mh = []
                        for j in range(mh):
                            patch_means_mw = []
                            for k in range(mw):
                                patch_mask = tf.squeeze(tf.where(mask[i,j,k]), axis=1)
                                patch_part = tf.gather(patch[i,j,k], patch_mask)
                                patch_part_mean = tf.reduce_mean(patch_part)
                                patch_means_mw.append(patch_part_mean)
                            patch_means_mh.append(patch_means_mw)
                        patch_means_all.append(patch_means_mh)
                    patch_means_all_concat = tf.concat(patch_means_all, axis=1)
                    # non_zero_avg: [22, 22]
                    non_zero_avg = tf.expand_dims(patch_means_all_concat, axis=0)
                    non_zero_avg = tf.expand_dims(non_zero_avg, axis=3)
                    channel_avg_pool.append(non_zero_avg)
                concated_tensor = tf.concat(channel_avg_pool, axis=-1)
                return concated_tensor

            pooled_tensor = avg_pool(padded_tensor)

        tf_op_type = AveragePooling2D

    elif len(kernel_shape) == 3:
        pooled_tensor = AveragePooling3D(
            pool_size=kernel_shape,
            strides=strides,
            padding=padding_.upper(),
        )(padded_tensor)
        tf_op_type = AveragePooling3D

    else:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'AveragePool supports only 1D, 2D, and 3D. ' +\
            f'opname: {graph_node.name} Type: AveragePool{len(kernel_shape)}D'
        print(error_msg)
        assert False, error_msg

    tf_layers_dict[graph_node_output.name]['tf_node'] = pooled_tensor

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'x': input_tensor,
                    'pool_size': kernel_shape,
                    'strides': strides,
                    'padding': padding_,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )

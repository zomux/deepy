#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano.tensor.var import TensorVariable
from deepy.utils.map_dict import MapDict

def convert_to_theano_var(obj):
    """
    Convert neural vars to theano vars.
    :param obj: NeuralVariable or list or dict or tuple
    :return: theano var, test var, tensor found, neural var found
    """
    from deepy.core.neural_var import NeuralVariable
    if type(obj) == tuple:
        return tuple(convert_to_theano_var(list(obj)))
    if type(obj) == list:
        unpacked_list = map(convert_to_theano_var, obj)
        normal_list = []
        test_list = []
        theano_var_found = False
        neural_var_found = False
        for normal_var, tensor_found, neural_found in unpacked_list:
            normal_list.append(normal_var)
            if tensor_found: theano_var_found = True
            if neural_found: neural_var_found = True
        return normal_list, theano_var_found, neural_var_found
    elif type(obj) == dict:
        normal_map = {}
        theano_var_found = False
        neural_var_found = False
        for key in obj:
            normal_var, tensor_found, neural_found = convert_to_theano_var(obj[key])
            normal_map[key] = normal_var
            if tensor_found: theano_var_found = True
            if neural_found: neural_var_found = True
        return normal_map, theano_var_found, neural_var_found
    elif type(obj) == MapDict:
        normal_map = {}
        theano_var_found = False
        neural_var_found = False
        for key in obj:
            normal_var, tensor_found, neural_found = convert_to_theano_var(obj[key])
            normal_map[key] = normal_var
            if tensor_found: theano_var_found = True
            if neural_found: neural_var_found = True
        return MapDict(normal_map), theano_var_found, neural_var_found
    elif type(obj) == NeuralVariable:
        theano_tensor = obj.tensor
        theano_tensor.tag.last_dim = obj.dim()
        return theano_tensor, False, True
    elif type(obj) == TensorVariable:
        return obj, True, False
    elif type(obj) == slice:
        normal_args = []
        theano_var_found = False
        neural_var_found = False
        for arg in [obj.start, obj.stop, obj.step]:
            normal_var, tensor_found, neural_found = convert_to_theano_var(arg)
            normal_args.append(normal_var)
            if tensor_found: theano_var_found = True
            if neural_found: neural_var_found = True
        return slice(*normal_args), theano_var_found, neural_var_found
    else:
        return obj, False, False

def convert_to_neural_var(obj):
    """
    Convert object and a test object into neural var.
    :param obj: tensor or list or dict or tuple
    :param test_obj: NeuralVar or list or dict or tuple
    :return:
    """
    from theano.tensor.var import TensorVariable
    from deepy.core.neural_var import NeuralVariable
    if type(obj) == list:
        return [convert_to_neural_var(item) for item in obj]
    elif type(obj) == tuple:
        return tuple(convert_to_neural_var(list(obj)))
    elif type(obj) == dict:
        merged_map = {}
        for key in obj:
            merged_map[key] = convert_to_neural_var(obj[key])
        return merged_map
    elif type(obj) == MapDict:
        merged_map = {}
        for key in obj:
            merged_map[key] = convert_to_neural_var(obj[key])
        return MapDict(merged_map)
    elif type(obj) == TensorVariable:
        deepy_var = NeuralVariable(obj)
        if hasattr(obj, 'tag') and hasattr(obj.tag, 'last_dim'):
            deepy_var.output_dim = obj.tag.last_dim
        return deepy_var
    else:
        return obj

def neural_computation(original_func, prefer_tensor=False):
    """
    An annotation to enable theano-based fucntions to be called with NeuralVar.
    :param original_func:
    :param prefer_tensor: a switch to return tensors when no inputs
    :return:
    """

    def wrapper(*args, **kwargs):
        normal_args, tensor_found_in_args, neural_found_in_args = convert_to_theano_var(args)
        normal_kwargs, tensor_found_in_kwargs, neural_found_in_kwargs = convert_to_theano_var(kwargs)

        tensor_found = tensor_found_in_args or tensor_found_in_kwargs
        neural_found = neural_found_in_args or neural_found_in_kwargs

        if tensor_found and neural_found:
            raise Exception("Theano tensor variables can not be used together with neural variables.")

        normal_result = original_func(*normal_args, **normal_kwargs)

        if tensor_found or (not neural_found and prefer_tensor):
            # No neural variables are inputted, so output tensors
            return normal_result
        else:
            # Output neural variables, auto set output_dim
            result_var = convert_to_neural_var(normal_result)
            if (isinstance(normal_result, TensorVariable) and
                    hasattr(normal_result.tag, "test_value") and
                    hasattr(normal_result.tag.test_value, "shape") and
                    normal_result.tag.test_value.shape):
                result_var.output_dim = normal_result.tag.test_value.shape[-1]
            return result_var
    return wrapper

def neural_computation_prefer_tensor(original_func):
    return neural_computation(original_func, prefer_tensor=True)
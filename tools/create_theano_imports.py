#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A tool for create imports from theano.tensor.
"""

import theano.tensor

if __name__ == '__main__':
    import_names = set()
    import_names.update([name for name in dir(theano.tensor.basic) if name[0].islower() and name[0] != "_"])
    import_names.update([name for name in dir(theano.tensor.subtensor) if name[0].islower() and name[0] != "_"])
    import_names.update(["sort", "argsort", "grad"])
    import_names.remove("concatenate")

    notes = """
# This file is automatically created, never edit it directly.

from wrapper import deepy_tensor, deepy_nnet

    """
    template = """
def THEANO_NAME(*args, **kwargs):
    return deepy_tensor.THEANO_NAME(*args, **kwargs)

    """
    with open("deepy/tensor/theano_imports.py", "w") as fout:
        fout.write(notes)
        for name in import_names:
            fout.write(template.replace("THEANO_NAME", name))

    nnet_template = """
def THEANO_NAME(*args, **kwargs):
    return deepy_nnet.THEANO_NAME(*args, **kwargs)

    """

    with open("deepy/tensor/theano_nnet_imports.py", "w") as fout:
        fout.write(notes)
        for name in [name for name in dir(theano.tensor.nnet) if name[0].islower() and name[0] != "_"]:
            fout.write(nnet_template.replace("THEANO_NAME", name))
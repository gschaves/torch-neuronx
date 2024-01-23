# ==============================================================================
# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================

import os
import sys
import glob
import json
import uuid
import torch
import ctypes
import shutil
import tempfile
import subprocess
import torch.utils.cpp_extension
from pathlib import Path
from collections import defaultdict
from torch_neuronx.xla_impl.ops import CustomCppOp
from typing import Callable


def _call_dll_func(func: Callable, *args):
    """
    Helper to call CustomOp CPU dll that has form

    char* func(*args, char* result, size_t max_length)

    where the function write the result into `result`, return nullptr
    on exceeded `max_length`. Returns the interpreted python string

    """
    MAX_LENGTH = 128
    result_buffer = ctypes.create_string_buffer(MAX_LENGTH)

    func.argtypes = [type(arg) for arg in args] + [ctypes.c_char_p, ctypes.c_size_t]
    func.restype = ctypes.c_char_p

    result = func(*args, result_buffer, ctypes.sizeof(result_buffer))
    return result.decode()


def _getNeuronLibInfo(path):
    """
    Loads operator registraion info from given library
    """

    neuronOps = defaultdict(set)
    lib = ctypes.CDLL(path)  # load the library in this process

    num_ns = lib.getNumNamespaces()
    # iterate over the namespaces registered
    for i in range(num_ns):
        idx = ctypes.c_int(i)
        # get the namespace name string
        ns = _call_dll_func(lib.getNamespace, idx)

        num_ops = lib.getNumNeuronOps(idx)
        # iterate over the custom ops registered in this namespace
        for j in range(num_ops):
            cnt = ctypes.c_int(j)
            # get the operator name string
            op = _call_dll_func(lib.getNeuronOpName, idx, cnt)
            fn = _call_dll_func(lib.getNeuronFnName, idx, cnt)

            # initialize the namespace
            hasattr(torch.ops, ns)
            # initialize the op
            hasattr(torch.ops.__getattribute__(ns), op)

            # get the function pointer
            shape = torch.ops.__getattribute__(ns).__getattribute__(op)

            # store namespace::op mapping
            neuronOps[ns].add((op, fn, shape))

    return neuronOps


def load(
    name,
    compute_srcs,
    shape_srcs,
    extra_cflags=[],
    extra_include_paths=[],
    build_directory=None,
    multicore=False,
    verbose=False,
):
    """
    Builds and loads Custom C++ Operators into Neuron

    Parameters:
    `name` : name of the library to create
    `compute_srcs` : C++ file (or list of files) to compile for gpsimd
    `shape_srcs` : C++ file (or list of files) to compile for host
    `extra_cflags` : list of options to pass during compilation
    `extra_include_paths` : list of include paths to use during compilation
    `build_directory` : directory to build library in
    `multicore`: compile the library in multicore mode, 1 library for each core
    `verbose` : show logging
    """

    if isinstance(compute_srcs, str):
        compute_srcs = [compute_srcs]
    if isinstance(shape_srcs, str):
        shape_srcs = [shape_srcs]

    # load build script
    build_module_path = "/opt/aws/neuron/gpsimd/script/"
    sys.path.append(build_module_path)
    try:
        # TODO: We need to check the version of build_custom_op,
        # and increment the version of SundaCustomOpLibrary in CMake before next release
        import build_custom_op
    except ModuleNotFoundError as error:
        print(error.__class__.__name__ + ": " + error.msg)
        raise Exception(
            f"unable to find neuron custom op build script in {build_module_path}, please install custom-ops rpm"
        )

    # initialize metadata
    metadata = defaultdict(dict)
    metadata["versions"] = build_custom_op.versions

    # create the temp directory
    tmp_dir = tempfile.mkdtemp()
    if verbose:
        print(f"Using temp dir: {tmp_dir}")

    # get path to include that has torchneuron/register.h
    includes = os.path.join(os.path.dirname(os.path.dirname(__file__)), "include")

    # build the shape function for x86 to run on the host
    shapefcn = f"lib{name}shapefcn"
    torch.utils.cpp_extension.load(
        name=shapefcn,
        sources=shape_srcs,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths + [includes],
        is_python_module=False,
        build_directory=tmp_dir,
        verbose=verbose,
    )
    # after this call (above^^), shape functions have been loaded by PT as normal PT custom ops

    metadata["shapelib"] = shapefcn + ".so"

    # find the ops registered in the shape function library created above
    shapelib = os.path.join(tmp_dir, shapefcn + ".so")
    neuronOps = _getNeuronLibInfo(shapelib)

    # process PyTorch shape functions
    schemas = []
    fn_names = []
    metadata["ops"] = defaultdict(dict)
    for ns in neuronOps:
        metadata["ops"][ns] = defaultdict(dict)
        for op, fn, shape in neuronOps[ns]:
            fn_names.append(fn)
            # get the function signature
            schema = torch._C._get_schema(f"{ns}::{op}", "")
            # convert to string to pass to build script
            args = ", ".join([f"{arg.type} {arg.name}" for arg in schema.arguments])
            rets = ", ".join([f"{ret.type} {ret.name}" for ret in schema.returns])
            schema_str = f"{schema.name}({args}) -> ({rets})"
            schemas.append(schema_str)
            if verbose:
                print(f"schemas for {ns}::{op}: {schema}")
            metadata["ops"][ns][op] = {"fn": fn, "schema": schema_str}

    # build the compute function to run on device
    # For backward compatibility, only pass in multicore flag when it is set to True
    # TODO: We might want to remove the if statement and have a 1-to-1 version match of SundaCustomOpLibrary
    # and torch_neuronx
    if multicore:
        # computelibs returned here is a list
        computelibs = build_custom_op.compile(
            name=f"lib{name}customop.so",
            fn_names=fn_names,
            src_files=compute_srcs,
            schema=schemas,
            build_directory=tmp_dir,
            cflags=extra_cflags,
            multicore=multicore,
            verbose=verbose,
        )
    else:
        # computelibs returned here can be a list or a single file name
        # Check and wrap it into a list if not a list
        computelibs = build_custom_op.compile(
            name=f"lib{name}customop.so",
            fn_names=fn_names,
            src_files=compute_srcs,
            schema=schemas,
            build_directory=tmp_dir,
            cflags=extra_cflags,
            verbose=verbose,
        )
        if not isinstance(computelibs, list):
            computelibs = [computelibs]

    # add info about computelib to metadata
    metadata["computelibs"]["names"] = [
        os.path.basename(computelib) for computelib in computelibs
    ]
    # Record the md5 value of the first library
    computelib_md5 = (
        subprocess.check_output(f"md5sum {computelibs[0]}", shell=True)
        .decode()
        .split()[0]
    )
    metadata["computelib"]["id"] = computelib_md5

    if build_directory:
        # write out the metadata
        meta_file = os.path.join(tmp_dir, "metadata.json")
        with open(meta_file, "w") as fp:
            json.dump(metadata, fp)

        # build the combined lib
        comblib = os.path.join(tmp_dir, f"lib{name}.so")
        cmd = f'ar r {comblib} {shapelib} {" ".join(computelibs)} {meta_file}'
        if verbose:
            print(cmd)
        p = subprocess.run(cmd, shell=True, capture_output=True)
        if p.returncode:
            if verbose:
                print(f"stdout: {p.stdout}")
                print(f"stderr: {p.stderr}")
            raise Exception("Error building customOp library")
        # copy from tmp dir to user-specified dir
        shutil.copy2(comblib, build_directory)

    for ns in neuronOps:
        for op_name, fn_name, shape_fcn in neuronOps[ns]:
            # create the customOp class object
            op = CustomCppOp(
                fn_name,
                shape_fcn,
                ",".join(computelibs),
                computelib_md5,
                metadata["versions"]["ulib_to_ucode_version"],
                metadata["versions"]["ulib_to_isa_version"],
            )
            # make op available like normal PT customOps
            torch.ops.__getattribute__(ns).__setattr__(op_name, op)


def load_library(libname, verbose=False):
    """
    Loads Custom C++ Operator into framework

    Parameters:
    `libname`: the library built by calling the `load` API
    `verbose` : print debug messages
    """
    if not isinstance(libname, str):
        stype = type(libname)
        raise Exception(f"libname must be a string, but found: {stype}")
    if not os.path.isfile(libname):
        raise Exception(f'unable to find library file: "{libname}"')

    # create the temp directory
    md5 = subprocess.check_output(f"md5sum {libname}", shell=True).decode().split()[0]
    path = Path("/var/tmp/neuron-custom-ops") / md5

    # check if this library has already been used, if not unpack it
    if not path.exists():
        path.mkdir(
            parents=True,
        )

        shutil.copy2(libname, path)

        cmd = f"cd {path} && ar x {libname}"
        if verbose:
            print(cmd)
        p = subprocess.run(cmd, shell=True, capture_output=True)
        if p.returncode:
            if verbose:
                print(f"stdout: {p.stdout}")
                print(f"stderr: {p.stderr}")
            raise Exception("Error extracting customOp library")

    # load metadata
    meta_file = path / "metadata.json"
    if not os.path.isfile(meta_file):
        raise Exception(f"Unable to find metadata in {libname}")
    with open(meta_file, "r") as fp:
        metadata = json.load(fp)

    # find shapelib
    shapelib = path / metadata["shapelib"]
    if not os.path.isfile(shapelib):
        raise Exception(f"Unable to find shapelib in {libname}")

    # find computelib
    computelibs = metadata["computelibs"]["names"]
    computelibs = [str(path / computelib) for computelib in computelibs]
    for lib in computelibs:
        if not os.path.isfile(lib):
            raise Exception(f"Unable to find computelib {lib} in {libname}")

    # load the library
    torch.ops.load_library(shapelib)

    # find the ops registered in the shape function library created above
    neuronOps = _getNeuronLibInfo(shapelib)

    for ns in neuronOps:
        for op_name, fn_name, shape_fcn in neuronOps[ns]:
            # create the customOp class object
            op = CustomCppOp(
                fn_name,
                shape_fcn,
                ",".join(computelibs),
                metadata["computelib"]["id"],
                metadata["versions"]["ulib_to_ucode_version"],
                metadata["versions"]["ulib_to_isa_version"],
            )
            # make op available like normal PT customOps
            torch.ops.__getattribute__(ns).__setattr__(op_name, op)

# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
import typing
import torch


def is_inline_tensor(node):
    """
    Checks if a node is an inline tensor. We assume inline tensors have the
    following sequence of nodes in a TorchScript graph:
        `prim::Constant` ->  `aten::to` -> `aten::detach` -> `aten::<operator>`
    """

    kind = node.kind()

    if kind != "prim::Constant":
        return False

    # Check that this is a tensor
    if node.outputsAt(0).type().kind() != "TensorType":
        return False

    # Check that the node is used once
    if len(list(node.outputsAt(0).uses())) != 1:
        return False

    # Check that node's consumer is an `aten::to` op
    consumer = node.outputsAt(0).uses()[0].user
    if consumer.kind() != "aten::to":
        return False

    # Check that the `aten::to`` op is used once
    if len(list(consumer.outputsAt(0).uses())) != 1:
        return False

    # Check that `aten::to`` op's consumer is an `aten::detach` op
    to_consumer = consumer.outputsAt(0).uses()[0].user
    if to_consumer.kind() != "aten::detach":
        return False

    # Check that the `aten::detach` op is used once
    if len(list(to_consumer.outputsAt(0).uses())) != 1:
        return False

    # We have found an inline tensor node
    return True


def remove_inline_tensors(trace: torch.ScriptFunction) -> typing.List[torch.Tensor]:
    """
    Removes inline tensors (and their associated nodes) from a TorchScript
    graph; a TorchScript input node is insesrted into the graph for each inline
    tensor that is removed. The inline tensor values are extracted and returned
    by this function.

    We assume inline tensors have the following sequence of nodes in
    a TorchScript graph:
        `prim::Constant` ->  `aten::to` -> `aten::detach` -> `aten::<operator>`

    The following is an example of a TorchScript graph containing an inline
    tensor:
    ```
    graph(%self : __torch__.Network,
        %tensor : Float(10, 3, strides=[3, 1], requires_grad=0, device=xla:1)):
    %4 : Float(3, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value= 1  2  3 [ CPUFloatType{3} ]]()
    %5 : Device = prim::Constant[value="cpu"]()
    %6 : int = prim::Constant[value=6]()
    %7 : bool = prim::Constant[value=0]()
    %8 : bool = prim::Constant[value=0]()
    %9 : NoneType = prim::Constant()
    %10 : Float(3, strides=[1], requires_grad=0, device=cpu) = aten::to(%4, %5, %6, %7, %8, %9)
    %inline_constant : Float(3, strides=[1], requires_grad=0, device=cpu) = aten::detach(%10)
    %12 : Float(10, 3, strides=[3, 1], requires_grad=0, device=xla:1) = aten::div(%tensor, %inline_constant)
    return (%12)
    ```

    After this pass is run, the resulting graph will be the following:
    ```
    graph(%self : __torch__.Network,
        %tensor : Float(10, 3, strides=[3, 1], requires_grad=0, device=xla:1),
        %13 : Tensor):
    %12 : Float(10, 3, strides=[3, 1], requires_grad=0, device=xla:1) = aten::div(%tensor, %13)
    return (%12)
    ```

    Args:
        trace (torch.ScriptFunction): A traced graph that will be modified
        in-place.

    Returns:
        A list of inline tensor values corresponding to each inline tensor node
        that is removed by this pass.
    """
    graph = trace.graph

    inline_nodes = list()
    inline_tensor_values = list()

    # Find all inline tensor nodes
    for node in graph.nodes():
        if node is None:
            continue

        if is_inline_tensor(node):
            inline_nodes.append(node)
            inline_tensor_values.append(
                node.outputsAt(0).toIValue()
            )  # Extract inline tensor values

    for node in inline_nodes:
        # Get new constant node to replace inline node
        replacement = graph.addInput()

        # Inline nodes are consumed as follows:
        # `prim::Constant` ->  `aten::to` -> `aten::detach` -> `aten::<operator>`
        # Thus, we need to update the downstream `aten::<operator>` to consume
        # the new replacement node
        aten_to = node.outputsAt(0).uses()[0].user
        aten_detach = aten_to.outputsAt(0).uses()[0].user
        aten_detach.outputsAt(0).replaceAllUsesWith(replacement)

    # Remove dead code
    torch._C._jit_pass_dce(graph)

    # Validate graph has no problems
    graph.lint()

    return inline_tensor_values

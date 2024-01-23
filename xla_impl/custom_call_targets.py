# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================

# Shared with neuronx-cc
AwsNeuronArgMax = 'AwsNeuronArgMax'
AwsNeuronArgMin = 'AwsNeuronArgMin'
AwsNeuronCustomOp = 'AwsNeuronCustomOp'
AwsNeuronGelu = 'AwsNeuronGelu'
AwsNeuronGeluBackward = 'AwsNeuronGeluBackward'
AwsNeuronNearestNeighbor2d = 'ResizeNearest'  # They share the same target name as TPU
AwsNeuronNearestNeighbor2dBackward = 'ResizeNearestGrad'
AwsNeuronSoftmax = 'AwsNeuronSoftmax'
AwsNeuronSoftmaxBackward = 'AwsNeuronSoftmaxBackward'
AwsNeuronTopK = 'AwsNeuronTopK'
AwsNeuronTransferWithStaticRing = 'AwsNeuronTransferWithStaticRing'
AwsNeuronRmsNorm = 'AwsNeuronRmsNorm'
AwsNeuronRmsNormBackward = 'AwsNeuronRmsNormBackward'
# The internal representation of triton kernel in compiler is AwsNeuronCustomNativeKernel
AwsNeuronTritonKernel = 'AwsNeuronCustomNativeKernel'

# Used in torch-neuronx only
TorchNeuronInlineMark = 'TorchNeuronInlineMark'
TorchNeuronOutputMark = 'TorchNeuronOutputMark'
TorchNeuronStartBackward = 'TorchNeuronStartBackward'
TorchNeuronUnloadPriorModels = 'TorchNeuronUnloadPriorModels'

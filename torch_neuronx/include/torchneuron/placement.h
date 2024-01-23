/* Copyright 2022, Amazon.com, Inc. or its affiliates. All Rights Reserved. */

#pragma once

#include <torch/script.h>

namespace neuron {

/**
 * Set the NeuronCore start/count for all Neuron models in a torch Module.
 *
 * This function should only be used when an application requires that models
 * are loaded to specific NeuronCores. By default, this function recursively
 * moves all Neuron submodules to the given cores. In the case where a Module
 * has multiple Neuron subgraphs which each require specific placement, each
 * submodule should be accessed via the module attributes and placed using
 * this function.
 *
 * @param module A torch module which contains one or more Neuron subgraphs.
 * @param start_nc The starting NeuronCore index where the Module be placed. The
 *      value -1 automatically loads to the optimal NeuronCore (least used).
 *      Note that this index is alwasy relative to neuron cores visible to this
 *      process.
 * @param nc_count The number of NeuronCores to use. The value -1 will load the
 *      number of required cores (1 for most models, > 1 for when using
 *      neuron-core-pipeline). Any value greater than than number of cores
 *      required by the model will replicate the model to multiple NeuronCores.
 **/
void set_neuron_cores(torch::jit::Module& module, int32_t start_nc=-1, int32_t nc_count=-1);

/**
 * Loads all Neuron models in a torch Module to as many NeuronCores as possible.
 *
 * This loads each Neuron model within a Module to multiple NeuronCores
 * without requiring multiple calls to `torch::jit:load`. This allows a single
 * Module to use multiple NeuonCores for concurrent threadsafe inferences.
 * Requests use a simple round-robin strategy to distribute across NeuronCores.
 *
 * The main benefit of using this function is that it simplifies application
 * code to distribute requests to multiple NeuronCores. It also avoids loading
 * multiple torchscript models which can be slow and consume excess memory.
 *
 * @param module A torch module which contains one or more Neuron subgraphs.
**/
void set_multicore(torch::jit::Module& module);

} // namespace neuron

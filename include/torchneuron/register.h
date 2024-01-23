#include <torch/torch.h>

#include <vector>
#include <map>

#pragma once

/*******************************************************************
 * This file contains the functionality to double register customOps
 * both for Neuron tracking and PyTorch
 */

class NeuronNamespace;
typedef void (*NeuronLibraryInit)(NeuronNamespace&);

/*
 * class NeuronNamespace
 * A class to track the custom operators being registered, and
 * also call regular PyTorch custom op registration APIs
 */
class NeuronNamespace {
public:
  NeuronNamespace(const char* ns, const char* file, uint32_t line) : name(ns), lib(torch::Library::DEF, ns, c10::nullopt, file, line) {}

  std::string name; //namespace name (ie. my_ops from my_ops::op_name)
  torch::Library lib; //regular PyTorch library object
  std::vector<std::pair<std::string, std::string> > ops; //op and compute names registered in this namespace

  /* this function imitates the regular torch::Library.def() from:
   * https://github.com/pytorch/pytorch/blob/master/torch/library.h#L625-L631
   * it saves the op name separately, and also calls the regular PyTorch 
   * library `def` function.
   *
   * Arguments:
   *    op_name - string name of the operator to use from PyTorch
   *    shape_fcn - function pointer to shape function
   *    compute_name - string name of the compute function that will get compiled for gpsimd
   *
   * Note: during gpsimd compilation, if this 3rd arg `compute_name` differs from the actual
   *    function name compilation will fail.
   */ 
  template<typename Func>
  void def(const char* op_name, Func&& shape_fcn, const char* compute_name) {
    ops.emplace_back(op_name, compute_name);
    lib.def(op_name, shape_fcn);
  }
};

/*
 * class NeuronOpRegistry
 * This is the singleton operator registry for a single customOp library
 * all ops/namespaces will be tracked here
 */
class NeuronOpRegistry {
public:
  static NeuronOpRegistry* get() {
    static NeuronOpRegistry inst;
    return &inst;
  }

  // entry point for creating a new namespace
  NeuronNamespace& addNS(const char* ns, NeuronLibraryInit init, const char* file, uint32_t line) {
    if(entries.count(ns) == 0) {
      // create a new namespace if it doesnt exist yet
      // the piecewise/forward is needed to pass the arguments to the NeuronNamespace constructor
      // without actually constructing an object, to avoid the move/dtor and have a single instance
      entries.emplace(std::piecewise_construct, std::forward_as_tuple(ns), std::forward_as_tuple(ns, file, line));
    }

    // get the namespace object for this namespace
    NeuronNamespace& nns = entries.at(ns);
    // call the user-written initializer function that will then call `def` for each op
    init(nns);

    return nns;
  }

  int size() {
    return entries.size();
  }

  /* 
   * this function is needed since we'll be accessing the registered namespaces/ops
   * from python and need to be able to access in an index-based order, since we
   * store namespace::op in a map, we'll create a vector of namespace names to 
   * traverse through
   */
  void sort() {
    if(names.size() == 0) {
      for(auto &e : entries) {
	names.push_back(e.first);
      }
    }
  }
  
  std::map<std::string, NeuronNamespace> entries;
  std::vector<std::string> names;
};

/*
 * library definition macro similar to PyTorch's TORCH_LIBRARY from:
 * https://github.com/pytorch/pytorch/blob/master/torch/library.h#L882-L891
 * we do the same things:
 *   (1) declare the user-defined init function NEURON_LIBRARY_init,
 *   (2) call the static namespace creation `addNS`,
 *   (3) and define the function signature for NEURON_LIBRARY_init that the user will implement
 */
#define NEURON_LIBRARY(ns, m)						\
  static void NEURON_LIBRARY_init_##ns(NeuronNamespace&);		\
  static const NeuronNamespace& NEURON_NAMESPACE_##ns =			\
    NeuronOpRegistry::get()->addNS(                                     \
                                   #ns,                                 \
                                   &NEURON_LIBRARY_init_##ns,           \
                                   __FILE__,                            \
                                   __LINE__);                           \
  void NEURON_LIBRARY_init_##ns(NeuronNamespace& m)

/**
 * Helper function to check string length and copy the result into buffer
 * return nullptr on exceed max_length with a error message using `debug_name`
*/
char* _check_and_copy_result(char* dst, const std::string src, size_t max_length, const char* debug_name) {
  if (src.size() > max_length) {
    std::cerr << "Error: in " << debug_name << ": " << src << 
      " exceeded maximum length " << max_length << std::endl;
    return nullptr;
  }
  strncpy(dst, src.c_str(), max_length);
  return dst;
}


/*
 * Functions that we'll use from Python to access the operator registration
 * info to identify the neuron custom ops in this library
 */
extern "C" {
  // returns the number of namespaces registered
  int getNumNamespaces() {
    return NeuronOpRegistry::get()->size();
  }

  // returns the namespace name for the namespace at index `i`,
  // copies the name into result with a max length of max_length,
  // return nullptr on length exceeded max_length
  const char* getNamespace(int i, char* result, size_t max_length) {
    NeuronOpRegistry* reg = NeuronOpRegistry::get();
    reg->sort();
    return _check_and_copy_result(result, reg->names[i], max_length, __func__);
  }

  // returns the number of ops registered in namespace `i`
  int getNumNeuronOps(int i) {
    NeuronOpRegistry* reg = NeuronOpRegistry::get();
    reg->sort();
    NeuronNamespace& nns = reg->entries.at(reg->names[i]);
    return nns.ops.size();
  }

  // returns the op name at index `j` in namespace `i`,
  // copies the name into result with a max length of max_length
  // return nullptr on length exceeded max_length
  const char* getNeuronOpName(int i, int j, char* result, size_t max_length) {
    NeuronOpRegistry* reg = NeuronOpRegistry::get();
    reg->sort();
    NeuronNamespace& nns = reg->entries.at(reg->names[i]);
    return _check_and_copy_result(result, nns.ops[j].first, max_length, __func__);
  }

  // returns the compute function name at index `j` in namespace `i`,
  // copies the name into result with a max length of max_length
  // return nullptr on length exceeded max_length
  const char* getNeuronFnName(int i, int j, char* result, size_t max_length) {
    NeuronOpRegistry* reg = NeuronOpRegistry::get();
    reg->sort();
    NeuronNamespace& nns = reg->entries.at(reg->names[i]);
    return _check_and_copy_result(result, nns.ops[j].second, max_length, __func__);
  }

  // returns the number of PyTorch ops registered
  int getNumTorchOps() {
    auto ops = torch::jit::getAllOperators();
    return ops.size();
  }

  // returns the "namespace::op_name" string for the PyTorch op at index `i`,
  // copies the name into result with a max length of max_length
  // return nullptr on length exceeded max_length
  const char* getTorchOpName(int i, char* result, size_t max_length) {
    auto ops = torch::jit::getAllOperators();
    return _check_and_copy_result(result, ops[i]->schema().name(), max_length, __func__);
  }
}

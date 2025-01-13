#ifndef CONIFER_CPP_H__
#define CONIFER_CPP_H__
#include "nlohmann/json.hpp"
#include <cassert>
#include <fstream>
#include <iostream>

namespace conifer{

/* ---
* Balanced tree reduce implementation.
* Reduces an array of inputs to a single value using the template binary operator 'Op',
* for example summing all elements with Op_add, or finding the maximum with Op_max
* Use only when the input array is fully unrolled. Or, slice out a fully unrolled section
* before applying and accumulate the result over the rolled dimension.
* Required for emulation to guarantee equality of ordering.
* --- */
constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }

template <int B>
constexpr int pow(int x) {
  return x == 0 ? 1 : B * pow<B>(x - 1);
}

constexpr int pow2(int x) { return pow<2>(x); }

constexpr int ceillog2(int x) { return (x < 2) ? 0 : 1 + ceillog2(x / 2); }

template <class T, class Op>
T reduce(std::vector<T> x, Op op) {
  int N = x.size();
  int leftN = pow2(floorlog2(N - 1)) > 0 ? pow2(floorlog2(N - 1)) : 0;
  //static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;
  if (N == 1) {
    return x.at(0);
  } else if (N == 2) {
    return op(x.at(0), x.at(1));
  } else {
    std::vector<T> left(x.begin(), x.begin() + leftN);
    std::vector<T> right(x.begin() + leftN, x.end());
    return op(reduce<T, Op>(left, op), reduce<T, Op>(right, op));
  }
}

template<class T>
class OpAdd {
public:
  T operator()(T a, T b) { return a + b; }
};

template<class T>
class OpMax {
public:
  T operator()(T a, T b) { return (a>b)? a:b; }
};

template<class T, class U>
class DecisionTree{

private:
  std::vector<int> feature;
  std::vector<int> children_left;
  std::vector<int> children_right;
  std::vector<T> threshold_;
  std::vector<U> value_;
  std::vector<double> threshold;
  std::vector<double> value;

public:

  U decision_function(std::vector<T> x) const{
    /* Do the prediction */
    int i = 0;
    while(feature[i] != -2){ // continue until reaching leaf
      bool comparison = x[feature[i]] <= threshold_[i];
      i = comparison ? children_left[i] : children_right[i];
    }
    return value_[i];
  }

  void init_(){
    /* Since T, U types may not be readable from the JSON, read them to double and the cast them here */
    std::transform(threshold.begin(), threshold.end(), std::back_inserter(threshold_),
                   [](double t) -> T { return (T) t; });
    std::transform(value.begin(), value.end(), std::back_inserter(value_),
                   [](double v) -> U { return (U) v; });
  }

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DecisionTree, feature, children_left, children_right, threshold, value);

}; // class DecisionTree

template<class T, class U, bool useAddTree = false>
class BDT{

private:

  unsigned int n_classes;
  unsigned int n_trees;
  unsigned int n_features;
  double norm;
  std::vector<double> init_predict;
  std::vector<U> init_predict_;
  // vector of decision trees: outer dimension tree, inner dimension class
  std::vector<std::vector<DecisionTree<T,U>>> trees;
  OpAdd<U> add;

  // For softmax
  int table_size = 1024;
  OpMax<U> max;

public:

  // Define how to read this class to/from JSON
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(BDT, n_classes, n_trees, n_features, norm, init_predict, trees);

  BDT(std::string filename){
    /* Construct the BDT from conifer cpp backend JSON file */
    std::ifstream ifs(filename);
    nlohmann::json j = nlohmann::json::parse(ifs);
    from_json(j, *this);
    /* Do some transformation to initialise things into the proper emulation T, U types */
    if(n_classes == 2) n_classes = 1;
    std::transform(init_predict.begin(), init_predict.end(), std::back_inserter(init_predict_),
                   [](double ip) -> U { return (U) ip; });
    for(unsigned int i = 0; i < n_trees; i++){
      for(unsigned int j = 0; j < n_classes; j++){
        trees.at(i).at(j).init_();
      }
    }
  }

  std::vector<U> decision_function(std::vector<T> x) const{
    /* Do the prediction */
    assert("Size of feature vector mismatches expected n_features" && x.size() == n_features);
    std::vector<U> values;
    std::vector<std::vector<U>> values_trees;
    values_trees.resize(n_classes);
    values.resize(n_classes, U(0));
    for(unsigned int i = 0; i < n_classes; i++){
      std::transform(trees.begin(), trees.end(), std::back_inserter(values_trees.at(i)),
                     [&i, &x](auto tree_v){ return tree_v.at(i).decision_function(x); });
      if(useAddTree){
        values.at(i) = init_predict_.at(i);
        values.at(i) += reduce<U, OpAdd<U>>(values_trees.at(i), add);
      }else{
        values.at(i) = std::accumulate(values_trees.at(i).begin(), values_trees.at(i).end(), U(init_predict_.at(i)));
      }
      values.at(i) *= (U) norm;
    }

    return values;
  }

  std::vector<double> _decision_function_double(std::vector<double> x) const{
    /* Do the prediction with data in/out as double, cast to T, U before prediction */
    std::vector<T> xt;
    std::transform(x.begin(), x.end(), std::back_inserter(xt),
                   [](double xi) -> T { return (T) xi; });
    std::vector<U> y = decision_function(xt);
    std::vector<double> yd;
    std::transform(y.begin(), y.end(), std::back_inserter(yd),
                [](U yi) -> double { return (double) yi; });
    return yd;
  }

  float softmax_real_val_from_idx(unsigned i) const {
    // Treat the index as the top N bits
    //int N = ceillog2(table_size); // number of address bits for table
    //if (floorlog2(i) <= N) { x = i; }
    //else { x = (i >> (floorlog2(i)-N)) & ((1<<N)-1);}

    float x = i/16; // scale down?
    return x;
  }

  unsigned softmax_idx_from_real_val(T x) const {
    // Slice the top N bits to get an index into the table
    //int N = ceillog2(table_size);
    //unsigned y = (int(x) >> (floorlog2(x)-N)) & ((1<<N)-1); // FIX ME: shouldn't be int(x)

    unsigned y = int(16*x); // scale up
    return { (y >= table_size) ? (table_size-1):y };
  }

  std::vector<U> init_exp_table(std::vector<U> lut_exp) const {
    for (unsigned int i=0; i<table_size; i++) {
        float x = softmax_real_val_from_idx(i);
        //std::cout << "x = softmax_real_val_from_idx(i) = " << x << ", ";
        //U exp_x = exp(x); // scale down here too?
        U exp_x = exp(x/32); // scale down here too
        lut_exp[i] = exp_x;
        //std::cout << "lut_exp[" << i << "] = " << exp_x << ", ";
    }
    std::cout << "\nLUT initialized!" << std::endl;

    return lut_exp;
  }

  std::vector<U> init_invert_table(std::vector<U> lut_inv) const {
    for (unsigned int i=0; i<table_size; i++) {
        float x = softmax_real_val_from_idx(i);
        U inv_x = 1./x;
        lut_inv[i] = inv_x;
    }

    return lut_inv;
  }

  std::vector<U> softmax(std::vector<T> x) const{
    /* Do the softmax activation here */
    /*
    std::vector<U> y_softmax;
    std::transform(y_raw.begin(), y_raw.end(), std::back_inserter(y_softmax),
                [](T yi) -> U { return (U) yi/2.; });

    return y_softmax;
    */

    #pragma HLS pipeline
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    std::vector<U> lut_exp = std::vector<U>(table_size);
    std::vector<U> lut_inv = std::vector<U>(table_size);
#else
    static bool initialized = false;
    static std::vector<U> lut_exp = std::vector<U>(table_size);
    static std::vector<U> lut_inv = std::vector<U>(table_size);
#endif

    if (!initialized) {
        std::cout << "LUT not initialized!" << std::endl;
        lut_exp = init_exp_table(lut_exp);
        lut_inv = init_invert_table(lut_inv);

        initialized = true;
    }

    // Find the max and compute all delta(x_i, x_max)
    float x_max = reduce<float, OpMax<float>>(x, max);
    std::vector<float> d_xi_xmax(x.size());
    for (unsigned i=0; i < x.size(); i++) {
        #pragma HLS unroll
        d_xi_xmax[i] = x_max - x[i];
        //d_xi_xmax[i] = x[i] - x_max;
    }

    // Calculate all the e^x's
    std::vector<U> exp_res(x.size());
    #pragma HLS array_partition variable=exp_res complete
    U exp_sum(0);

    for (unsigned i=0; i<x.size(); i++) {
        #pragma HLS unroll
        //std::cout << "d_xi_xmax[i] = " << d_xi_xmax[i] << std::endl;
        unsigned d_xi = softmax_idx_from_real_val(d_xi_xmax[i]);
        exp_res[i] = lut_exp[d_xi];
        //std::cout << "d_xi = " << d_xi << ", exp from lut = " << exp_res[i] << std::endl;
    }

    // Explicitly sum the results with an adder tree
    exp_sum = reduce<U, OpAdd<U>>(exp_res, add);

    std::vector<U> res(x.size());
    //std::cout << "\nexp_sum = " << exp_sum << std::endl;
    float inv_exp_sum = lut_inv[softmax_idx_from_real_val(exp_sum)];
    for (unsigned int i=0; i<x.size(); i++) {
        #pragma HLS unroll
        res[i] = exp_res[i] * inv_exp_sum;
        //res[i] = exp_res[i] / exp_sum;
    }

    // Convert the U type output back to float
    std::vector<float> y_softmax;
    std::transform(res.begin(), res.end(), std::back_inserter(y_softmax),
                [](U res) -> float { return (float) res; });

    return y_softmax;
  }

  std::vector<double> _softmax_double(std::vector<double> y_raw) const{
    /* Cast to U,T from double in/out */
    std::vector<T> yt_raw;
    std::transform(y_raw.begin(), y_raw.end(), std::back_inserter(yt_raw),
                   [](double yi_raw) -> T { return (T) yi_raw; });
    std::vector<U> y_softmax = softmax(yt_raw);
    std::vector<double> yd_softmax;
    std::transform(y_softmax.begin(), y_softmax.end(), std::back_inserter(yd_softmax),
                [](U yi_softmax) -> double { return (double) yi_softmax; });

    //std::cout << "Softmax! in conifer.h" << std::endl;
    return yd_softmax;
  }

}; // class BDT

} // namespace conifer

#endif

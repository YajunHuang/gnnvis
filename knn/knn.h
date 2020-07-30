#ifndef KNN_H
#define KNN_H

#include <stdlib.h>
#include <float.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <vector>
#include <pthread.h>


#include "ANNOY/annoylib.h"
#include "ANNOY/kissrandom.h"

#include <gsl/gsl_rng.h>


typedef float real;

struct arg_struct
{
  void *ptr;
  int id;

  arg_struct(void *x, int y) :ptr(x), id(y) {}
};

struct annoy_arg_struct
{
  void *ptr;
  int id, num_trees;
  std::string index_save_path;

  annoy_arg_struct(void *x, int y, int n_trees, std::string index_path) 
  :ptr(x), id(y), 
  num_trees(n_trees), index_save_path(index_path) {}
};

// struct nnd_arg_struct
// {
//   void *ptr;
//   int id, num_threads, num_neighbors;

//   nnd_arg_struct(void *x, int y, int n_neighbors, int n_threads) 
//   :ptr(x), id(y), 
//   num_neighbors(n_neighbors), num_threads(n_threads) {}
// };

class KNN {
private:
    long long num_samples, num_dim, num_neighbors;
    int num_threads;
    real perplexity;
    // int num_neighbors, num_threads, num_trees;   // TO DO: move num_trees to local variable
    std::vector<int> *knn_vec, *temp_knn_vec;
    real *data;
    AnnoyIndex<int, real, Euclidean, Kiss64Random> *annoy_index;
    // construct index for distributed similarity computing
    long long num_edge, *head;
    std::vector<long long> next, reverse;

    static const gsl_rng_type * gsl_T;
	static gsl_rng * gsl_r;

    void clean_data();
    //   void clean_model();
    void normalize();
    real calcDist(long long x, long long y);
    static void *annoy_thread_caller(void *arg);
    void annoy_thread(int id, int num_trees, std::string index_save_path);
    static void *nnd_thread_caller(void *arg);
    void nnd_thread(int id);
    void knn_similarity(real perplexity);
    static void *similarity_thread_caller(void *args);
    void similarity_thread(int id);
    static void *search_reverse_thread_caller(void *args);
    void search_reverse_thread(int id);
    void test_accuracy();

public:
    std::vector<int> edge_from, edge_to;
    std::vector<real> edge_weight;
    KNN(long long n_neigh, int n_thread);
    KNN(real *data, long long num_samples, long long num_dim, long long n_neigh, int n_thread);
    KNN(real *data, std::vector<int> *knn_vec, AnnoyIndex<int, real, Euclidean, Kiss64Random> *annoy_index, long long n_neigh, int n_thread);
    KNN();
    ~KNN();
    void set_n_neigh(long long n_neigh);
    void set_n_thread(int n_thread);
    // void load_data(std::string file_path, long long num_samples, long long num_dim);
    void load_data(real *data, long long num_samples, long long num_dim);
    void save_knn(std::string file_path);
    void annoy_knn(int num_trees, std::string index_save_path);
    void nnd_knn(int num_iter);
    void construct_knn(int num_trees, int num_iter, real perplexity);
};


#endif
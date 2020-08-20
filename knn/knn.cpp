#include "knn.h"


KNN::KNN(long long n_neigh, int n_thread) : num_neighbors(n_neigh), num_threads(n_thread)
{
    data = NULL;
    knn_vec = NULL;
	annoy_index = NULL;
    head = NULL;
}

KNN::KNN(real *data, long long num_samples, long long num_dim, long long n_neigh, int n_thread) : 
    data(data), num_samples(num_samples), num_dim(num_dim), num_neighbors(n_neigh), num_threads(n_thread)
{
    knn_vec = NULL;
	annoy_index = NULL;
    head = NULL;
}

KNN::KNN()
{
    data = NULL;
    knn_vec = NULL;
	annoy_index = NULL;
    head = NULL;
}

KNN::~KNN()
{
    clean_data();
}

void KNN::set_n_neigh(long long n_neigh)
{
    std::cout << "The num_neighbors will be changed from " << num_neighbors << " to " << n_neigh << std::endl;
    this->num_neighbors = n_neigh;
    clean_data();
}

void KNN::set_n_thread(int n_thread)
{
    this->num_threads = n_thread;
    std::cout << "The num_threads will be changed from " << num_threads << " to " << n_thread << std::endl;
}

const gsl_rng_type *KNN::gsl_T = NULL;
gsl_rng *KNN::gsl_r = NULL;


/**
 * 
 */ 
void KNN::clean_data()
{
    if (data) {delete[] data; data = NULL;}
    if (knn_vec) {delete[] knn_vec; knn_vec = NULL;}
    if (annoy_index) {delete annoy_index; annoy_index = NULL;}
    if (head) {delete[] head; head = NULL;}
}

/**
 * 
 */
// void KNN::load_data(std::string file_path, long long num_samples, long long num_dim) 
// {
//     clean_data();
//     std::ifstream infile;
//     infile.open(file_path);
//     if (!infile) 
//     {
//         std::cerr << "\nfile not found!\n";
//         exit(1);
//     }
//     std::cout << "Reading data from " << file_path << std::endl;
//     this->num_samples = num_samples;
//     this->num_dim = num_dim;
//     data = new real[num_samples * num_dim];

//     for (long long i = 0; i < num_samples; i++) 
//     {
//         for (long long j = 0; j < num_dim; j++) 
//         {
//             infile >> data[i * num_dim + j];
//         }
//     }
//     std::cout << "Read data done, " << "the data shape is (" << num_samples << ", " << num_dim << ")" << std::endl;
// }

void KNN::load_data(real *data, long long num_samples, long long num_dim)
{
    clean_data();
    this->data = data;
    this->num_samples = num_samples;
    this->num_dim = num_dim;
    std::cout << "Read data done, " << "the data shape is (" << num_samples << ", " << num_dim << ")" << std::endl;
}

void KNN::save_knn(std::string file_path)
{
    std::ofstream outfile;
    outfile.open(file_path);
    for (long long p = 0; p < num_edge; ++p)
	{
		double tmp = edge_weight[p];
        outfile << edge_from[p] << " " << edge_to[p] << " " << tmp << std::endl;
	}
	outfile.flush();
    outfile.close();
}

/**
 * 
 */
void KNN::normalize()
{
  real *mean = new real[num_dim];
  for (long long i = 0; i < num_dim; i++) 
  {
    mean[i] = 0;
  } 
  for (long long i = 0, ll = 0; i < num_samples; i++, ll += num_dim) 
  {
    for (long long j = 0; j < num_dim; j++)
    {
      mean[j] += data[ll + j];
    }
  }
  for (long long i = 0; i < num_dim; i++) 
  {
    mean[i] /= num_samples;
  }
  real max_mean_diff = 0;
  for (long long i = 0, ll = 0; i < num_samples; i++, ll += num_dim) 
  {
    for (long long j = 0; j < num_dim; j++)
    {
      data[ll + j] -= mean[j];
      if (max_mean_diff < fabs(data[ll + j]))
      {
        max_mean_diff = fabs(data[ll + j]);
      }
    }
  }
  for (long long i = 0; i < num_samples * num_dim; i++)
  {
    data[i] /= max_mean_diff;
  } 
  delete[] mean;
}

/**
 * 
 */
real KNN::calcDist(long long x, long long y)
{
  real dist = 0;
  long long x_loc = x * num_dim, y_loc = y * num_dim;
  for (long long i = 0; i < num_dim; i++)
  {
    dist += (data[x_loc + i] - data[y_loc + i]) * (data[x_loc + i] - data[y_loc + i]);
  }
  return dist;
}

/**
 * 
 */
void KNN::annoy_knn(int num_trees, std::string index_save_path) 
{
  std::cout << "Run annoy_knn..." << std::endl;
  // build random tree index by annoy
  annoy_index = new AnnoyIndex<int, real, Euclidean, Kiss64Random>(num_dim);    // new AnnoyIndex<long long, real, Euclidean, Kiss64Random>(num_dim); ??
  for (long long i=0; i < num_samples; i++)
  {
    annoy_index->add_item(i, &data[i * num_dim]);
  }
  annoy_index->build(num_trees);
  annoy_index->save(index_save_path.c_str());
  // Query knn of items from random tree index(annoy_index)
  knn_vec = new std::vector<int>[num_samples];
  pthread_t *ptrs = new pthread_t[num_threads];
  for (int i=0; i<num_threads; i++)
  {
    annoy_arg_struct *args = new annoy_arg_struct(this, i, num_trees, index_save_path);
    pthread_create(&ptrs[i], NULL, KNN::annoy_thread_caller, args);
  }
  for (int i=0; i<num_threads; i++)
  {
    pthread_join(ptrs[i], NULL);
  }
  delete[] ptrs;
  delete annoy_index;
  annoy_index = NULL;
  std::cout << "Done." << std::endl;
}

/**
 * function pointer ??
 */
void *KNN::annoy_thread_caller(void *args)
{
  KNN *ptr = (KNN*)(((annoy_arg_struct*)args)->ptr);
  int id = ((annoy_arg_struct*)args)->id;
  int num_trees = ((annoy_arg_struct*)args)->num_trees;
  std::string index_save_path = ((annoy_arg_struct*)args)->index_save_path;
  ptr->annoy_thread(id, num_trees, index_save_path);
  pthread_exit(NULL);
}

/**
 * annoy knn query thread
 */
void KNN::annoy_thread(int id, int num_trees, std::string index_save_path)
{
  long long low_pos = id * num_samples / num_threads;
  long long high_pos = (id + 1) * num_samples / num_threads;
  AnnoyIndex<int, real, Euclidean, Kiss64Random> *cur_annoy_index = NULL;
  if (id > 0)
  {
    cur_annoy_index = new AnnoyIndex<int, real, Euclidean, Kiss64Random>(num_dim);
    cur_annoy_index->load(index_save_path.c_str());
  }
  else
  {
    cur_annoy_index = annoy_index;
  }
  for (long long i = low_pos; i < high_pos; i++) 
  {
    cur_annoy_index->get_nns_by_item(i, num_neighbors + 1, (num_neighbors + 1) * num_trees, &knn_vec[i], NULL);
    for (long long j = 0; j < knn_vec[i].size(); j++)
    {
      if (knn_vec[i][j] == i) 
      {
        knn_vec[i].erase(knn_vec[i].begin() + j);
        break;
      }
    }
  }
  if (id > 0) 
  {
    delete cur_annoy_index;
  }
}

/**
 * 
 */
void KNN::nnd_knn(int num_iter)
{
  for (int it = 0; it < num_iter; it++)
  {
    std::cout << "nnd_knn... " << it << std::endl;
    temp_knn_vec = knn_vec;
    knn_vec = new std::vector<int>[num_samples];
    pthread_t *ptrs = new pthread_t[num_threads];
    for (int i = 0; i < num_threads; ++i) pthread_create(&ptrs[i], NULL, KNN::nnd_thread_caller, new arg_struct(this, i));
    for (int i = 0; i < num_threads; ++i) pthread_join(ptrs[i], NULL);
    delete[] ptrs;
    // delete[] temp_knn_vec;
	// temp_knn_vec = NULL;
  }
}

/**
 * 
 */
void *KNN::nnd_thread_caller(void *args)
{
  KNN *ptr = (KNN*)(((arg_struct*)args)->ptr);
  int id = ((arg_struct*)args)->id;
  ptr->nnd_thread(id);
  pthread_exit(NULL);
}

/**
 * 
 */
void KNN::nnd_thread(int id)
{
    // std::cout << "nnd_thread " << id << ": " << num_neighbors << ", " << num_threads << std::endl;
    long long low_pos = id * num_samples / num_threads;
    long long high_pos = (id + 1) * num_samples / num_threads;
    long long *check = new long long[num_samples];
    std::priority_queue< pair<real, int> > heap;
    long long x, y, i, j, l1, l2;
	for (x = 0; x < num_samples; ++x) 
    {
        check[x] = -1; 
    }
    for (x = low_pos; x < high_pos; x++)
	{
		check[x] = x;
        std::vector<int> &v1 = temp_knn_vec[x];
		l1 = v1.size();
        for (i = 0; i < l1; i++)
		{
            y = v1[i];
            check[y] = x;
            heap.push(std::make_pair(calcDist(x, y), y));
            if (heap.size() == num_neighbors + 1)
            {
                heap.pop();
            }
        }
        for (i = 0; i < l1; i++)
        {
            std::vector<int> &v2 = temp_knn_vec[v1[i]];
			l2 = v2.size();
            for (j = 0; j < l2; ++j) 
            {
                if (check[y = v2[j]] != x)
                {
                    check[y] = x;
                    heap.push(std::make_pair(calcDist(x, y), (int)y));
                    if (heap.size() == num_neighbors + 1) heap.pop();
                }
            }
        }
        while (!heap.empty())
		{
			knn_vec[x].push_back(heap.top().second);
			heap.pop();
		}
    }
}

void KNN::knn_similarity(real perplexity)
{
    this->perplexity = perplexity;
    num_edge = 0;
    head = new long long[num_samples];
    long long i, x, y, p, q;
    real sum_weight = FLT_MIN;
    for (i = 0; i < num_samples; i++)
    {
        head[i] = -1;
    }
    for (x = 0; x < num_samples; x++)
    {
        for (i = 0; i < knn_vec[x].size(); i++)
        {
            edge_from.push_back((int)x);
            edge_to.push_back((int)(y = knn_vec[x][i]));
            edge_weight.push_back(calcDist(x, y));
            next.push_back(head[x]);
            reverse.push_back(-1);
            head[x] = num_edge++;
        }
    }
    delete[] knn_vec, temp_knn_vec;
    knn_vec = temp_knn_vec = NULL;
    pthread_t *pt = new pthread_t[num_threads];
	for (int j = 0; j < num_threads; ++j) pthread_create(&pt[j], NULL, KNN::similarity_thread_caller, new arg_struct(this, j));
	for (int j = 0; j < num_threads; ++j) pthread_join(pt[j], NULL);
	delete[] pt;

    std::cout << num_edge << " edges before reverse." << std::endl;
	pt = new pthread_t[num_threads];
	for (int j = 0; j < num_threads; ++j) pthread_create(&pt[j], NULL, KNN::search_reverse_thread_caller, new arg_struct(this, j));
	for (int j = 0; j < num_threads; ++j) pthread_join(pt[j], NULL);
	delete[] pt;

    for (x = 0; x < num_samples; ++x)
	{
		for (p = head[x]; p >= 0; p = next[p])
		{
			y = edge_to[p];
			q = reverse[p];
			if (q == -1)
			{
				edge_from.push_back((int)y);
				edge_to.push_back((int)x);
				edge_weight.push_back(0);
				next.push_back(head[y]);
				reverse.push_back(p);
				q = reverse[p] = head[y] = num_edge++;
			}
			if (x > y) {
                sum_weight += edge_weight[p] + edge_weight[q];
				edge_weight[p] = edge_weight[q] = (edge_weight[p] + edge_weight[q]) / 2;               
			}
		}
	}

  std::cout << num_edge << " edges after reverse." << std::endl;
  std::cout << "Done, sum_weight=" << sum_weight << std::endl;
}

void *KNN::similarity_thread_caller(void *args)
{
  KNN *ptr = (KNN*)(((arg_struct*)args)->ptr);
  int id = ((arg_struct*)args)->id;
  ptr->similarity_thread(id);
  pthread_exit(NULL);
}

void KNN::similarity_thread(int id)
{
    long long low_pos = id * (num_samples / num_threads);
    long long high_pos = (id + 1) * (num_samples / num_threads);

    long long x, iter, p;
    real beta, lo_beta, hi_beta, sum_weight, H, tmp;
    for (x = low_pos; x < high_pos; x++)
    {
        beta = 1;
        lo_beta = hi_beta = -1;
        for (iter = 0; iter < 200; iter++)
        {
            H = 0;
            sum_weight = FLT_MIN;
            for (p = head[x]; p >= 0; p = next[p])
            {
                sum_weight += tmp = exp(-beta * edge_weight[p]);
                H += beta * (edge_weight[p] * tmp);
            }
            H = (H / sum_weight) + log(sum_weight);
            if (fabs(H - log(perplexity)) < 1e-5) break;
            if (H > log(perplexity))
            {
                lo_beta = beta;
                if (hi_beta < 0) beta *= 2; else beta = (beta + hi_beta) / 2;
            }
            else {
                hi_beta = beta;
                if (lo_beta < 0) beta /= 2; else beta = (lo_beta + beta) / 2;
            }
            if(beta > FLT_MAX) beta = FLT_MAX;
        }
        
        for (p = head[x], sum_weight = FLT_MIN; p >= 0; p = next[p])
        {
            sum_weight += edge_weight[p] = exp(-beta * edge_weight[p]);
        }
        for (p = head[x]; p >= 0; p = next[p])
        {
            edge_weight[p] /= sum_weight;
        }
    }
}

void *KNN::search_reverse_thread_caller(void *arg)
{
	KNN *ptr = (KNN*)(((arg_struct*)arg)->ptr);
	ptr->search_reverse_thread(((arg_struct*)arg)->id);
	pthread_exit(NULL);
}

void KNN::search_reverse_thread(int id)
{
	long long lo = id * num_samples / num_threads;
	long long hi = (id + 1) * num_samples / num_threads;
	long long x, y, p, q;
	for (x = lo; x < hi; ++x)
	{
		for (p = head[x]; p >= 0; p = next[p])
		{
			y = edge_to[p];
			for (q = head[y]; q >= 0; q = next[q])
			{
				if (edge_to[q] == x) break;
			}
			reverse[p] = q;
		}
	}
}


/**
 * 
 */
void KNN::construct_knn(int num_trees, int num_iter, real perplexity) 
{
  /* gsl set random generator and seed */
	gsl_rng_env_setup();		
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

  // normalize the input data as as (x - mean(x)) / max(abs(x - mean(x)))
  normalize();
  // construct knn graph using annoy
  annoy_knn(num_trees, "annoy_index");
  test_accuracy();
  // improve knn graph using nndecent
  nnd_knn(num_iter);
  test_accuracy();
  // compute similarity
  knn_similarity(perplexity);
}

void KNN::test_accuracy()
{
	long long test_case = 100;
	std::priority_queue< pair<real, int> > *heap = new std::priority_queue< pair<real, int> >;
	long long hit_case = 0, i, j, x, y;
	for (i = 0; i < test_case; ++i)
	{
		x = floor(gsl_rng_uniform(gsl_r) * (num_samples - 0.1));
		for (y = 0; y < num_samples; ++y) if (x != y)
		{
			heap->push(std::make_pair(calcDist(x, y), y));
			if (heap->size() == num_neighbors + 1) heap->pop();
		}
		while (!heap->empty())
		{
			y = heap->top().second;
			heap->pop();
			for (j = 0; j < knn_vec[x].size(); ++j) if (knn_vec[x][j] == y)
				++hit_case;
		}
	}
    delete heap;
	printf("Test knn accuracy : %.2f%%\n", hit_case * 100.0 / (test_case * num_neighbors));
}


/**
 * 
 */

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <set>
#include <initializer_list>
#include <iostream>
#include <cassert>

#define BP_VERBOSE

struct N {
    
    int id; //current node id
    int label; //selected label after optimization
    int label_orig;
    int count_labels = 0; //current node labels
    int count_neighbours = 0;
    float * msg_label = nullptr; //holds incoming msg; indexing: label * count_neighbours_max (provided by Fmsg_index lambda) + node_id -> message
    float * msg_label_swap = nullptr; //temporary
    int * neighbours = nullptr; //neighbour node ids
    N* node_map = nullptr; //node_id -> N* (globally shared)

    __host__
    static N* CudaBulkAllocNodes(int num_nodes){
        N* ns;
        cudaMalloc(&ns, num_nodes*sizeof(N));
        return ns;
    }
    __host__
    static N* BulkAllocNodes(int num_nodes){
        N* ns = new N[num_nodes];
        return ns;
    }

    __host__
    void forget_mem(){
        msg_label = nullptr;
        msg_label_swap = nullptr;
        neighbours = nullptr;
        node_map = nullptr;
    }
    
    __host__
    void CudaCopy(N* node_host,
                  N* nodes_gpu_start,
                  float* device_label_node,
                  float* device_label_node_swap,
                  int* device_neighbours,
                  int max_count_labels,
                  int max_count_neighbours){
        ///copy node data to temporary buffer at node_host, while allocating some internal data on gpu 
        
        node_host->id = id;
        node_host->label = label;
        node_host->label_orig = label_orig;
        node_host->count_labels = count_labels;
        node_host->count_neighbours = count_neighbours;

        cudaMemcpy(device_label_node,
                   msg_label,
                   count_labels * count_neighbours * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(device_label_node_swap,
                   msg_label_swap,
                   count_labels * count_neighbours * sizeof(float),
                   cudaMemcpyHostToDevice);
        
        node_host->msg_label = device_label_node;
        node_host->msg_label_swap = device_label_node_swap;

        cudaMemcpy(device_neighbours,
                   neighbours,
                   count_neighbours*sizeof(int),
                   cudaMemcpyHostToDevice);

        node_host->neighbours = device_neighbours;
        
        node_host->node_map = nodes_gpu_start;
    }

    __host__
    N(int identity) : id(identity) {}
    
    __host__
    N(){}
    
    __host__
    ~N(){
        if(msg_label) delete [] msg_label;

        if(msg_label_swap) delete [] msg_label_swap;

        if(neighbours) delete [] neighbours;
    }

    __host__ __device__
    int get_id() const {
        return id;
    }

    __host__ __device__
    int get_count_labels() const {
        return count_labels;
    }

    __host__ __device__
    int get_count_neighbours() const {
        return count_neighbours;
    }
    
    __host__ __device__
    void set_node_map(N* n_map){
        assert(n_map);
        node_map = n_map;
    }

    __host__ __device__
    int get_label() const {
        return label;
    }

    __host__ __device__
    void set_labels(int const count){
        assert(count>0);
        count_labels = count;
    }

    __host__
    void set_neighbours(std::set<int> const & neigh_ids){
        if(neigh_ids.count(id)){
            count_neighbours = neigh_ids.size()-1;
        }else{
            count_neighbours = neigh_ids.size();
        }
        if(neighbours) delete [] neighbours;

        neighbours = new int[count_neighbours];
        int j = 0;
        for(auto i: neigh_ids){
            if(i!=id){
                neighbours[j++] = i;
            }
        }
    }

    __host__
    void confirm_neighbours(int max_count_labels, int max_count_neighbours){
        
        if(msg_label) delete [] msg_label;
        if(msg_label_swap) delete [] msg_label_swap;

        msg_label = new float[count_labels * count_neighbours];
        msg_label_swap = new float[count_labels * count_neighbours];

        std::fill_n(msg_label, count_labels * count_neighbours, 0.);
        std::fill_n(msg_label_swap, count_labels * count_neighbours, 0.);
    }

    template<class Fnode, class Fedge>
    __host__ __device__
    void update_belief(Fnode f_node,
                       Fedge f_edge){
        ///update label
        float belief_best = std::numeric_limits<float>::max();
                    
        for(int l=0; l<count_labels; ++l){
            float accum = 0.;
            for(int j = 0; j < count_neighbours; ++j){
                accum += msg_label[l * count_neighbours + j];
            }
            float val = accum + f_node(this, l);
            if(val < belief_best){
                belief_best = val;
                label = l;
            }
        }
    }

    template<class Fnode, class Fedge>
    __host__ __device__
    void distribute_msg(Fnode f_node,
                        Fedge f_edge){
        ///distribute message using one with minimum value
        for(int l_cur=0; l_cur<count_labels; ++l_cur){ //per destination labels
            for(int i=0; i<count_neighbours; ++i){ //per source node
                int other_id = neighbours[i];
                N * node_other = &node_map[other_id];
                float val_best = std::numeric_limits<float>::max();
                for(int l_other=0; l_other<node_other->count_labels; ++l_other){
                    float potential = f_node(this, l_cur) + f_edge(this, l_cur, node_other, l_other);
                    float msg_neighbour = 0.;
                    for(int j = 0; j < node_other->count_neighbours; ++j){
                        int index = l_other * node_other->count_neighbours + j;
                        msg_neighbour += node_other->neighbours[j] == id ? 0 : node_other->msg_label[index];
                    }
                    val_best = min(val_best, potential + msg_neighbour);
                }
                msg_label_swap[l_cur * count_neighbours + i] = val_best; //incoming msg from other to current node's l_cur
            }
        }
    }

    __host__ __device__
    void update_msg(){
        auto temp = msg_label;
        msg_label = msg_label_swap;
        msg_label_swap = temp;
    }
};

template<class Fnode, class Fedge>
__global__ static void cycle_aux(int iter,
                                 int num_threads,
                                 N* nodes,
                                 int num_nodes,
                                 Fnode f_node,
                                 Fedge f_edge){

    int t = threadIdx.x;
    int chunk_nominal = num_nodes / num_threads;
    int remain = num_nodes % num_threads;
    int chunk_adjust = t < remain ? chunk_nominal+1 : chunk_nominal;
    int start = chunk_adjust * min(t, remain) + chunk_nominal * max(t-remain, 0);

#ifdef BP_VERBOSE
    if(t==0) printf("node count: %d\n", num_nodes);
#endif
    
    for(int i=0; i<iter; ++i){

#ifdef BP_VERBOSE
        if(t==0) printf("iter: %d\n", i);
#endif
        for(int j=start; j<start+chunk_adjust; ++j){
            nodes[j].distribute_msg(f_node, f_edge);
        }
        
        __syncthreads();
        
        for(int j=start; j<start+chunk_adjust; ++j){
            nodes[j].update_msg();
        }

        __syncthreads();

        if(i+1==iter){
            for(int j=start; j<start+chunk_adjust; ++j){
                nodes[j].update_belief(f_node, f_edge);
            }
        }
    }
}

template<int THREADS, class Fnode, class Fedge>
__host__
static void cycle(int const iter,
                  N* nodes,
                  int num_nodes,
                  Fnode f_node,
                  Fedge f_edge){

    cycle_aux<<<1, THREADS>>>(iter,
                              THREADS,
                              nodes,
                              num_nodes,
                              f_node,
                              f_edge);
}

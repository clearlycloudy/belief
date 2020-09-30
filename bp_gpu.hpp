#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <set>
#include <initializer_list>
#include <iostream>
#include <cassert>

struct N {
    
    int id; //current node id
    int label; //selected label after optimization
    int count_labels = 0; //current node labels
    int count_neighbours = 0;
    float ** msg_label = nullptr; //holds incoming msg, label -> node_id -> message
    float ** msg_label_swap = nullptr; //temporary
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
    void CudaCopy(N* node_host, N* node_gpu, N* nodes_gpu_start, std::vector<float*> & recycle){
        ///copy node data to gpu side, located at node_gpu
        //node_host serves as temporary storage on host side
        //currently super non-optimal, todo: builk copying and flatten data structure
        
        node_host->id = id;
        node_host->label = label;
        node_host->count_labels = count_labels;
        node_host->count_neighbours = count_neighbours;
        
        float** temp_msg_label = new float*[count_labels]; //host side
        float** temp_msg_label_swap = new float*[count_labels];
        for(int i=0; i<count_labels; ++i){
            float* buf;
            float* buf_swap;
            cudaMalloc(&buf, count_neighbours*sizeof(float)); //device side
            cudaMalloc(&buf_swap, count_neighbours*sizeof(float));

            cudaMemcpy(buf, msg_label[i], count_neighbours*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(buf_swap, msg_label_swap[i], count_neighbours*sizeof(float), cudaMemcpyHostToDevice);
            
            temp_msg_label[i] = buf; //store device memory location
            temp_msg_label_swap[i] = buf_swap;

            recycle.push_back(buf);
            recycle.push_back(buf_swap);
        }

        float** device_msg_label; //device side
        float** device_msg_label_swap;
        cudaMalloc(&device_msg_label, count_labels*sizeof(float*));
        cudaMalloc(&device_msg_label_swap, count_labels*sizeof(float*));

        cudaMemcpy(device_msg_label, temp_msg_label, count_labels*sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(device_msg_label_swap, temp_msg_label_swap, count_labels*sizeof(float*), cudaMemcpyHostToDevice);
        
        node_host->msg_label = device_msg_label;
        node_host->msg_label_swap = device_msg_label_swap;

        int* device_neighbours;
        cudaMalloc(&device_neighbours, count_neighbours*sizeof(int));
        cudaMemcpy(device_neighbours, neighbours, count_neighbours*sizeof(int), cudaMemcpyHostToDevice);
        node_host->neighbours = device_neighbours;
        node_host->node_map = nodes_gpu_start;
        
        cudaMemcpy(node_gpu, node_host, sizeof(N), cudaMemcpyHostToDevice);

        delete [] temp_msg_label;
        delete [] temp_msg_label_swap;
    }
    __device__
    void CudaDelete(){
        for(int i=0; i<count_labels; ++i){
            cudaFree(msg_label[i]);
            cudaFree(msg_label_swap[i]);
        }
        cudaFree(msg_label);
        cudaFree(msg_label_swap);
        cudaFree(neighbours);
    }
    __host__
    N(int identity) : id(identity) {}
    
    __host__
    N(){}
    
    __host__
    ~N(){
        if(msg_label){
            for(int i=0; i<count_labels; ++i){
                if(msg_label[i]) delete [] msg_label[i];
            }
            delete [] msg_label;
        }

        if(msg_label_swap){
            for(int i=0; i<count_labels; ++i){
                if(msg_label_swap[i]) delete [] msg_label_swap[i];
            }
            delete [] msg_label_swap;
        }

        if(neighbours) delete [] neighbours;
    }

    __host__ __device__
    int get_id() const {
        return id;
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

        if(msg_label){
            for(int i=0; i<count_labels; ++i){
                if(msg_label[i]) delete [] msg_label[i];
            }
            delete [] msg_label;
        }

        if(msg_label_swap){
            for(int i=0; i<count_labels; ++i){
                if(msg_label_swap[i]) delete [] msg_label_swap[i];
            }
            delete [] msg_label_swap;
        }
    
        msg_label = new float* [count];
        msg_label_swap = new float* [count];
        count_labels = count;
    
        for(int i=0; i<count_labels; ++i){
            msg_label[i] = nullptr;
            msg_label_swap[i] = nullptr;
        }
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
    void confirm_neighbours(){
        for(int i=0;i<count_labels;++i){
            if(msg_label[i]) delete [] msg_label[i];
            if(msg_label_swap[i]) delete [] msg_label_swap[i];
    
            msg_label[i] = new float [count_neighbours];
            msg_label_swap[i] = new float [count_neighbours];

            std::fill_n(msg_label[i], count_neighbours, 0.);
            std::fill_n(msg_label_swap[i], count_neighbours, 0.);
        }
    }

    template<class Fnode, class Fedge>
    __host__ __device__
    void update_belief(Fnode f_node,
                       Fedge f_edge){
        ///update label
        // printf("count_labels: %d, count_neighbours: %d\n", count_labels, count_neighbours);
        float belief_best = std::numeric_limits<float>::max();
        for(int l=0; l<count_labels; ++l){
            float accum = 0.;
            for(int j = 0; j < count_neighbours; ++j){
                accum += msg_label[l][j];
            }
            float val = accum + f_node(this, l);
            if(val < belief_best){
                belief_best = val;
                label = l;
                // printf("better val found: %f\n", val);
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
                        msg_neighbour += node_other->neighbours[j] == id ? 0 : node_other->msg_label[l_other][j];
                    }
                    val_best = min(val_best, potential + msg_neighbour);
                }
                msg_label_swap[l_cur][i] = val_best; //incoming msg from other to current node's l_cur
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

constexpr int THREADS = 1024;

        
template<class Fnode, class Fedge>
__global__ static void cycle(int const iter,
                             N* nodes,
                             int num_nodes,
                             Fnode f_node,
                             Fedge f_edge){
    
    int t = threadIdx.x;
    int chunk_nominal = num_nodes / 1024;
    int remain = num_nodes % 1024;
    int chunk_adjust = t < remain ? chunk_nominal+1 : chunk_nominal;
    int start = chunk_adjust * min(t, remain) + chunk_nominal * max(t-remain, 0);

    // printf("start: %d\n", start);    
    for(int i=0; i<iter; ++i){

        if(t==0) printf("iter: %d\n", i);
        
        for(int j=start; j<start+chunk_adjust; ++j){
            assert(j>=0);
            assert(j<num_nodes);
            nodes[j].update_belief(f_node, f_edge);
        }
        
        __syncthreads();

        for(int j=start; j<start+chunk_adjust; ++j){
            nodes[j].distribute_msg(f_node, f_edge);
        }
        
        __syncthreads();
        
        for(int j=start; j<start+chunk_adjust; ++j){
            nodes[j].update_msg();
        }

        __syncthreads();
    }
}

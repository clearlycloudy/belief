#include <cmath>
#include <random>
#include <unordered_map>
#include <thread>
#include <set>
#include <cassert>
#include <vector>
#include <utility>
#include <iostream>

#include <curand_kernel.h>

#include "../bp_gpu.hpp"
#include "lodepng.h"

using namespace std;

constexpr int KERNEL_SIZE_LABEL = 1;
constexpr int KERNEL_SIZE = 1;
constexpr int ITERATIONS = 4;
constexpr float FRAC_NEIGH = 0.3;
constexpr float FRAC_LABEL = 0.3;
constexpr int NUM_THREADS = 256;

default_random_engine gen;

__host__ __device__
float colour_diff(unsigned char const * const c1, unsigned char const * const c2){
    ///approximate perceptual colour diff
    
    unsigned char r1 = c1[0];
    unsigned char g1 = c1[1];
    unsigned char b1 = c1[2];

    unsigned char r2 = c2[0];
    unsigned char g2 = c2[1];
    unsigned char b2 = c2[2];

    float rr = (r1+r2)/2.;
    
    return sqrt((2.+rr/256.)*(r1-r2)*(r1-r2) +
                4*(g1-g2)*(g1-g2) +
                (2.+(255.-rr)/256.)*(b1-b2)*(b1-b2));
}

__host__
pair<N*, int> init_tree(vector<unsigned char> const & img,
                        int const h,
                        int const w,
                        unordered_map<N*,pair<int,int>> & coordinate_map,
                        vector<vector<pair<int,int>>> & label_map){

    int nodes_count = img.size()/4;
    
    vector<N*> ret(nodes_count);

    N* ns = new N[nodes_count];
    
    int id = 0;
    for(auto &i: ret){
        i = &ns[id];
        i->id = id;
        id++;
    }
    
    for(int i=0;i<h;++i){
        for(int j=0;j<w;++j){
            int index = i*w+j;
            assert(index<ret.size());
            N * n = ret[index];
            coordinate_map[n] = {i, j};
            int count_labels = 0;
            while(count_labels<=0){
                int id_label = 0;
                if(n->get_id()<label_map.size())
                    label_map[n->get_id()].clear();
                for(int k=-KERNEL_SIZE_LABEL;k<=KERNEL_SIZE_LABEL;++k){
                    for(int l=-KERNEL_SIZE_LABEL;l<=KERNEL_SIZE_LABEL;++l){
                        int ii = i+k;
                        int jj = j+l;
                        uniform_real_distribution<float> distr(0.,1.);
                        if(ii>=0 && ii<h &&
                           jj>=0 && jj<w &&
                           distr(gen) < FRAC_LABEL){
                            if(n->get_id() >= label_map.size())
                                label_map.resize(n->get_id()+1);
                            if(label_map[n->get_id()].size() <= id_label){
                                label_map[n->get_id()].resize(id_label+1);
                            }
                            label_map[n->get_id()][id_label] = {ii,jj};
                            id_label++;
                        }
                    }
                }
                count_labels = id_label;
            }
            n->set_labels(count_labels);
        }
    }
    unordered_map<N*, set<N*>> neigh_map;
    for(int i=0;i<h;++i){
        for(int j=0;j<w;++j){
            int index = i*w+j;
            assert(index<ret.size());
            N * n = ret[index];
            int count_neighbour = 0;
            while(count_neighbour<=0){
                for(int k=-KERNEL_SIZE;k<=KERNEL_SIZE;++k){
                    for(int l=-KERNEL_SIZE;l<=KERNEL_SIZE;++l){
                        int ii = i+k;
                        int jj = j+l;
                        uniform_real_distribution<float> distr(0.,1.);
                        if(ii>=0 && ii<h &&
                           jj>=0 && jj<w &&
                           (ii!=i || jj!=j)&&
                           distr(gen) < FRAC_NEIGH){
                            int index2 = ii*w+jj;
                            neigh_map[n].insert(ret[index2]);
                            count_neighbour++;
                        }
                    }
                }
            }
        }
    }

    for(auto [n, neigh]: neigh_map){
        set<int> neigh_ids;
        for(auto i: neigh){
            neigh_ids.insert(i->get_id());
        }
        n->set_neighbours(neigh_ids);
    }

    for(auto i: ret){
        i->confirm_neighbours();
    }

    for(auto i: ret){
        i->set_node_map(ret[0]);
    }

    return {ns, nodes_count};
}

__host__
vector<unsigned char> bp_run(vector<unsigned char> const & img,
                             int const h,
                             int const w){
    
    unordered_map<N*,pair<int,int>> coordinate_map;
    vector<vector<pair<int,int>>> label_map; //node id -> label -> coordinates

    N* ns;
    int node_count;
    std::tie(ns, node_count) = init_tree(img, h, w, coordinate_map, label_map);
    assert(node_count>0);

    
    pair<int,int> ** label_map_array = new pair<int,int>*[label_map.size()];
    
    for(int i=0; i<label_map.size(); ++i){
        int num_labels = label_map[i].size();
        //allocate on device
        pair<int,int>* device_label_map_array_inner;
        cudaMalloc(&device_label_map_array_inner, num_labels*sizeof(pair<int,int>));

        cudaMemcpy(device_label_map_array_inner,
                   &label_map[i][0],
                   num_labels*sizeof(pair<int,int>), cudaMemcpyHostToDevice);

        label_map_array[i] = device_label_map_array_inner;
    }

    pair<int,int>** device_label_map_array;
    cudaMalloc(&device_label_map_array, label_map.size()*sizeof(pair<int,int>*));

    cudaMemcpy(device_label_map_array,
               label_map_array,
               label_map.size()*sizeof(pair<int,int>*), cudaMemcpyHostToDevice);

    unsigned char * img_data;
    cudaMalloc(&img_data, img.size() * sizeof(unsigned char));
    cudaMemcpy(img_data,
               &img[0],
               img.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
        
    auto potential_node = [=] __device__ (N* const n,
                                          int const l) -> float {
        return 1.;
    };

    auto potential_edge = [=] __device__ (N* const n0,
                                          int const l0,
                                          N* const n1,
                                          int const l1) -> float {
        auto [y0, x0] = device_label_map_array[n0->get_id()][l0];
        auto [y1, x1] = device_label_map_array[n1->get_id()][l1];
        return colour_diff(&img_data[y0*w*4+x0*4], &img_data[y1*w*4+x1*4]);
    };

    N* ns_gpu = N::CudaBulkAllocNodes(node_count); //device mem alloc

    printf("last error: %s\n", cudaGetErrorString(cudaGetLastError()));
    
    N* ns_host = N::BulkAllocNodes(node_count); //host mem alloc

    printf("last error: %s\n", cudaGetErrorString(cudaGetLastError()));

    std::vector<float*> recycle;
    for(int i=0; i<node_count; ++i){
        ns[i].CudaCopy(&ns_host[i], &ns_gpu[i], ns_gpu, recycle);
    }

    printf("last error: %s\n", cudaGetErrorString(cudaGetLastError()));
    
    cycle<<<1, NUM_THREADS>>>(ITERATIONS, ns_gpu, node_count, potential_node, potential_edge);
    
    //copy labels back
    N* ns_copy_back = new N[node_count];
    cudaMemcpy(ns_copy_back,
               ns_gpu,
               sizeof(N) * node_count,
               cudaMemcpyDeviceToHost);
    for(int i=0; i<node_count; ++i){
        ns[i].label = ns_copy_back[i].label;
        ns_copy_back[i].msg_label = nullptr;
        ns_copy_back[i].msg_label_swap = nullptr;
        ns_copy_back[i].neighbours = nullptr;
        ns_copy_back[i].node_map = nullptr;
    }
    delete [] ns_copy_back;
    
    //free stuff
    for(int i=0; i<label_map.size(); ++i){
        cudaFree(label_map_array[i]);
    }
    cudaFree(device_label_map_array);
    delete [] label_map_array;
    
    //free gpu memory
    
    for(int i=0; i<node_count; ++i){
        N& n = ns_host[i];
        cudaFree(n.msg_label);
        cudaFree(n.msg_label_swap);
        cudaFree(n.neighbours);

        n.msg_label = nullptr;
        n.msg_label_swap = nullptr;
        n.neighbours = nullptr;
        n.node_map = nullptr;
    }
    delete [] ns_host;
    
    for(auto i: recycle){
        cudaFree(i);
    }
    cudaFree(ns_gpu);

    cudaFree(img_data);
    
    vector<unsigned char> out(img.size(),0);

    auto f_save = [&](int start, int block_size){
        for(int i=start; i<start+block_size; ++i){
            N * n = &ns[i];
            auto [y,x] = coordinate_map[n];
            auto [yy,xx] = label_map[n->get_id()][n->get_label()];
            out[y*w*4+x*4] = img[yy*w*4+xx*4];
            out[y*w*4+x*4+1] = img[yy*w*4+xx*4+1];
            out[y*w*4+x*4+2] = img[yy*w*4+xx*4+2];
            out[y*w*4+x*4+3] = 255;
        }
    };

    int processors = thread::hardware_concurrency();
    processors = max(processors,1);
    
    vector<thread> t(processors-1);
    int chunk = node_count / processors;
    int remain = node_count % processors;

    for(int i=0;i<t.size();++i){
        t[i] = thread(f_save, i*chunk, chunk);
    }
    f_save(t.size()*chunk, chunk+remain);

    for(auto &i: t){
        i.join();
    }

    cudaFree(ns_gpu);
    
    delete [] ns;

    return out;
}

int main(){
    cout << "enter input file path" << endl;
    string file;
    cin >> file;
    vector<unsigned char> img;
    unsigned int h, w;
    lodepng::decode(img, w, h, file);
    auto output = bp_run(img, h, w);
    string file_out = "out_" + file;
    lodepng::encode(file_out, output, w, h);
    return 0;
}



    

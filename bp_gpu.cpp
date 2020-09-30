#include "bp_gpu.hpp"

using namespace std;

// __host__ N::N(int identity) : id(identity) {}

// __host__ N::~N(){
//     if(msg_label){
// 	for(int i=0; i<count_labels; ++i){
// 	    if(msg_label[i]) delete [] msg_label[i];
// 	}
// 	delete [] msg_label;
//     }

//     if(msg_label_swap){
// 	for(int i=0; i<count_labels; ++i){
// 	    if(msg_label_swap[i]) delete [] msg_label_swap[i];
// 	}
// 	delete [] msg_label_swap;
//     }

//     if(neighbours) delete [] neighbours;
// }

// __host__ __device__ int N::get_id() const {
//     return id;
// }

// __host__ void N::set_node_map(N** n_map){
//     assert(n_map);
//     node_map = n_map;
// }

// __host__ __device__ int N::get_label() const {
//     return label;
// }

// __host__ void N::set_labels(int count){
//     assert(count>0);

//     if(msg_label){
// 	for(int i=0; i<count_labels; ++i){
// 	    if(msg_label[i]) delete [] msg_label[i];
// 	}
// 	delete [] msg_label;
//     }

//     if(msg_label_swap){
// 	for(int i=0; i<count_labels; ++i){
// 	    if(msg_label_swap[i]) delete [] msg_label_swap[i];
// 	}
// 	delete [] msg_label_swap;
//     }
	
//     msg_label = new float* [count];
//     msg_label_swap = new float* [count];
//     count_labels = count;
    
//     for(int i=0; i<count_labels; ++i){
// 	msg_label[i] = nullptr;
// 	msg_label_swap[i] = nullptr;
//     }
// }

// __host__  void N::set_neighbours(set<int> const & neigh_ids){
//     if(neigh_ids.count(id)){
// 	count_neighbours = neigh_ids.size()-1;
//     }else{
// 	count_neighbours = neigh_ids.size();
//     }
//     if(neighbours) delete [] neighbours;

//     neighbours = new int[count_neighbours];
//     int j = 0;
//     for(auto i: neigh_ids){
// 	if(i!=id){
// 	    neighbours[j++] = i;
// 	}
//     }
// }

// __host__ void N::confirm_neighbours(){
//     for(int i=0;i<count_labels;++i){
// 	if(msg_label[i]) delete [] msg_label[i];
// 	if(msg_label_swap[i]) delete [] msg_label_swap[i];
	
// 	msg_label[i] = new float [count_neighbours];
// 	msg_label_swap[i] = new float [count_neighbours];

//         fill_n(msg_label[i], count_neighbours, 0.);
// 	fill_n(msg_label_swap[i], count_neighbours, 0.);
//     }
// }

// __host__ __device__ void N::update_msg(){
//     auto temp = msg_label;
//     msg_label = msg_label_swap;
//     msg_label_swap = temp;
// }

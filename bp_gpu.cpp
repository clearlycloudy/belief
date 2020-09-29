#include "bp.hpp"

#include <iostream>
#include <cassert>
#include <thread>
#include <mutex>
#include <condition_variable>

#define VERBOSE

using namespace std;

N::N(int identity) : id(identity) {}

N::~N(){
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

int N::get_id() const {
    return id;
}
void N::set_node_map(N** n_map){
    assert(n_map);
    node_map = n_map;
}

int N::get_label() const {
    return label;
}

void N::set_labels(int count){
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

void N::set_neighbours(set<int> const & neigh_ids){
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

void N::confirm_neighbours(){
    for(int i=0;i<count_labels;++i){
	if(msg_label[i]) delete [] msg_label[i];
	if(msg_label_swap[i]) delete [] msg_label_swap[i];
	
	msg_label[i] = new float [count_neighbours];
	msg_label_swap[i] = new float [count_neighbours];

        fill_n(msg_label[i], count_neighbours, 0.);
	fill_n(msg_label_swap[i], count_neighbours, 0.);
    }
}

void N::update_belief(function<float(N* const, int const)> f_node,
                      function<float(N* const, int const, N* const, int const)> f_edge){
    ///update label
    
    float belief_best = numeric_limits<int>::max();
    for(int l=0; l<count_labels; ++l){
	float accum = 0.;
	for(int j = 0; j < count_neighbours; ++j){
	    accum += msg_label[l][j];
	}
        float val = accum + f_node(this, l);
        if(val < belief_best){
            belief_best = val;
            label = l;
        }
    }
}
    
void N::distribute_msg(function<float(N* const, int const)> f_node,
                       function<float(N* const, int const, N* const, int const)> f_edge){    
    ///distribute message using one with minimum value
    
    for(int l_cur=0; l_cur<count_labels; ++l_cur){ //per destination labels
        for(int i=0; i<count_neighbours; ++i){ //per source node
	    int other_id = neighbours[i];
	    N * node_other = node_map[other_id];
            float val_best = numeric_limits<int>::max();
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

void N::update_msg(){
    swap(msg_label, msg_label_swap);
}

void N::cycle(int const iter,
              vector<N*> & ns,
              function<float(N* const, int const)> f_node,
              function<float(N* const, int const, N* const, int const)> f_edge){
    ///perform iter cycles of algorithm
    
    int processors = thread::hardware_concurrency()/2; 
    processors = max(processors,1);
    
    vector<thread> t(processors-1);
    int chunk = ns.size()/processors;

    uint32_t stage = 0;
    uint32_t sync_count = 0;
    mutex mut;
    condition_variable cond_var;
    
    auto f_work = [&](int const t_idx, int const start, int const chunk){
        
                      for(int _=0; _<iter; ++_){
#ifdef VERBOSE
                          if(t_idx==0) printf("iter: %d\n",_);
#endif
                      
                          {
                              unique_lock<mutex> lock(mut);
                              cond_var.wait( lock, [&](){return stage == 0;} );
                          }
                          for(int j= start; j < start+chunk; ++j){
                              auto i = ns[j];
                              i->update_belief(f_node, f_edge);
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              if(++sync_count == processors)
                                  stage = (stage+1)%3;
                              sync_count = sync_count % processors;
                              cond_var.notify_all();
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              cond_var.wait( lock, [&](){return stage == 1;} );
                          }
                          for(int j= start; j < start+chunk; ++j){
                              auto i = ns[j];
                              i->distribute_msg(f_node, f_edge);
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              if(++sync_count == processors)
                                  stage = (stage+1)%3;
                              sync_count = sync_count % processors;
                              cond_var.notify_all();
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              cond_var.wait( lock, [&](){return stage == 2;} );
                          }
                          for(int j= start; j < start+chunk; ++j){
                              auto i = ns[j];
                              i->update_msg();
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              if(++sync_count == processors)
                                  stage = (stage+1)%3;
                              sync_count = sync_count % processors;
                              cond_var.notify_all();
                          }
                      }
                  };

    int remain = ns.size() % processors;

    for(int i=0;i<t.size();++i){
        t[i] = thread(f_work, i, i*chunk, chunk);
    }
    
    f_work(t.size(), t.size()*chunk, chunk+remain);
    
    for(int i=0;i<t.size();++i){
        t[i].join();
    }
}

#include "bp.hpp"

#include <iostream>
#include <cassert>
#include <thread>
#include <mutex>
#include <condition_variable>

#define VERBOSE

using namespace std;

void N::debug_label(){
    for(int l=0; l<msg_label.size(); ++l){
        auto & m = msg_label[l];
        printf("label: %d, incoming messages: ", l);
        for(auto [n, v]: m){
            printf("msg: %f ", v);
        }
        printf("\n");
    }
}

int N::get_label() const {
    return label;
}

void N::set_labels(int count){
    assert(count>0);
    msg_label.resize(count);
    msg_label_swap.resize(count);
    count_labels = count;
    msg_neigh_accum.resize(count, 0.);
}

void N::set_neighbour(N* const n){
    if(n!=this && n){
        neighbour.push_back(n);
    }
}

void N::update_belief(function<float(N* const, int const)> f_node,
                      function<float(N* const, int const, N* const, int const)> f_edge){
    ///update label

    assert(msg_label.size()==count_labels);
    
    float belief_best = numeric_limits<int>::max();
    for(int l=0; l<count_labels; ++l){
        auto const &m = msg_label[l];
        float b = 0.;
        for(auto [n, msg]: m){
            b += msg;
        }
        float val = b + f_node(this, l);
        if(val < belief_best){
            belief_best = val;
            label = l;
        }
    }
}

void N::accum_msg(){
    ///cache computation: collect neighbouring nodes' messages
    for(int l_cur=0; l_cur<count_labels;++l_cur){
        float accum = 0.;
        for(auto const neigh: neighbour){
            if(auto it = msg_label[l_cur].find(neigh); it!=msg_label[l_cur].end()){
                accum += it->second;
            }       
        }
        msg_neigh_accum[l_cur] = accum;
    }
}
    
void N::distribute_msg(function<float(N* const, int const)> f_node,
                       function<float(N* const, int const, N* const, int const)> f_edge){    
    ///distribute message using one with minimum value
    
    for(int l_cur=0; l_cur<count_labels;++l_cur){ //destination labels
        for(auto const other: neighbour){ //source nodes
            float val_best = numeric_limits<int>::max();
            for(int l_other=0; l_other<other->count_labels; ++l_other){
                float potential = f_node(this, l_cur) + f_edge(this, l_cur, other, l_other);
                float msg_redundant = 0.;
                if(auto const it = other->msg_label[l_other].find(this); it!=other->msg_label[l_other].end()){
                    msg_redundant = it->second;
                }
                float msg_neighbour = other->msg_neigh_accum[l_other] - msg_redundant;
                val_best = min(val_best, potential + msg_neighbour);
            }
            assert(l_cur<msg_label_swap.size());
            msg_label_swap[l_cur][other] = val_best; //incoming msg from other to current node's l_cur
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
                                  stage = (stage+1)%4;
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
                                  stage = (stage+1)%4;
                              sync_count = sync_count % processors;
                              cond_var.notify_all();
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              cond_var.wait( lock, [&](){return stage == 2;} );
                          }
                          for(int j= start; j < start+chunk; ++j){
                              auto i = ns[j];
                              i->accum_msg();
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              if(++sync_count == processors)
                                  stage = (stage+1)%4;
                              sync_count = sync_count % processors;
                              cond_var.notify_all();
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              cond_var.wait( lock, [&](){return stage == 3;} );
                          }
                          for(int j= start; j < start+chunk; ++j){
                              auto i = ns[j];
                              i->update_msg();
                          }
                          {
                              unique_lock<mutex> lock(mut);
                              if(++sync_count == processors)
                                  stage = (stage+1)%4;
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

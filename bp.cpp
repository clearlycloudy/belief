#include "bp.hpp"

#include <iostream>
#include <cassert>
#include <thread>

using namespace std;
    
N::N(int val): id(val) {}

void N::debug_label(){
    for(auto &[l, m]: msg_label){
	printf("id: %d, label: %d:", id, l);
	for(auto [n, v]: m){
	    printf("(id: %d, msg: %f) ", n->id, v);
	}
	printf("\n");
    }
}

void N::set_labels(vector<int> const l){
    assert(l.size()>0);
    for(auto i: l) msg_label[i] = {};
    labels = vector<int>(l.begin(),l.end());
}

//prerequisite: finish set_labels before using this
void N::set_neighbour_aux(N* n){
    assert(n!=this);
    neighbour.insert(n);
    for(auto &[l, m]: msg_label){
	m[n] = 0.;
    }
}
void N::set_neighbour(N* n){
    set_neighbour_aux(n);
    n->set_neighbour_aux(this);
}

void N::update_belief(function<double(N* const, int const)> f_node,
		      function<double(N* const, int const, N* const, int const)> f_edge){
    ///update label using arg min
    
    for(auto const & [l, m]: msg_label){
	double b = 0.;
	for(auto [n, msg]: m){
	    b += msg;
	}
	belief[l] = b + f_node(this, l);
    }

    //pick one label
    double v_best = numeric_limits<int>::max();
    int label_best;
    for(auto [l, v]: belief){
	if(v_best>v){
	    v_best = v;
	    label_best = l;
	}
    }
    label = label_best;
}
    
void N::distribute_msg(function<double(N* const, int const)> f_node,
		       function<double(N* const, int const, N* const, int const)> f_edge){
    
    ///distribute message using one with minimum value
    
    for(auto other: neighbour){
	for(auto l_other: other->labels){ //fix other's l_other, this is the target destination for message
	    double val_best = numeric_limits<int>::max();
	    for(auto l_cur: labels){ //go thru current node's labels

		double val = 0.;
		
		double potential = f_node(this, l_cur) + f_edge(this, l_cur, other, l_other);

		double msg_neighbour = 0.;
		for(auto neigh: neighbour){
		    if(neigh!=other){
			msg_neighbour += msg_label[l_cur][other];
		    }
		}
		val = potential + msg_neighbour;
	        val_best = min(val_best, val);
	    }
	    other->msg_label_swap[l_other][this] = val_best; //msg to other's l_other from current node
	}
    }
}

void N::update_msg(){
    swap(msg_label, msg_label_swap);
    msg_label_swap.clear();
}

void N::cycle(vector<N*> & ns,
	      function<double(N* const, int const)> f_node,
	      function<double(N* const, int const, N* const, int const)> f_edge){

    unsigned int const processors = thread::hardware_concurrency();
    vector<thread> t(processors);
    int chunk = ns.size()/processors;

    auto f_update_belief = [&](int t_idx, int chunk){
			       for(int j= t_idx*chunk; j < (t_idx+1)*chunk; ++j){
				   auto i = ns[j];
				   i->update_belief(f_node, f_edge);
			       }
			   };
    auto f_distribute = [&](int t_idx, int chunk){
			    for(int j= t_idx*chunk; j < (t_idx+1)*chunk; ++j){
				auto i = ns[j];
				i->distribute_msg(f_node, f_edge);
			    }
			};
    auto f_update_msg = [&](int t_idx, int chunk){
			    for(int j= t_idx*chunk; j < (t_idx+1)*chunk; ++j){
				auto i = ns[j];
				i->update_msg();
			    }
			};
    
    int remain = ns.size() % processors;

    for(int i=0;i<t.size();++i){
	t[i] = thread(f_update_belief, i, chunk);
    }
    f_update_belief(t.size(), remain);
    
    for(int i=0;i<t.size();++i){
	t[i].join();
    }
    
    for(int i=0;i<t.size();++i){
	t[i] = thread(f_distribute, i, chunk);
    }
    f_distribute(t.size(), remain);
    
    for(int i=0;i<t.size();++i){
	t[i].join();
    }

    for(int i=0;i<t.size();++i){
	t[i] = thread(f_update_msg, i, chunk);
    }
    f_update_msg(t.size(), remain);
    
    for(int i=0;i<t.size();++i){
	t[i].join();
    }

    // for(auto i: ns){
    // 	i->update_belief(f_node, f_edge);
    // }	
    // for(auto i: ns){
    // 	i->distribute_msg(f_node, f_edge);
    // }
    // for(auto i: ns){
    // 	i->update_msg();
    // }
}

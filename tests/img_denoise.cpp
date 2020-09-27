#include <iostream>
#include <vector>
#include <cassert>
#include <functional>
#include <cmath>
#include <random>
#include <unordered_map>

#include "../bp.hpp"
#include "lodepng.h"

using namespace std;

constexpr int KERNEL_SIZE_LABEL = 1;
constexpr int KERNEL_SIZE = 3;
constexpr int ITERATIONS = 3;
constexpr double FRAC_NEIGH = 0.2;
constexpr double FRAC_LABEL = 0.75;

default_random_engine gen;

#define VERBOSE

double colour_diff(unsigned char const * const c1, unsigned char const * const c2){
///approximate perceptual colour diff
    
    unsigned char r1 = c1[0];
    unsigned char g1 = c1[1];
    unsigned char b1 = c1[2];

    unsigned char r2 = c2[0];
    unsigned char g2 = c2[1];
    unsigned char b2 = c2[2];

    double rr = (r1+r2)/2.;
    
    return sqrt((2.+rr/256.)*(r1-r2)*(r1-r2) +
		4*(g1-g2)*(g1-g2) +
		(2.+(255.-rr)/256.)*(b1-b2)*(b1-b2));
}

vector<N*> init_tree(vector<unsigned char> const & img,
		     int const h,
		     int const w,
		     unordered_map<N*,pair<int,int>> & coordinate_map,
		     unordered_map<N*,unordered_map<int,pair<int,int>>> & label_map){
    
    assert(img.size()%4==0);
    vector<N*> ret(img.size()/4);
    
    int id = 0;
    for(auto &i: ret){
    	i = new N;
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
		label_map[n].clear();
		for(int k=-KERNEL_SIZE_LABEL;k<=KERNEL_SIZE_LABEL;++k){
		    for(int l=-KERNEL_SIZE_LABEL;l<=KERNEL_SIZE_LABEL;++l){
			int ii = i+k;
			int jj = j+l;
			uniform_real_distribution<float> distr(0.,1.);
			if(ii>=0 && ii<h &&
			   jj>=0 && jj<w &&
			   (ii!=i || jj!=j)&&
			   distr(gen) < FRAC_LABEL){
			    label_map[n][id_label] = {ii,jj};
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
	for(auto i: neigh){
	    n->set_neighbour(i);
	}
    }
    
    return ret;
}

int img_pixel(vector<unsigned char> const & img,
	      int const h,
	      int const w,
	      int const channel,
	      int y, int x){
    assert(y>=0&&y<h);
    assert(x>=0&&x<w);
    return (int) img[y*w*channel+x*channel];
}
vector<unsigned char> bp_run(vector<unsigned char> const & img,
			     int const h,
			     int const w){
    
    unordered_map<N*,pair<int,int>> coordinate_map;
    unordered_map<N*,unordered_map<int,pair<int,int>>> label_map;
    
    vector<N*> ns = init_tree(img, h, w, coordinate_map, label_map);
    assert(ns.size()>0);	
    
    auto potential_node = [&](N* const n,
			      int const l) -> double {
			      return 1.;
			  };

    auto potential_edge = [&](N* const n0,
			      int const l0,
			      N* const n1,
			      int const l1) -> double {
			      assert(label_map[n0].count(l0));
			      assert(label_map[n1].count(l1));
			      auto [y0, x0] = label_map[n0][l0];
			      auto [y1, x1] = label_map[n1][l1];
			      return colour_diff(&img[y0*w*4+x0*4], &img[y1*w*4+x1*4]);
			  };

    for(int t=0;t<ITERATIONS;++t){
#ifdef VERBOSE
        printf("iter: %d\n",t);
#endif
	N::cycle(ns, potential_node, potential_edge);
    }

    vector<unsigned char> out(img.size(),0);

    for(auto i: ns){
	auto [y,x] = coordinate_map[i];
	auto [yy,xx] = label_map[i][i->label];
	unsigned char r = img[yy*w*4+xx*4];
	unsigned char g = img[yy*w*4+xx*4+1];
	unsigned char b = img[yy*w*4+xx*4+2];
	out[y*w*4+x*4] = r;
	out[y*w*4+x*4+1] = g;
	out[y*w*4+x*4+2] = b;
	out[y*w*4+x*4+3] = 255;
    }
    
    for(auto i: ns){
	delete i;
    }

    return out;
}

int main(){
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



    

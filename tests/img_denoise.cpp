#include <iostream>
#include <vector>
#include <cassert>
#include <functional>
#include <cmath>
#include <random>
#include <unordered_map>
#include <thread>

#include "../bp.hpp"
#include "lodepng.h"

using namespace std;

constexpr int KERNEL_SIZE_LABEL = 2;
constexpr int KERNEL_SIZE = 2;
constexpr int ITERATIONS = 4;
constexpr float FRAC_NEIGH = 0.8;
constexpr float FRAC_LABEL = 0.8;

default_random_engine gen;

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
                              int const l) -> float {
                              return 1.;
                          };

    auto potential_edge = [&](N* const n0,
                              int const l0,
                              N* const n1,
                              int const l1) -> float {
                              assert(label_map[n0].count(l0));
                              assert(label_map[n1].count(l1));
                              auto [y0, x0] = label_map[n0][l0];
                              auto [y1, x1] = label_map[n1][l1];
                              return colour_diff(&img[y0*w*4+x0*4], &img[y1*w*4+x1*4]);
                          };

    N::cycle(ITERATIONS, ns, potential_node, potential_edge);

    vector<unsigned char> out(img.size(),0);

    auto f_save = [&](int start, int block_size){
                      for(int i=start; i<start+block_size; ++i){
                          auto n = ns[i];
                          auto [y,x] = coordinate_map[n];
                          auto [yy,xx] = label_map[n][n->get_label()];
                          out[y*w*4+x*4] = img[yy*w*4+xx*4];
                          out[y*w*4+x*4+1] = img[yy*w*4+xx*4+1];
                          out[y*w*4+x*4+2] = img[yy*w*4+xx*4+2];
                          out[y*w*4+x*4+3] = 255;
                      }
                  };

    int processors = thread::hardware_concurrency();
    processors = max(processors,1);
    
    vector<thread> t(processors-1);
    int chunk = ns.size() / processors;
    int remain = ns.size() % processors;

    for(int i=0;i<t.size();++i){
        t[i] = thread(f_save, i*chunk, chunk);
    }
    f_save(t.size()*chunk, chunk+remain);

    for(auto &i: t){
        i.join();
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



    

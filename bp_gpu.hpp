#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <set>
#include <initializer_list>

class N {
    
private:
    int id; //current node id
    int label; //selected label after optimization
    int count_labels = 0; //current node labels
    int count_neighbours = 0;
    float ** msg_label = nullptr; //holds incoming msg, label -> node_id -> message
    float ** msg_label_swap = nullptr; //temporary
    int * neighbours = nullptr; //neighbour node ids
    N** node_map = nullptr; //node_id -> N* (globally shared)
    
public:

    N(int identity);
    ~N();

    int get_id() const;
    
    void set_node_map(N** n_map);
    
    int get_label() const;
    
    void set_labels(int const count_labels);

    void set_neighbours(std::set<int> const & neigh_ids);
    
    static void cycle(int const iter,
                      std::vector<N*> & nodes,
                      std::function<float(N* const, int const)> f_node,
                      std::function<float(N* const, int const, N* const, int const)> f_edge);
    
    void update_belief(std::function<float(N* const, int const)> f_node,
                       std::function<float(N* const, int const, N* const, int const)> f_edge);
    
    void distribute_msg(std::function<float(N* const, int const)> f_node,
                        std::function<float(N* const, int const, N* const, int const)> f_edge);

    void accum_msg();
    
    void update_msg();

    void confirm_neighbours();

};

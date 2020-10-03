#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <set>
#include <initializer_list>

class N {
    
private:
    
    int label; //selected label after optimization
    int label_orig;
    int count_labels; //current node labels
    std::vector<std::unordered_map<N*,float>> msg_label; //holds incoming messages, label -> neighbour -> message
    std::vector<std::unordered_map<N*,float>> msg_label_swap;
    std::vector<N*> neighbour;
    std::vector<float> msg_neigh_accum;
public:

    int get_label() const;

    int get_label_orig() const;

    void set_label_orig(int l);

    void set_labels(int const count_labels);

    void set_neighbour(N* const n);
    
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

    void debug_label();
};

#include <map>
#include <functional>
#include <vector>
#include <set>
#include <initializer_list>

class N {
public:
    
    int id;
    int label;
    std::vector<int> labels; //current node labels
    std::map<int,std::map<N*,double>> msg_label; //holds incoming messages, label -> neighbour -> message
    std::map<int,std::map<N*,double>> msg_label_swap;
    std::set<N*> neighbour;
    std::map<int,double> belief;

    N(int val);

    void set_labels(std::vector<int> l);

    void set_neighbour(N* n);
    
    void set_neighbour_aux(N* n);
    
    static void cycle(std::vector<N*> & nodes,
		      std::function<double(N* const, int const)> f_node,
		      std::function<double(N* const, int const, N* const, int const)> f_edge);
    
    void update_belief(std::function<double(N* const, int const)> f_node,
		       std::function<double(N* const, int const, N* const, int const)> f_edge);
    
    void distribute_msg(std::function<double(N* const, int const)> f_node,
			std::function<double(N* const, int const, N* const, int const)> f_edge);
    
    void update_msg();

    void debug_label();
};

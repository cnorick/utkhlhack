#ifndef __node_h__
#define __node_h__

#include <vector>
#include <random>

using namespace std;

class Node {
    private:
    public:
        vector<Node*> prev; // The nodes that are providing inputs to this node.
        vector<Node*> next; // The nodes that this node is providing inputs to.
        vector<double*> weightsPrev; // The weight that each of the nodes from the previous layer has on this node.

        // The weight that this nodes has on each of the nodes in the next layer.
        // These are pointers to the weights held in the next layer's nodes'
        // weightsPrev vector. That way, they're always in sync.
        vector<double*> weightsNext;         

        double bias;

        static mt19937 weightGen; //(random_device());
        

    public:
        double sigma;
        double h;
        double delta;

        
        Node(vector<Node*> &inputs);

        // Call this one for the input layer.
        Node();

        ~Node();

        void setNext(vector<Node> &next);

        // Sets h, sigma, and delta to 0.
        void initialize();

        // Returns the weights between this neuron and the previous layer.
        vector<double*> getWeights();

        // Updates the sigma and h values of the node based on the weights of the previous layer.
        void updateOutput();

        // Back propagates the delta values from the next layer to this node.
        void backprop();

        // Updates the weight vector of the node based on the delta values
        // produced by backprop().
        void updateWeights(double learningRate);

        // Links this node with the next layer given by next.
        void setNext(vector<Node*> next);

};

#endif

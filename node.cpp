#include "node.h"
#include <random>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

mt19937 Node::weightGen = mt19937((random_device())());

Node::Node(vector<Node*> &inputs) {
    this->prev = inputs;

    uniform_real_distribution<> dis(-0.1, 0.1);

    bias = dis(weightGen);

    for(unsigned int k = 0; k < prev.size(); k++) {
        double* weight = new double(dis(weightGen));

        weightsPrev.push_back(weight);
    
        // Link the previous layer with this one.
        prev[k]->next.push_back(this);
        prev[k]->weightsNext.push_back(weight);
    }
}

Node::Node() {
    // no inputs are provided if this is the input layer.
}

Node::~Node() {
    for(unsigned int i = 0; i < weightsPrev.size(); i++)
        delete weightsPrev[i];
}

void Node::initialize() {
    sigma = h = delta = 0;
}

vector<double*> Node::getWeights() {
    return weightsPrev;
}

void Node::updateOutput() {
    for(unsigned int i = 0; i < prev.size(); i++) {
        h += *weightsPrev[i] * prev[i]->sigma;
    }
    h += bias;

    sigma = 1.0 / (1.0 + exp(-1.0 * h));
//    cout << "      sigma: " << sigma << endl;
}

void Node::backprop() {
    for(unsigned int i = 0; i < next.size(); i++) {
        delta += sigma * (1 - sigma) * next[i]->delta * (*weightsNext[i]);
    }
}

void Node::updateWeights(double learningRate) {
    for(unsigned int i = 0; i < prev.size(); i++) {
        *weightsPrev[i] += learningRate * delta * prev[i]->sigma;
    }

    bias += learningRate * delta;
}

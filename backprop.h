#ifndef __backprop_h__
#define __backprop_h__

#include <fstream>
#include <vector>
#include <ctime>
#include <random>
#include "node.h"
#include "pattern.h"

using namespace std;

class Backprop {
    private:
        int numLayers; // Including input and output.
        int *numNeurons; // per layer including input and output.
        double learningRate;
        int numEpochs;
        vector<Pattern*> training;
        vector<Pattern*> validation;
        vector<Pattern*> testing;


        vector<vector<Node*>> layers; // First index is layer, second is node in that layer.


    public:
        Backprop(int numLayers, int *numNeurons, double learningRate, int numEpochs,
                ifstream& training, ifstream& validation, ifstream& testing);
        ~Backprop();

       void trainNet();
       void validateNet(int i);
       void testNet();

       vector<double> getResult(const vector<double> &inputs);


    private:
        // parses the input from the ifstream into the vector.
        void parseInput(ifstream &, vector<Pattern*> &);
        void run();
        
        // Initializes all output values (h and sigma) and
        // delta values o heach hidden and output node to 0.
        void initialize();

        void printWeights();

        // Loads pattern p into the input layer.
        void loadPattern(const Pattern &p);

        // Compares the output with the pattern p and updates the output node's delta value.
        void checkOutput(const Pattern &p);
};

#endif

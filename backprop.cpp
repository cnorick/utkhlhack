#include "backprop.h"
#include "node.h"
#include "pattern.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <string>
#include <iostream>
#include <random>
#include <ctime>

using namespace std;

Backprop::Backprop(int numLayers, int *numNeurons, double learningRate, int numEpochs,
        ifstream& training, ifstream& validation, ifstream& testing)
:numLayers(numLayers), numNeurons(numNeurons), learningRate(learningRate), numEpochs(numEpochs)
{
    srand(time(NULL));

    // First layer is input layer and has no layer providing it input.
    layers.push_back(vector<Node*>());
    for(int i = 0; i < numNeurons[0]; i++) {
        layers[0].push_back(new Node());
    }

    // Set up each hidden/output layer to have inputs from the previous layer.
    for(int i = 1; i < numLayers; i++) {
        layers.push_back(vector<Node*>());
        for(int j = 0; j < numNeurons[i]; j++) {
            // Create a new layer and link it with the previous layer.
            layers[i].push_back(new Node(layers[i-1]));
        }
    }

    parseInput(training, this->training);
    parseInput(validation, this->validation);
    parseInput(testing, this->testing);

    run();
}

Backprop::~Backprop() {
    for(unsigned int i = 0; i < layers.size(); i++) {
        for(unsigned int j = 0; j < layers[i].size(); j++) {
            delete layers[i][j];
        }
    }

    for(unsigned int i = 0; i < training.size(); i++) {
        delete training[i];
    }
    for(unsigned int i = 0; i < validation.size(); i++) {
        delete validation[i];
    }
    for(unsigned int i = 0; i < testing.size(); i++) {
        delete testing[i];
    }
}

void Backprop::parseInput(ifstream &is, vector<Pattern*> &v) {
    int numInputs = numNeurons[0];
    int numOutputs = numNeurons[numLayers - 1];

    string line;
    while(getline(is, line)) {
        istringstream ss(line);

        vector<double> inputs;
        vector<double> outputs;
        for(int i = 0; i < numInputs; i++) {
            double val;
            ss >> val;
            inputs.push_back(val);
        }
        for(int i = 0; i < numOutputs; i++) {
            double val;
            ss >> val;
            outputs.push_back(val);
        }

        v.push_back(new Pattern(inputs, outputs));
    }
}

void Backprop::run() {
    for(int i = 0; i < numEpochs; i++){
        trainNet();
        validateNet(i);
    }
    testNet();
}

void Backprop::trainNet() {
    // for each training pattern.
    for(unsigned int i = 0; i < training.size(); i++) {
        initialize();

        loadPattern(*training[i]);
        
        // Update every hidden layer's h and sigma based on the last.
        for(unsigned int j = 1; j < layers.size(); j++) {
            for(unsigned int k = 0; k < layers[j].size(); k++) {
                Node *cur = layers[j][k];
                cur->updateOutput();
            }
        }

        // Update the output layer's delta values given the training
        // pattern's expected output.
        checkOutput(*training[i]);

        // Back propagate the delta values to all of the hidden layers.
        // Not the input or output layers.
        for(int j = layers.size() - 2; j >= 1; j--) {
            for(unsigned int k = 0; k < layers[j].size(); k++) {
                Node *cur = layers[j][k];
                cur->backprop();
            }
        }

        // Update the weights of all the layers based on the delta values.
        for(unsigned int j = 1; j < layers.size(); j++) {
            for(unsigned int k = 0; k < layers[j].size(); k++) {
                Node *cur = layers[j][k];
                cur->updateWeights(learningRate);
            }
        }
    }
}

void Backprop::validateNet(int epoch) {
    // Sum will hold the sum for each of the output nodes.
    double *sum = new double[layers.back().size()]();

    // Load each validation set into input node one at a time.
    for(unsigned int i = 0; i < validation.size(); i++) {
        initialize();
        Pattern v = *validation[i];

        loadPattern(v);


        // Update the outputs of each node from beginning to end.
        for(unsigned int j = 1; j < layers.size(); j++) {
            for(unsigned int k = 0; k < layers[j].size(); k++) {
                Node *cur = layers[j][k];
                cur->updateOutput();
            }
        }

        // Compare output to the expected value for each of the output nodes.
        for(unsigned int j = 0; j < layers.back().size(); j++) {
            double expectedoutput = v.output[j];
            double sigmaoutput = layers.back()[j]->sigma;
            sum[j] += pow(expectedoutput - sigmaoutput, 2);
        }
    }

    double rmse = 0;
    // Compute root mean square error for each of the output nodes.
    for(unsigned int i = 0; i < layers.back().size(); i++) {
        int numPatterns = validation.size();
        rmse += sqrt((1.0 / (2.0 * (double)numPatterns)) * sum[i]);
    }
    // epoch, error
    cout << epoch <<  ", " << rmse / (double)layers.back().size() << endl;

    delete sum;
}

void Backprop::testNet() {
    cout << "testing" << endl;
    validateNet(-1);
}

void Backprop::initialize() {
    vector<vector<Node*>>::iterator lit;
    vector<Node*>::iterator nit;

    // Initialize all nodes.
    for(lit = layers.begin(); lit != layers.end(); lit++) {
        for(nit = lit->begin(); nit != lit->end(); nit++) {
            (*nit)->initialize();
        }
    }
}

void Backprop::printWeights() {
    for(unsigned int i = 0; i < layers.size(); i++) {
        for(unsigned int j = 0; j < layers[i].size(); j++) {
            for(unsigned int k = 0; k < layers[i][j]->getWeights().size(); k++) {
                cout << "layer: " << i << ", neuron: " << j << ", connection: ";
                cout << k << ", weight: " << layers[i][j]->getWeights()[k] << endl;
            }
        }
    }
}

void Backprop::loadPattern(const Pattern &p) {
    for(unsigned int j = 0; j < p.input.size(); j++) {
        // store input values in input layer neurons as their sigma values.
        layers[0][j]->sigma = p.input[j];
    }
}

void Backprop::checkOutput(const Pattern &p) {
    vector<Node*> &b = layers.back();

    // Calculate the delta value for each node in the output layer.
    for(unsigned int i = 0; i < b.size(); i++) {
        double expectedOutput = p.output[i];

        b[i]->delta = b[i]->sigma * (1.0 - b[i]->sigma) * (expectedOutput - b[i]->sigma);
    }
}

vector<double> Backprop::getResult(const vector<double> &inputs) {
    Pattern p(inputs);
    vector<double> output;

    initialize();
    loadPattern(p);

    // Update every hidden layer's h and sigma based on the last.
    for(unsigned int j = 1; j < layers.size(); j++) {
        for(unsigned int k = 0; k < layers[j].size(); k++) {
            Node *cur = layers[j][k];
            cur->updateOutput();
        }
    }

    for(unsigned int i = 0; i < layers.back().size(); i++)
        output.push_back(layers.back()[i]->sigma);

    return output;
}

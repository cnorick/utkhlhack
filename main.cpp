#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string>
#include <sstream>
#include "backprop.h"

using namespace std;

void usage(char *exeName) {
    cerr << "Usage: " << exeName << " [-q] numNeurons in each layer - learningRate numEpochs trainingFile";
    cerr << " validationFile testingFile" << endl;
    cerr << "-q: after training the net, asks you for input." << endl;
}

int main(int argc, char **argv) {
    bool query = false;

    // Read in input
    if(argc < 6) { 
        usage(argv[0]);
        return 1;
    }

    vector<int> numNeurons;

    int i = 1;

    // Check for options.
    while(argv[i][0] == '-') {
        if(argv[i][1] == 'q')
            query = true;
        i++;
    }

    // Parse number of layers.
    while(argv[i][0] != '-' && i < argc) {
        numNeurons.push_back(atoi(argv[i]));
        i++;
    }
    i++;

    if(argc - i != 5) {
        usage(argv[0]);
        return 1;
    }

    int numLayers = numNeurons.size();
    double learningRate = atof(argv[i++]);
    int numEpochs = atoi(argv[i++]);
    char *trainingFile = argv[i++];
    char *validationFile = argv[i++];
    char *testingFile = argv[i++];

    ifstream training(trainingFile);
    ifstream validation(validationFile);
    ifstream testing(testingFile);

    // Run Backprop
    Backprop b(numLayers, numNeurons.data(), learningRate, numEpochs, training, validation, testing);


    if(query) {
        string in;
        cout << "insert values to test: ";
        while(getline(cin, in)) {
            vector<double> inputs;
            double n;

            istringstream is(in);

            while(is >> n)
                inputs.push_back(n);

            if((int)inputs.size() < numNeurons[0]) {
                cerr << numNeurons[0] << " inputs required, but only " << inputs.size() << " specified." << endl;
                continue;
            }


            vector<double> result = b.getResult(inputs);

            cout << "result: ";
            for(unsigned int k = 0; k < result.size(); k++)
                cout << result[k] << " ";
            cout << endl;
            cout << "insert values to test: ";
        }
    }
}


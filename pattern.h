#ifndef __pattern_h__
#define __pattern_h__

class Pattern {
    public:
        vector<double> input;
        vector<double> output;

        Pattern(vector<double> input, vector<double> output) {
            this->input = input;
            this->output = output;
        }

        Pattern(vector<double> input) {
            this->input = input;
        }
};

#endif

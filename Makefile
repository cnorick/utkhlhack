CXXFLAGS := -O3 -Wall -g -Wno-unused-function
CXXFLAGS += -std=c++11
CXX    := g++

all: backprop
	@echo compiling
	@ctags -R

backprop: backprop.o main.o node.o
	@$(CXX) $(CXXFLAGS) node.o backprop.o main.o -o backprop 

backprop.o: backprop.cpp backprop.h node.o pattern.h
	@$(CXX) $(CXXFLAGS) -c backprop.cpp

node.o: node.cpp node.h
	@$(CXX) $(CXXFLAGS) -c node.cpp
 
main.o: main.cpp backprop.o
	@$(CXX) $(CXXFLAGS) -c main.cpp
 
clean:
	@echo cleaning directory
	@rm -f backprop *.o tags

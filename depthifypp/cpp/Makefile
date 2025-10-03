SHELL := /bin/bash
CC := g++
CFLAGS := -std=c++17 -Wall -pedantic -O3
OMPFLAG := -fopenmp
DEBUG_FLAG := -g

CONDA_HOME := $(shell dirname $(shell dirname $(shell which conda)))
ENV := $(shell basename $(shell dirname $(shell dirname $(shell which python))))
LIB_DIR :=$(CONDA_HOME)/envs/$(ENV)/lib/
INCLUDE_DIR :=$(CONDA_HOME)/envs/$(ENV)/include/

OBJ := depthify_worker_utils.o 


all: echo $(OBJ) depthify_worker_main.o depthifypp

clean: clean_depthifypp
	rm ./*.o	

clean_depthifypp:
	rm ./depthifypp
	
echo:
	@echo CONDA HOME : $(CONDA_HOME)
	@echo ENV : $(ENV)
	@echo LIB DIR : $(LIB_DIR)
	@echo INCLUDE DIR : $(INCLUDE_DIR)
	@echo --------------------------------------------------
	@echo

%.o: %.cpp
	$(CC) $(CFLAGS) -L $(LIB_DIR) -I $(INCLUDE_DIR) -c $< -o $@ -lgdal -Wl,-rpath=$(LIB_DIR)

depthify_worker_utils.o: depthify_worker_utils.cpp
	$(CC) $(CFLAGS) -L $(LIB_DIR) -I $(INCLUDE_DIR) $(OMPFLAG) -c depthify_worker_utils.cpp -o depthify_worker_utils.o -lgdal -lboost_program_options -lboost_stacktrace_addr2line -ldl -Wl,-rpath=$(LIB_DIR)
	
depthify_worker_main.o: depthify_worker_main.cpp
	$(CC) $(CFLAGS) -L $(LIB_DIR) -I $(INCLUDE_DIR) $(OMPFLAG) -c depthify_worker_main.cpp -o depthify_worker_main.o -lgdal -lboost_program_options -lboost_stacktrace_addr2line -ldl -Wl,-rpath=$(LIB_DIR)

depthifypp: depthify_worker_main.o $(OBJ)
	$(CC) $(CFLAGS) -L $(LIB_DIR) -I $(INCLUDE_DIR) $(OMPFLAG)  depthify_worker_main.o $(OBJ) -o depthifypp -lgdal -lboost_program_options -lboost_stacktrace_addr2line -ldl -Wl,-rpath=$(LIB_DIR)
JAVA_INCLUDE = /usr/lib/jvm/java/include
EIGEN = ./Eigen
DYNLIB = logistic_ridge.so
RESOURCE_DIR = ../main/resources
RESOURCE = $(RESOURCE_DIR)/$(DYNLIB)

CXX = g++
CXXFLAGS = -O3 -fPIC -msse4
CPPFLAGS = -I$(JAVA_INCLUDE) -I$(JAVA_INCLUDE)/linux -I.
LDFLAGS =

SRC = logistic_ridge.cpp
OBJ = $(SRC:.cpp=.o)

all: $(RESOURCE)

$(EIGEN):
	wget http://bitbucket.org/eigen/eigen/get/3.2.7.tar.gz
	tar xzf 3.2.7.tar.gz
	mv eigen-eigen*/Eigen .
	rm 3.2.7.tar.gz
	rm -rf eigen-eigen*

%.o: %.cpp $(EIGEN)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(DYNLIB): $(OBJ)
	$(CXX) $(LDFLAGS) -shared -o $@ $^

$(RESOURCE): $(DYNLIB)
	mkdir -p $(RESOURCE_DIR)
	cp $(DYNLIB) $(RESOURCE)

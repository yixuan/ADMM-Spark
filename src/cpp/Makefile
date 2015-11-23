JAVA_INCLUDE = /usr/lib/jvm/java/include
EIGEN_INCLUDE = .
DYNLIB = logistic_ridge.so
RESOURCE_DIR = ../main/resources
RESOURCE = $(RESOURCE_DIR)/$(DYNLIB)

CXX = g++
CXXFLAGS = -O3 -fPIC -msse4
CPPFLAGS = -I$(JAVA_INCLUDE) -I$(JAVA_INCLUDE)/linux -I$(EIGEN_INCLUDE)
LDFLAGS =

SRC = logistic_ridge.cpp
OBJ = $(SRC:.cpp=.o)

all: $(RESOURCE)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(DYNLIB): $(OBJ)
	$(CXX) $(LDFLAGS) -shared -o $@ $^

$(RESOURCE): $(DYNLIB)
	mkdir -p $(RESOURCE_DIR)
	cp $(DYNLIB) $(RESOURCE)

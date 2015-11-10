JAVA_INCLUDE = /usr/lib/jvm/java/include
EIGEN_INCLUDE = .
LIBNAME = logistic_ridge.so

CXX = g++
CXXFLAGS = -O3 -fPIC -msse4
CPPFLAGS = -I$(JAVA_INCLUDE) -I$(JAVA_INCLUDE)/linux -I$(EIGEN_INCLUDE)
LDFLAGS =

SRC = logistic_ridge.cpp
OBJ = $(SRC:.cpp=.o)

all: $(LIBNAME)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(LIBNAME): $(OBJ)
	$(CXX) $(LDFLAGS) -shared -o $@ $^

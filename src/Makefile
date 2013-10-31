CC=g++

#The target is the executable file
TARGET_FOLDER=../bin/
TARGET=GestPlay

#Set all the flags, libraries, include files
CFLAGS= #-Wall
INC= -I../include `pkg-config opencv --cflags`
LFLAGS= `pkg-config opencv --libs` 
LIBS=          #-L../tree #-lGL -lGLU -lglut #-I/usr/include/GL

#All the source files and their object files
SRC=main.cpp MotionDetector.cpp
OBJS=$(SRC:.cpp=.o)

#
# The following part of the makefile is generic; it can be used to 
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#


.PHONY: depend clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INC) -o $(TARGET) $(OBJS) $(LFLAGS) $(LIBS)
 
.cpp.o:
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

depend: $(SRC)
	makedepend $(INC) $^

clean:
	rm -f *.o $(TARGET)

#


#CFLAGS=-c -Wall
#LDFLAGS=
#SOURCES=main.cpp hello.cpp factorial.cpp
#OBJECTS=$(SOURCES:.cpp=.o)
#EXECUTABLE=hello

#all: $(SOURCES) $(EXECUTABLE)
	
#$(EXECUTABLE): $(OBJECTS) 
#	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

#.cpp.o:
#	$(CC) $(CFLAGS) $< -o $@

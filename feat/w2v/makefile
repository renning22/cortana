CC = g++
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -std=c++11 -Wno-unused-result -I$(HOME)/private/nanoflann/include

all: translate

translate : translate.cpp
	$(CC) translate.cpp -o translate $(CFLAGS)
clean:
	rm -rf translate

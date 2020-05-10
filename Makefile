# assuming Eigen is in /usr/local/include
main:
	mkdir -p bin/
	clang++ -std=c++17 -c lib/net.cpp -o bin/net.o
	clang++ -std=c++17 -c example/main.cpp -o bin/main.o -I lib/
	clang++ bin/main.o bin/net.o -o bin/classifier
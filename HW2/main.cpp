#include "kernal.cuh"
#include <stdio.h>
#include <string>
int main(int argc, char ** argv) {
	std::string inputImage, outputImage;
	if(argc == 2)
		exec(argv[1], argv[2]);
	else
	{
		inputImage = std::string(DATAPATH) + "cinque_terre_small.jpg";
		outputImage = std::string(DATAPATH) + "HW2.jpg";
		exec(inputImage.c_str(), outputImage.c_str());
	}

	return 0;
}
#include <stdio.h>
#include <stdint.h>

#define FILTER_WIDTH 3

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(int * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

void writePnm(uchar3 *pixels, int width, int height,
			  char *fileName)
{
	FILE *f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "P3\n%i\n%i\n255\n", width, height);

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

	fclose(f);
}

// Convert RGB image to grayscale image
void convertRgb2Gray(uchar3* inPixels, int width, int height, uint8_t* outPixels)
{
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int i = r * width + c;
			uint8_t red = inPixels[i].x;
			uint8_t green = inPixels[i].y;
			uint8_t blue = inPixels[i].z;
			outPixels[i] =  0.299f * red + 0.587f * green + 0.114f * blue;
		}
	}
}

// Find the energy of a grayscale image
void findEnergy(uint8_t * inPixels, int width, int height,
                float * filterX, float * filterY, int filterWidth,
                int* outPixels)
{
	for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
	{
		for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
		{
            float outPixelX = 0;
			float outPixelY = 0;
			for (int filterR = 0; filterR < filterWidth; filterR++)
			{
				for (int filterC = 0; filterC < filterWidth; filterC++)
				{
					int inPixelsR = outPixelsR + filterR - filterWidth/2;
					int inPixelsC = outPixelsC + filterC - filterWidth/2;
					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);
                    uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];
                    outPixelX += inPixel * filterX[filterR * filterWidth + filterC];
                    outPixelY += inPixel * filterY[filterR * filterWidth + filterC];
				}
			}
			outPixels[outPixelsR * width + outPixelsC] = abs(outPixelX) + abs(outPixelY);
		}
	}
}

// Find the energy map and directions from the energy
void findEnergyMapAndDirections(int * energy, int width, int height,
            int * energyMap, int * directions)
{
    for (int c = 0; c < width; c++)
    {
        energyMap[(height - 1) * width + c] = energy[(height - 1) * width + c];
        directions[(height - 1) * width + c] = 0;
    }

	for (int r = height - 2; r >= 0; r--)
    {
		for (int c = 0; c < width; c++)
        {
            int leftC = max(c - 1, 0);
            int rightC = min(c + 1, width - 1);
            int min = energyMap[(r + 1) * width + leftC];
            int nextIdx = leftC;
            for (int i = leftC + 1; i <= rightC; i++)
            {
                int currentEnergy = energyMap[(r + 1) * width + i];
                if (currentEnergy < min)
                {
                    min = currentEnergy;
                    nextIdx = i;
                }
            }
			      energyMap[r * width + c] = energy[r * width + c] + min;
            directions[r * width + c] = nextIdx;
		}
	}
}

// Find seam at start position is c
void findSeamAt(int c, int * directions, int width, int height, int * seam)
{
    seam[0] = c;
    for (int r = 1; r < height; r++)
    {
        seam[r] = directions[(r - 1) * width + seam[r - 1]];
    }
}

// Find the seam having the smallest energy
void findSeam(int * energyMap, int * directions, int width, int height, int * seam)
{
    int seamIdx = 0;
    int minSeamEnergy = energyMap[0];
    for (int c = 1; c < width; c++)
    {
        int curSeamEnergy = energyMap[c];
        if (curSeamEnergy < minSeamEnergy)
        {
            minSeamEnergy = curSeamEnergy;
            seamIdx = c;
        }
    }
    findSeamAt(seamIdx, directions, width, height, seam);
}

// Remove seam
void removeSeam(uchar3 * inPixels, int width, int height, int * seam,
                uchar3 * outPixels)
{
    int newWidth = width - 1;
	for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < seam[r]; c++)
            outPixels[r * newWidth + c] = inPixels[r * width + c];
        for (int c = seam[r]; c < newWidth; c++)
            outPixels[r * newWidth + c] = inPixels[r * width + c + 1];
	}
}

// Draw seam
void drawSeam(uchar3 * inPixels, int width, int height, int * seam)
{
    for (int r = 0; r < height; r++)
        inPixels[r * width + seam[r]] = make_uchar3(0, 0, 0);
}

void seamCarvingByHost(uchar3 * inPixels, int width, int height, int newWidth,
                    float * filterX, float * filterY, int filterWidth,
                    uchar3 * outPixels)
{
	int * seam, * directions;
    uint8_t * grayImg;
	int * energy, * energyMap;
    uchar3 * tempImg,*srcInPixels;

    seam = (int *)malloc(height * sizeof(int));
    tempImg = (uchar3 *)malloc(width * height * sizeof(uchar3));
    srcInPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    memcpy(srcInPixels,inPixels,width * height * sizeof(uchar3));
	for (int i = 0; i < width - newWidth; i++)
    {
        int currentWidth = width - i;

        directions = (int *)malloc(currentWidth * height * sizeof(int));
        grayImg = (uint8_t *)malloc(currentWidth * height * sizeof(uint8_t));
        energy = (int *)malloc(currentWidth * height * sizeof(int));
        energyMap = (int *)malloc(currentWidth * height * sizeof(int));


		    convertRgb2Gray(srcInPixels, currentWidth, height, grayImg);
        findEnergy(grayImg, currentWidth, height, filterX, filterY, filterWidth, energy);
        findEnergyMapAndDirections(energy, currentWidth, height, energyMap, directions);
		findSeam(energyMap, directions, currentWidth, height, seam);
        removeSeam(srcInPixels, currentWidth, height, seam, tempImg);

        // Swap srcInPixels and tempImg
        uchar3 * temp;
        temp = srcInPixels;
        srcInPixels = tempImg;
        tempImg = temp;

        // Free memory
        free(directions);
        free(grayImg);
        free(energy);
        free(energyMap);
	}
    memcpy(outPixels,srcInPixels, newWidth * height*sizeof(uchar3));
    free(seam);
    free(tempImg);
}

// PARALLEL SEAM CARVING
__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, 
		uint8_t * outPixels)
{ 
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (c < width && r < height)
    {
        int i = r * width + c;
        float red = inPixels[i].x;
        float green = inPixels[i].y;
        float blue = inPixels[i].z;
        outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
}

__global__ void findEnergyKernel(uint8_t * inPixels, int width, int height, 
                    float * filterX, float * filterY, int filterWidth, 
                    int * outPixels)
{
    int outPixelsR = blockIdx.y * blockDim.y + threadIdx.y;
    int outPixelsC = blockIdx.x * blockDim.x + threadIdx.x;

    if (outPixelsR < height && outPixelsC < width)
    { 
        float outPixelX = 0;
        float outPixelY = 0;
        for (int filterR = 0; filterR < filterWidth; filterR++)
        {
            for (int filterC = 0; filterC < filterWidth; filterC++)
            {
                int inPixelsR = outPixelsR + filterR - filterWidth/2;
                int inPixelsC = outPixelsC + filterC - filterWidth/2;
                inPixelsR = min(max(0, inPixelsR), height - 1);
                inPixelsC = min(max(0, inPixelsC), width - 1);
                uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];
                outPixelX += inPixel * filterX[filterR * filterWidth + filterC];
                outPixelY += inPixel * filterY[filterR * filterWidth + filterC];
            }
        }
        outPixels[outPixelsR * width + outPixelsC] = abs(outPixelX) + abs(outPixelY);
    }
}


__global__ void findEnergyMapAndDirectionsKernel(int * energy, int width, int height,
            int * energyMap, int * directions, int r)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
  

	if (c >= width)
        return;

	int leftC = max(c - 1, 0);
	int rightC = min(c + 1, width - 1);
	int min = energyMap[(r + 1) * width + leftC];
	int nextIdx = leftC;
	for (int i = leftC + 1; i <= rightC; i++)
	{
        int currentEnergy = energyMap[(r + 1) * width + i];
        if (currentEnergy < min)
        {
            min = currentEnergy;
            nextIdx = i;
        }
	}
	energyMap[r * width + c] = energy[r * width + c] + min;
	directions[r * width + c] = nextIdx;
}

__global__ void findMinKernel(int* row, int* mins, int*indices,
    int n) {
	extern __shared__ int sharedMins[];

	int nElement = blockIdx.x * blockDim.x * 2;

	sharedMins[threadIdx.x]= nElement+threadIdx.x;
	sharedMins[threadIdx.x+blockDim.x] =nElement+threadIdx.x+blockDim.x;
	__syncthreads();

	 for (int strike = 1; strike <=blockDim.x; strike *= 2){
		 	int i = nElement + threadIdx.x * 2 * strike;
			if (threadIdx.x < blockDim.x / strike && i+strike<n)
			{
				if ( row[i+strike]< row[i]) {
					row[i]=row[i+strike];
					int s_idx=threadIdx.x*2*strike;
					sharedMins[s_idx]=sharedMins[s_idx+strike];
				}
			}
      __syncthreads();
	 }

	if(threadIdx.x==0){
		mins[blockIdx.x ] = row[blockIdx.x * blockDim.x*2];
		indices[blockIdx.x ] = sharedMins[threadIdx.x];
	}

}



__global__ void removeSeamKernel(uchar3 * inPixels, int width, int height, int * seam,
                                uchar3 * outPixels)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (r < height)
	{
    if (c < seam[r])
    {
      outPixels[r * (width-1) + c] = inPixels[r * width + c];
    }
    else if (c < width - 1)
    {
      outPixels[r * (width-1) + c] = inPixels[r * width + c + 1];
    }   
	}
}



void seamCarvingByDevice(uchar3 * inPixels, int width, int height, int newWidth,
                        float * filterX, float * filterY, int filterWidth,
                        uchar3 * outPixels, dim3 blockSize, int version)
{
    // Allocate device memories (main components)
    uchar3 * d_inPixels;
    float * d_filterX, * d_filterY;

    size_t nBytesInPixels = width * height * sizeof(uchar3);
    size_t nBytesOutPixels = newWidth * height * sizeof(uchar3);
    size_t nBytesFilter = filterWidth * filterWidth * sizeof(float);

    CHECK(cudaMalloc(&d_inPixels, nBytesInPixels));
    CHECK(cudaMalloc(&d_filterX, nBytesFilter));
    CHECK(cudaMalloc(&d_filterY, nBytesFilter));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_inPixels, inPixels, nBytesInPixels, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filterX, filterX, nBytesFilter, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filterY, filterY, nBytesFilter, cudaMemcpyHostToDevice));

    //
    int * seam;
    seam = (int *)malloc(height * sizeof(int));

    // Allocate device memories (temporary parameters)
    int * d_seam, * d_directions;
    uint8_t * d_grayImg;
    int * d_energy, * d_energyMap;
    uchar3 * d_tempImg;
    int *temp_row= (int*)malloc(width*sizeof(int));
    size_t nBytesSeam = height * sizeof(int);

    CHECK(cudaMalloc(&d_seam, nBytesSeam));
    CHECK(cudaMalloc(&d_tempImg, nBytesInPixels));


    CHECK(cudaMalloc(&d_directions,  width * height * sizeof(int)));
    CHECK(cudaMalloc(&d_grayImg,  width * height * sizeof(uint8_t) ));
    CHECK(cudaMalloc(&d_energy,  width * height * sizeof(int) ));
    CHECK(cudaMalloc(&d_energyMap,  width * height * sizeof(int)));

    for (int i = 0; i < width - newWidth; i++)
    {
        int currentWidth = width - i;

        // Continue allocate temporary parameters
        size_t nElements = currentWidth * height;


        // Calculate grid size
        dim3 gridSize((currentWidth - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

        // Call convertRgb2GrayKernel
        convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, currentWidth, height, d_grayImg);
        cudaDeviceSynchronize();
        // Call findEnergyKernel
        findEnergyKernel<<<gridSize, blockSize>>>(d_grayImg, currentWidth, height, d_filterX, d_filterY, filterWidth, d_energy);
        cudaDeviceSynchronize();
        // Call findEnergyMapAndDirectionsKernel
        
        cudaMemcpy(temp_row, &d_energy[(height - 1) * currentWidth], currentWidth * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_energyMap[(height - 1) * currentWidth],temp_row , currentWidth * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(&d_directions[(height - 1) * currentWidth], 0, currentWidth * sizeof(int));
        for (int r = height - 2; r >= 0; r--)
        {
            
            findEnergyMapAndDirectionsKernel<<<gridSize.x, blockSize.x>>>(d_energy, currentWidth, height, d_energyMap, d_directions, r);
            cudaDeviceSynchronize();
        }
        //

        int * directions = (int *)malloc(currentWidth * height * sizeof(int));
        CHECK(cudaMemcpy(directions, d_directions, nElements * sizeof(int), cudaMemcpyDeviceToHost));


        // find min and min index in row
        int mins_size= gridSize.x*sizeof(int);
        int* mins = (int*) malloc(mins_size);
        int* min_indices = (int*) malloc(mins_size);
        int* d_mins;
        int* d_min_indices;
        CHECK(cudaMalloc( &d_mins, mins_size));
        CHECK(cudaMalloc(&d_min_indices, mins_size));

        // Use the kernel function to find intermediate minimums
        int numBlk=  (currentWidth-1)/(blockSize.x*2)+1;
        findMinKernel<<<numBlk, blockSize.x,(blockSize.x*2)*sizeof(int)>>>(d_energyMap, d_mins, d_min_indices, currentWidth);

        // Compute final minimum
        cudaMemcpy(mins, d_mins, mins_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(min_indices, d_min_indices, mins_size,cudaMemcpyDeviceToHost);

        int minimum = mins[0];
        int seamIdx = min_indices[0];

        for (int j = 1; j <numBlk ; j++) {
          if (mins[j] < minimum) {
            minimum = mins[j];
            seamIdx = min_indices[j];
          }
        }
        findSeamAt(seamIdx, directions, currentWidth, height, seam);


        free(mins);
        free(min_indices);

        CHECK(cudaMemcpy(d_seam, seam, nBytesSeam, cudaMemcpyHostToDevice));

        // Call removeSeamKernel
        removeSeamKernel<<<gridSize, blockSize>>>(d_inPixels, currentWidth, height, d_seam, d_tempImg);
        cudaDeviceSynchronize();
        // Swap d_inPixels and d_tempImg
        uchar3 * d_temp;
        d_temp = d_inPixels;
        d_inPixels = d_tempImg;
        d_tempImg = d_temp;

        // Free memory
        free(directions);
     

    }

    CHECK(cudaMemcpy(outPixels, d_inPixels, nBytesOutPixels, cudaMemcpyDeviceToHost));

    // Free memory
    free(seam);
    free(temp_row);
    CHECK(cudaFree(d_directions));
    CHECK(cudaFree(d_grayImg));
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_energyMap));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_filterX));
    CHECK(cudaFree(d_filterY));
    CHECK(cudaFree(d_seam));
    CHECK(cudaFree(d_tempImg));
}

void seamCarving(uchar3 * inPixels, int width, int height, int newWidth,
                float * filterX, float * filterY, int filterWidth,
                uchar3 * outPixels,
                bool useDevice=false, dim3 blockSize=dim3(1, 1), int version=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
        printf("\nSeam carving by host\n");
        seamCarvingByHost(inPixels, width, height, newWidth, filterX, filterY, filterWidth, outPixels);
    }
    else
    {
        printf("\nSeam carving by device\n");
        seamCarvingByDevice(inPixels, width, height, newWidth, filterX, filterY, filterWidth, outPixels, blockSize, version);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

void printError(uchar3 * deviceResult, uchar3 * hostResult, int width, int height)
{
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
}

int main(int argc, char ** argv)
{
    // CHECK NUMBER OF ARGUMENTS
    // 0: name of input image file
    // 1: number of seam deleted
    // 2: blockSize.x
    // 3: blockSize.y
    // if (argc !=2 && argc != 4)
	// {
	// 	printf("The number of arguments is invalid\n");
	// 	return EXIT_FAILURE;
	// }

    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // READ INPUT IMAGE FILE
    int width, height;
    uchar3 * inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);

    // CHECK NEW RESOLUTION IS ACCEPTABLE
    int newWidth = atoi(argv[3]);
    if (newWidth < 0 || newWidth > width)
    {
        printf("\nNew resolution is not acceptable\n");
        return EXIT_FAILURE;
    }

    // DEFINE SOBEL FILTER
    float sobel_x[9] = {1.0, 0.0, -1.0,
                        2.0, 0.0, -2.0,
                        1.0, 0.0, -1.0};
    float sobel_y[9] = {1.0, 2.0, 1.0,
                        0.0, 0.0, 0.0,
                        -1.0, -2.0, -1.0};

    // ALLOCATE MEMORIES
    int filterWidth = FILTER_WIDTH;

    // SEAM CARVING BY HOST
    uchar3 * correctOutPixels = (uchar3 *)malloc(newWidth * height * sizeof(uchar3));
    seamCarving(inPixels, width, height, newWidth, sobel_x, sobel_y, filterWidth, correctOutPixels);
    writePnm(correctOutPixels, newWidth, height, (char*)"host.pnm");
    

    // DETERMINE BLOCK SIZE
    dim3 blockSize(32, 32); // Default 
    if (argc == 6){
        blockSize.x = atoi(argv[4]);
        blockSize.x = atoi(argv[5]);

    }

    // SEAM CARVING BY DEVICE
    uchar3 * outPixels = (uchar3 *)malloc(newWidth * height * sizeof(uchar3));
    seamCarving(inPixels, width, height, newWidth, sobel_x, sobel_y, filterWidth, outPixels, true, blockSize, 1);

    printError(outPixels, correctOutPixels, newWidth, height);

    writePnm(outPixels, newWidth, height, argv[2]);

    
    return EXIT_SUCCESS;
}

#include <stdio.h>
#include <stdint.h>

#define FILTER_WIDTH 3

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

// SEQUENTIAL SEAM CARVING
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
    // Allocate parameters
	int * seam, * directions;
    uint8_t * grayImg;
	int * energy, * energyMap;
    uchar3 * tempImg, * srcInPixels;

    seam = (int *)malloc(height * sizeof(int));
    tempImg = (uchar3 *)malloc(width * height * sizeof(uchar3));
    srcInPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));

    // Copy data from inPixels to srcInPixels
    memcpy(srcInPixels, inPixels, width * height * sizeof(uchar3));

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

    // Copy result to outPixels
    memcpy(outPixels, srcInPixels, newWidth * height * sizeof(uchar3));

    // Free memory
    free(seam);
    free(tempImg);
    free(srcInPixels);
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
        printf("New image size (width x height): %i x %i\n", newWidth, height);
        seamCarvingByHost(inPixels, width, height, newWidth, filterX, filterY, filterWidth, outPixels);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main(int argc, char ** argv)
{
    // CHECK NUMBER OF ARGUMENTS
    // 1: name of input image file
    // 2: name of output image file
    // 3: new width 1
    // 4: new width 2
    // 5: new width 3
    if (argc != 6)
	{
		printf("The number of arguments is invalid\n");
	 	return EXIT_FAILURE;
	}

    // READ INPUT IMAGE FILE
    int width, height;
    uchar3 * inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);

    // CHECK NEW RESOLUTION IS ACCEPTABLE
    int newWidth1 = atoi(argv[3]);
    int newWidth2 = atoi(argv[4]);
    int newWidth3 = atoi(argv[5]);
    if (newWidth1 < 0 || newWidth1 > width ||
        newWidth2 < 0 || newWidth2 > width ||
        newWidth3 < 0 || newWidth3 > width)
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

    // SEAM CARVING BY HOST WITH WIDTH 1
    uchar3 * outPixels1 = (uchar3 *)malloc(newWidth1 * height * sizeof(uchar3));
    seamCarving(inPixels, width, height, newWidth1, sobel_x, sobel_y, filterWidth, outPixels1);

    // SEAM CARVING BY HOST WITH WIDTH 2
    uchar3 * outPixels2 = (uchar3 *)malloc(newWidth2 * height * sizeof(uchar3));
    seamCarving(inPixels, width, height, newWidth2, sobel_x, sobel_y, filterWidth, outPixels2);

    // SEAM CARVING BY HOST WITH WIDTH 3
    uchar3 * outPixels3 = (uchar3 *)malloc(newWidth3 * height * sizeof(uchar3));
    seamCarving(inPixels, width, height, newWidth3, sobel_x, sobel_y, filterWidth, outPixels3);
    
    // WRITE RESULTS TO FILES
    char * outFileNameBase = strtok(argv[2], ".");
    writePnm(outPixels1, newWidth1, height, concatStr(outFileNameBase, "_host1.pnm"));
    writePnm(outPixels2, newWidth2, height, concatStr(outFileNameBase, "_host2.pnm"));
    writePnm(outPixels3, newWidth3, height, concatStr(outFileNameBase, "_host3.pnm"));

    // FREE MEMORIES
    free(inPixels);
    free(outPixels1);
    free(outPixels2);
    free(outPixels3);
    
    return EXIT_SUCCESS;
}

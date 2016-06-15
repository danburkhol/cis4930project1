/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the rc machines
   ==================================================================
*/

/* 
	Daniel Burkholder
	June 6 2016

	USF Summer 2016
	CIS 4930 - Programming Massively Parallel Systems

	Project 1
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;



bucket *histogram;		/* list of all buckets in the histogram   */
long long PDH_acnt;		/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double PDH_res;			/* value of w                             */
atom *atom_list;		/* list of all data points                */



// CUDA Error Check
void checkCudaError(cudaError_t e, char in[]) {
	if (e != cudaSuccess) {
		printf("CUDA Error: %s, %s \n", in, cudaGetErrorString(e));
		exit(EXIT_FAILURE);
	}
}


__device__ double
p2p_distance(atom *a, int ind1, int ind2) {
	double x1 = a[ind1].x_pos;
	double x2 = a[ind2].x_pos;

	double y1 = a[ind1].y_pos;
	double y2 = a[ind2].y_pos;

	double z1 = a[ind1].z_pos;
	double z2 = a[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


__global__ void 
PDH_baseline(bucket *histo, atom *atomList, double w, int size) {
	int i, j, pos;
	double dist;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = i + 1;

	

	for (int a = j; a < size; a++) {
		//printf("i: %d, j: %d \n", i, j);
		dist = p2p_distance(atomList, i, a);
		pos = (int) (dist / w);

		//__syncthreads();
		//histo[pos].d_cnt++;
		atomicAdd( &histo[pos].d_cnt, 1);
		//__syncthreads();
	}

	//printf("\n");

}

__global__ void
PDH2D_baseline(bucket *histo, atom *atomList, double w) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i < j) {
		double dist = p2p_distance(atomList, i, j);
		int pos = (int) (dist / w);

		histo[pos].d_cnt++;

		printf("%d, %d : %d, %f \n", i, j, pos, dist);
	}

	__syncthreads();

}




void output_histogram(bucket *histo){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histo[i].d_cnt);
		total_cnt += histo[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char const *argv[])
{
	PDH_acnt = atoi(argv[1]);	// Number of atoms
	PDH_res = atof(argv[2]);	// Input Distance: W

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;

	size_t histogramSize = sizeof(bucket)*num_buckets;
	size_t atomSize = sizeof(atom)*PDH_acnt;


	histogram = (bucket *)malloc(histogramSize);
	atom_list = (atom *)malloc(atomSize);

	srand(1);
	/* generate data following a uniform distribution */
	for(int i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}




	// Malloc space on device, copy to device
	bucket *d_histogram = NULL;
	atom *d_atom_list = NULL;

	checkCudaError( cudaMalloc((void**) &d_histogram, histogramSize), 
		"Malloc Histogram");
	checkCudaError( cudaMalloc((void**) &d_atom_list, atomSize), 
		"Malloc Atom List");

	checkCudaError( cudaMemcpy(d_histogram, histogram, histogramSize, cudaMemcpyHostToDevice), 
		"Copy histogram to Device");
	checkCudaError( cudaMemcpy(d_atom_list, atom_list, atomSize, cudaMemcpyHostToDevice), 
		"Copy atom_list to Device");


	// Setup: Measure Runtime
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	// CUDA Kernel Call
	PDH_baseline <<<ceil(PDH_acnt/32), 32>>> (d_histogram, d_atom_list, PDH_res, PDH_acnt);

	checkCudaError(cudaGetLastError(), "Checking Last Error, Kernel Launch");


	// Report kernel runtime
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to generate: %f ms \n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);



	checkCudaError( cudaMemcpy(histogram, d_histogram, histogramSize, cudaMemcpyDeviceToHost),
		"Copy device histogram to host");


	output_histogram(histogram);


	checkCudaError(cudaFree(d_histogram), "Free device histogram");
	checkCudaError(cudaFree(d_atom_list), "Free device atom_list");

	free(histogram);
	free(atom_list);

	checkCudaError(cudaDeviceReset(), "Device reset");


	return 0;
}











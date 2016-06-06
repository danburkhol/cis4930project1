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
	long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;

	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;

	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}


__device__ double
cuda_p2p_distance(atom *a, int ind1, int ind2) {
	
	double x1 = a[ind1].x_pos;
	double x2 = a[ind2].x_pos;

	double y1 = a[ind1].y_pos;
	double y2 = a[ind2].y_pos;

	double z1 = a[ind1].z_pos;
	double z2 = a[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

__global__ void
cuda_PDH_baseline(bucket *ahisto, atom *atomlist, double w) {
	int i, j, pos;
	i = blockIdx.y * blockDim.y + threadIdx.y;	// x

	j = blockIdx.x * blockDim.x + threadIdx.x;	// y


	
	if (j > i) {
		double dist = cuda_p2p_distance(atomlist, i, j);
		pos = (int) (dist / w);
		ahisto[pos].d_cnt++;
	}
	

	//printf("i: %d, j: %d \n", i, j);
}


/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


void checkCudaError(cudaError_t e, char in[]) {
	if (e != cudaSuccess) {
		printf("CUDA ERROR: %s, %s \n", in, cudaGetErrorString(e));
		exit(EXIT_FAILURE);
	}
}




int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram();

	// ========== ==========
	// Reset data
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	cudaError_t err = cudaSuccess;


	size_t histogramSize = sizeof(bucket)*num_buckets;
	size_t atomSize = sizeof(atom)*PDH_acnt;

	// Setup CUDA Memory and Malloc
	bucket *d_histogram = NULL;
	atom *d_atomList = NULL;


	err = cudaMalloc( (void **) &d_histogram, histogramSize);
	checkCudaError(err, "Malloc d_histogram");

	err = cudaMalloc( (void **) &d_atomList, atomSize);
	checkCudaError(err, "Malloc d_atomList");


	// Copy Host to Device Memory
	// Target, Source, Size, function

	err = cudaMemcpy(d_histogram, histogram, histogramSize, cudaMemcpyHostToDevice);
	checkCudaError(err, "MemCpy histogram, host to device");

	err = cudaMemcpy(d_atomList, atom_list, atomSize, cudaMemcpyHostToDevice);
	checkCudaError(err, "MemCpy atomList, host to device");



	// 32 threads per block
	double numThreads = 32.0;
	double numBlocks = ceil(PDH_acnt / numThreads) + 1.0;

	dim3 dimBlock(numBlocks, 1, 1);
	dim3 dimGrid(16, 16, 1);

	// <<< blocks, threads >>>
	printf("Launching CUDA Kernel with %f Blocks of 32 Threads\n", numBlocks);


	// ===== Launch Kernel =====

	gettimeofday(&startTime, &Idunno);

	cuda_PDH_baseline<<<dimBlock, dimGrid>>>(d_histogram, d_atomList, PDH_res);

	err = cudaGetLastError();
	checkCudaError(err, "Launching Kernel");


	// Copy results, modified d_histogram back to host histogram
	// (target, source, size, function)
	err = cudaMemcpy(histogram, d_histogram, histogramSize, cudaMemcpyDeviceToHost);
	checkCudaError(err, "MemCpy histogram, device to host");

	printf("Running time for GPU (Ignore 'CPU') \n");
	report_running_time();

	output_histogram();
	

	// Free Device Memory
	err = cudaFree(d_histogram);
	checkCudaError(err, "Free histogram");

	err = cudaFree(d_atomList);
	checkCudaError(err, "Free atomList");

	// Frea Host Memory
	free(histogram);
	free(atom_list);

	//Reset Device
	err = cudaDeviceReset();
	checkCudaError(err, "Device Reset");

	return 0;
}



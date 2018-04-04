#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__
void resolveWeight(int8_t *A,int8_t *B,int8_t *C, int n, int wx){

	int k = blockIdx.x*blockDim.x+threadIdx.x;

	int off = k * n;

	int8_t temp = 0;
	int b = 0;

	if(k<wx){
		for(int i=0;i<n;i++){
			b = off+i;
			temp+=A[i]*B[b];
		}
		C[k]=temp;
	}


}

int main(){

	FILE *fout = fopen("out.txt","w");
	int T = 512;
	int nodes,weightsX,weightsY;

	for(int o = 1; o < 10000;o++){
		if(o%1000 == 0){
			printf("Starting run %d\n" ,o);
		}
		fprintf(fout,"%d",o);
		nodes = o;
		weightsX = o;
		weightsY = o;
		int8_t *node, *d_node, *weight, *d_weight,*out,*gout,*d_out;

		node = (int8_t *)malloc(nodes*sizeof(int8_t));
		weight = (int8_t *)malloc(weightsX*weightsY*sizeof(int8_t));
		out = (int8_t *)malloc(weightsX*sizeof(int8_t));
		gout = (int8_t *)malloc(weightsX*sizeof(int8_t));

		srand(time(0));

		//Set up matrix 1
		for(int i = 0;i<nodes;i++){
			node[i] = rand() % 3 - 1;
		}

		//Set up matrix 2
		for(int i = 0;i<(weightsX*weightsY);i++){
			weight[i] = rand() % 3 - 1;
		}

		//Set up matrices for results
		for(int i = 0;i<weightsX;i++){
			out[i] = 0;
			gout[i] = 0;
		}

		unsigned int sstart = clock();
		for(int i = 0; i<weightsX;i++){
			for(int j = 0; j<weightsY;j++){
				out[i]+=(node[j]*weight[(nodes*i)+j]);
			}
		}
		//printf("Sequential time taken in ms %li\n" ,(clock() - sstart));
		int seqtime = clock()-sstart;
		fprintf(fout,",%d",seqtime);

		//CUDA parallel code
		cudaMalloc(&d_node,nodes*sizeof(int8_t));
		cudaMalloc(&d_weight,weightsX*weightsY*sizeof(int8_t));
		cudaMalloc(&d_out,weightsX*sizeof(int8_t));

		cudaMemcpy(d_node,node,nodes*sizeof(int8_t),cudaMemcpyHostToDevice);
		cudaMemcpy(d_weight,weight,weightsX*weightsY*sizeof(int8_t),cudaMemcpyHostToDevice);
		cudaMemcpy(d_out,gout,weightsX*sizeof(int8_t),cudaMemcpyHostToDevice);

		unsigned int pstart = clock();
		resolveWeight<<<weightsX+(T-1)/T,T>>>(d_node,d_weight,d_out,nodes,weightsX);
		cudaDeviceSynchronize();
		//printf("Parallel time taken in ms %li\n" ,(clock() - pstart));
		int partime = clock()-pstart;
		fprintf(fout,",%d",partime);

		cudaMemcpy(gout,d_out,weightsX*sizeof(int8_t),cudaMemcpyDeviceToHost);


//		printf("Value at 3: %i\n",out[2]);
//		printf("Value on gpu at 3: %i\n",gout[2]);
//		printf("Value at 5120: %i\n",out[5119]);
//		printf("Value on gpu at 5120: %i\n",gout[5119]);
//		printf("Value at 10000: %i\n",out[9999]);
//		printf("Value on gpu at 10000: %i\n",gout[9999]);

		int err = 0;
		for(int i = 0; i < weightsX; i++){
			err += out[i]-gout[i];
		}
//		printf("Error with CUDA: %i\n",err);
		fprintf(fout,",%d\n",err);

		free(node);
		free(weight);
		free(out);
		free(gout);
		cudaDeviceReset();
	}
	fclose(fout);
}

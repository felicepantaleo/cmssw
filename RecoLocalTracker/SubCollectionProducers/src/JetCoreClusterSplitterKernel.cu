texture<float, 1, cudaReadModeElementType>tex;

__global__ void kernel (void)
{
	  int i = blockIdx.x *blockDim.x + threadIdx.x;
	  float x = tex1Dfetch(tex, i);


}


extern "C" void cudaClusterSplitter_(int* gpu_mapcharge) {

	// bind texture to buffer
	cudaBindTexture(0, tex, gpu_mapcharge, 168000*sizeof(int));
	dim3 block(128,1,1);
	dim3 grid(block.x,1,1);
    kernel<<<grid,block>>>();
    cudaUnbindTexture(tex);



}

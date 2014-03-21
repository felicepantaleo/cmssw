//__global__ void shiftPitchLinear(Element* element)
//{
//  int xid = blockIdx.x * blockDim.x + threadIdx.x;
//  if(xid < element->length)
//     element->timestamp[xid] += 1+tex1D(texRef, element->xHit[xid] + 64*element->yHit[xid]);
//}



__global__ void kernel (void)
{
    float chi2 = 0;

}


extern "C" void cudaClusterSplitter_(void) {

    kernel<<<1,1>>>();
    return;


}

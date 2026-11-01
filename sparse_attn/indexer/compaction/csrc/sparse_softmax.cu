// Experimental virtual token-domain softmax, no production dispatch.
// Reduction scheduling follows PyTorch v2.8.0 ATen/native/cuda/SoftMax.cu
// (generic CUDA kernel only); BSD-3-Clause upstream license in this directory.
// Changes: gather sparse scores through per-token owners; retain only row stats
// between passes instead of materializing the dense score input.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cfloat>
#include <climits>
#include <cmath>

namespace {
template<typename T> struct Add {
  __device__ T operator()(T a,T b) const {return a+b;}
  __device__ T combine(T a,T b) const {return a+b;}
  __device__ T warp_shfl_down(T a,int delta) const {return WARP_SHFL_DOWN(a,delta);}
};
template<typename T> struct Max {
  __device__ T operator()(T a,T b) const {return a<b?b:a;}
  __device__ T combine(T a,T b) const {return a<b?b:a;}
  __device__ T warp_shfl_down(T a,int delta) const {return WARP_SHFL_DOWN(a,delta);}
};
template<template<typename> class Op>
__device__ float block_reduce(float value,float *shared,float initial) {
  float result=at::native::cuda_utils::BlockReduce<float,Op<float>>(value,Op<float>(),initial,shared);
  if(threadIdx.x==0) shared[0]=result;
  __syncthreads();
  return shared[0];
}


template<typename T>
__global__ void sparse_stats(const T* scores,const int64_t* perm,const int64_t* keys,
                            const int64_t* offsets,float* stats,int R,int G) {
  __shared__ float shared[32];
  int h=blockIdx.x,kv=h/G,t=threadIdx.x;
  int start=offsets[kv*1025+t],stop=offsets[kv*1025+t+1];
  float maximum=-FLT_MAX;
  int64_t previous=-1;
  for(int i=start;i<stop;++i) {
    int64_t key=keys[(int64_t)kv*R+i];
    if(key!=previous) {
      int r=perm[(int64_t)kv*R+i];
      maximum=::max(maximum,static_cast<float>(scores[(int64_t)h*R+r]));
    }
    previous=key;
  }
  maximum=block_reduce<Max>(maximum,shared,-FLT_MAX);
  float sum=0.f;previous=-1;
  for(int i=start;i<stop;++i) {
    int64_t key=keys[(int64_t)kv*R+i];
    if(key!=previous) {
      int r=perm[(int64_t)kv*R+i];
      float x=static_cast<float>(scores[(int64_t)h*R+r]);
      sum=sum+std::exp(x-maximum);
    }
    previous=key;
  }
  sum=block_reduce<Add>(sum,shared,0.f);
  if(t==0) {stats[2*h]=maximum;stats[2*h+1]=sum;}
}
template<typename T>
__global__ void candidate_probability(const T* scores,const float* stats,T* out,int R) {
  int h=blockIdx.y,r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r<R) out[(int64_t)h*R+r]=static_cast<T>(std::exp(static_cast<float>(scores[(int64_t)h*R+r])-stats[2*h])/stats[2*h+1]);
}
}
at::Tensor sparse_softmax(at::Tensor scores,at::Tensor perm,at::Tensor keys,at::Tensor offsets,int64_t G) {
  TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==at::kBFloat16 && scores.dim()==2 && scores.is_contiguous(),"contiguous CUDA BF16 scores required");
  for(auto t:{perm,keys,offsets}) TORCH_CHECK(t.device()==scores.device() && t.scalar_type()==at::kLong && t.is_contiguous(),"contiguous int64 metadata on same device required");
  TORCH_CHECK(G>0 && perm.dim()==2 && keys.sizes()==perm.sizes() && perm.size(0)*G==scores.size(0) && perm.size(1)==scores.size(1),"candidate shape mismatch");
  TORCH_CHECK(offsets.dim()==2 && offsets.size(0)==perm.size(0) && offsets.size(1)==1025 && scores.size(1)<INT_MAX,"invalid sparse offsets");
  c10::cuda::CUDAGuard device_guard(scores.device());
  auto stats=at::empty({scores.size(0),2},scores.options().dtype(at::kFloat));
  auto out=at::empty_like(scores);
  auto stream=at::cuda::getCurrentCUDAStream();
  using T=at::BFloat16;
  sparse_stats<T><<<scores.size(0),1024,0,stream>>>(scores.data_ptr<T>(),perm.data_ptr<int64_t>(),keys.data_ptr<int64_t>(),offsets.data_ptr<int64_t>(),stats.data_ptr<float>(),scores.size(1),G);
  if(scores.size(1)>0) candidate_probability<T><<<dim3((scores.size(1)+255)/256,scores.size(0)),256,0,stream>>>(scores.data_ptr<T>(),stats.data_ptr<float>(),out.data_ptr<T>(),scores.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

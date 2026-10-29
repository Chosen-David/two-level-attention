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

template<typename T> __device__ T gather_score(const T* scores,const int* owners,int h,int token,int R,int S,int G) {
  int r=owners[(h/G)*S+token];
  return r>=0?scores[(int64_t)h*R+r]:static_cast<T>(-INFINITY);
}

template<bool EXP,typename T,int ILP>
__device__ float thread_reduce(const T* scores,const int* owners,int h,int R,int S,int G,float maximum) {
  // Preserve native per-thread token order, including row alignment handling.
  int shift=(static_cast<int64_t>(h)*S)%ILP;
  int size=S,base=0,offset=threadIdx.x;
  float value=EXP?0.f:-FLT_MAX;
  auto consume=[&](int token) {
    float x=static_cast<float>(gather_score(scores,owners,h,token,R,S,G));
    if constexpr (EXP) value=value+std::exp(x-maximum);
    else value=::max(value,x);
  };
  if(shift>0) {
    base-=shift;size+=shift;
    if(offset>=shift && offset<size) consume(base+offset);
    size-=blockDim.x>size?size:blockDim.x;
    base+=blockDim.x;
  }
  int last=size%(ILP*blockDim.x);
  for(;offset*ILP<size-last;offset+=blockDim.x) {
    #pragma unroll
    for(int j=0;j<ILP;++j) consume(base+offset*ILP+j);
  }
  offset=size-last+threadIdx.x;
  for(;offset<size;offset+=blockDim.x) consume(base+offset);
  return value;
}

template<typename T,int ILP>
__global__ void normalize_stats(const T* scores,const int* owners,float* stats,int R,int S,int G) {
  __shared__ float shared[32];
  int h=blockIdx.x;
  float maximum=block_reduce<Max>(thread_reduce<false,T,ILP>(scores,owners,h,R,S,G,0.f),shared,-FLT_MAX);
  float sum=block_reduce<Add>(thread_reduce<true,T,ILP>(scores,owners,h,R,S,G,maximum),shared,0.f);
  if(threadIdx.x==0) {stats[2*h]=maximum;stats[2*h+1]=sum;}
}

template<typename T>
__global__ void materialize_probability(const T* scores,const int* owners,const float* stats,T* out,int R,int S,int G) {
  int h=blockIdx.y;
  int token=blockIdx.x*blockDim.x+threadIdx.x;
  if(token<S) {
    float score=static_cast<float>(gather_score(scores,owners,h,token,R,S,G));
    // Preserve native std::exp/divide and final model-dtype rounding.
    out[(int64_t)h*S+token]=static_cast<T>(std::exp(score-stats[2*h])/stats[2*h+1]);
  }
}
}

template<typename T>
__global__ void compact_probability(const T* scores,const float* stats,T* out,int R) {
  int h=blockIdx.y,r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r<R) {
    float x=static_cast<float>(scores[(int64_t)h*R+r]);
    out[(int64_t)h*R+r]=static_cast<T>(std::exp(x-stats[2*h])/stats[2*h+1]);
  }
}

// Fuse per-head probability rounding and GQA reduction without a [H,R]
// probability intermediate. The full token output preserves native TopK ties.
template<typename T>
__global__ void fused_probability_mean(const T* scores,const int* owners,
                                      const float* stats,T* out,int R,int S,int G) {
  int kv=blockIdx.y,token=blockIdx.x*blockDim.x+threadIdx.x;
  if(token>=S) return;
  int r=owners[kv*S+token];
  float sum=0.f;
  if(r>=0) {
    for(int g=0;g<G;++g) {
      int h=kv*G+g;
      float x=static_cast<float>(scores[(int64_t)h*R+r]);
      T rounded=static_cast<T>(std::exp(x-stats[2*h])/stats[2*h+1]);
      float probability=static_cast<float>(rounded);
      sum+=std::isnan(probability)?0.f:probability;
    }
  }
  out[(int64_t)kv*S+token]=static_cast<T>(sum/G);
}

at::Tensor virtual_compact_softmax(at::Tensor scores,at::Tensor owners,int64_t S,int64_t G,bool fused_mean) {
  TORCH_CHECK(scores.is_cuda() && owners.is_cuda() && scores.device()==owners.device(),"same CUDA device required");
  TORCH_CHECK(scores.is_contiguous() && owners.is_contiguous(),"contiguous inputs required");
  TORCH_CHECK(scores.dim()==2 && owners.dim()==2 && owners.scalar_type()==at::kInt,"matrix scores/int32 owners required");
  TORCH_CHECK(G>0 && scores.size(0)==owners.size(0)*G && owners.size(1)==S,"GQA shape mismatch");
  TORCH_CHECK(S>2048 && S<INT_MAX && scores.size(1)<INT_MAX,"unsupported size");
  TORCH_CHECK(scores.scalar_type()==at::kHalf || scores.scalar_type()==at::kBFloat16 || scores.scalar_type()==at::kFloat,"fp16/bf16/fp32 required");
  TORCH_CHECK(owners.numel()<INT_MAX,"owner indexing exceeds int32 prototype scope");
  c10::cuda::CUDAGuard device_guard(scores.device());
  auto prop=at::cuda::getCurrentDeviceProperties();
  int bytes=scores.element_size(),ilp=16/bytes;
  TORCH_CHECK((S+1023)/1024>=10,"native reference would use register softmax; fallback required");
  bool native_smem=S<(prop->sharedMemPerBlock-128)/bytes && S%ilp==0;
  TORCH_CHECK(!native_smem,"native reference would use shared softmax; fallback required");
  auto stats=at::empty({scores.size(0),2},scores.options().dtype(at::kFloat));
  auto out=fused_mean ? at::empty({owners.size(0),S},scores.options())
                      : at::empty({scores.size(0),scores.size(1)},scores.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,scores.scalar_type(),"virtual_softmax",[&] {
    constexpr int ILP=16/sizeof(scalar_t);
    normalize_stats<scalar_t,ILP><<<scores.size(0),1024,0,stream>>>(scores.data_ptr<scalar_t>(),owners.data_ptr<int>(),stats.data_ptr<float>(),scores.size(1),S,G);
    if(fused_mean) fused_probability_mean<scalar_t><<<dim3((S+255)/256,owners.size(0)),256,0,stream>>>(scores.data_ptr<scalar_t>(),owners.data_ptr<int>(),stats.data_ptr<float>(),out.data_ptr<scalar_t>(),scores.size(1),S,G);
    else if(scores.size(1)>0) compact_probability<scalar_t><<<dim3((scores.size(1)+255)/256,scores.size(0)),256,0,stream>>>(scores.data_ptr<scalar_t>(),stats.data_ptr<float>(),out.data_ptr<scalar_t>(),scores.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

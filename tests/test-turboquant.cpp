// TurboQuant CPU reference tests: FWHT self-inverse, roundtrip MSE, bit-packing
#include "ggml.h"
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
static constexpr int HEAD_DIM=128, BLOCK_SIZE=32, BLOCKS_PER_CHUNK=HEAD_DIM/BLOCK_SIZE, N_VECTORS=10000, N_ELEMENTS=N_VECTORS*HEAD_DIM;
static void fwht_f32(float * x,int n){for(int h=1;h<n;h*=2)for(int i=0;i<n;i+=h*2)for(int j=i;j<i+h;j++){float a=x[j],b=x[j+h];x[j]=a+b;x[j+h]=a-b;}float s=1.0f/sqrtf((float)n);for(int i=0;i<n;i++)x[i]*=s;}
static int test_fwht(){float a[HEAD_DIM],b[HEAD_DIM];for(int i=0;i<HEAD_DIM;i++)a[i]=sinf((float)(i+1)*.7f)*2;memcpy(b,a,sizeof(a));fwht_f32(b,HEAD_DIM);fwht_f32(b,HEAD_DIM);float e=0;for(int i=0;i<HEAD_DIM;i++)e=fmaxf(e,fabsf(b[i]-a[i]));printf("  FWHT self-inverse max_err=%.2e %s\n",e,e<1e-5?"ok":"FAILED");return e<1e-5?0:1;}
static void random_vectors(float * d,int n,unsigned s){for(int i=0;i<n;i++){s=s*1664525u+1013904223u;d[i]=((float)(s>>8)/(float)(1<<24))*2-1;}}
static int test_mse(ggml_type t,float lo,float hi){auto * tr=ggml_get_type_traits(t);assert(tr->from_float_ref&&tr->to_float);std::vector<float>s(N_ELEMENTS),d(N_ELEMENTS);std::vector<uint8_t>q((size_t)N_ELEMENTS/BLOCK_SIZE*ggml_type_size(t));random_vectors(s.data(),N_ELEMENTS,42);tr->from_float_ref(s.data(),q.data(),N_ELEMENTS);tr->to_float(q.data(),d.data(),N_ELEMENTS);double n=0;for(int v=0;v<N_VECTORS;v++){double e=0,z=0;for(int i=0;i<HEAD_DIM;i++){double x=s[v*HEAD_DIM+i],y=d[v*HEAD_DIM+i];e+=(x-y)*(x-y);z+=x*x;}if(z>1e-20)n+=e/z;}float mse=n/N_VECTORS;printf("  %s roundtrip MSE*d=%.4f [%.3f..%.3f] %s\n",ggml_type_name(t),mse,lo,hi,mse>=lo&&mse<=hi?"ok":"FAILED");return mse>=lo&&mse<=hi?0:1;}
static int test_bits(ggml_type t,bool sanity){auto * tr=ggml_get_type_traits(t);float s[HEAD_DIM],d[HEAD_DIM];for(int i=0;i<HEAD_DIM;i++)s[i]=cosf(i*.31415f);size_t n=BLOCKS_PER_CHUNK*ggml_type_size(t);std::vector<uint8_t>a(n),b(n);tr->from_float_ref(s,a.data(),HEAD_DIM);if(!sanity){tr->from_float_ref(s,b.data(),HEAD_DIM);bool ok=memcmp(a.data(),b.data(),n)==0;printf("  %s pack determinism %s\n",ggml_type_name(t),ok?"ok":"FAILED");return ok?0:1;}tr->to_float(a.data(),d,HEAD_DIM);bool finite=true,nz=false;for(float x:d){finite&=std::isfinite(x);nz|=fabsf(x)>1e-10f;}printf("  %s dequantize finite=%s nonzero=%s %s\n",ggml_type_name(t),finite?"yes":"NO",nz?"yes":"NO",finite&&nz?"ok":"FAILED");return finite&&nz?0:1;}
int main(){printf("TurboQuant CPU reference tests\n==============================\n");int f=0;f+=test_fwht();f+=test_mse(GGML_TYPE_TURBO3_0,.025f,.045f);f+=test_mse(GGML_TYPE_TURBO4_0,.005f,.015f);f+=test_bits(GGML_TYPE_TURBO3_0,false);f+=test_bits(GGML_TYPE_TURBO4_0,false);f+=test_bits(GGML_TYPE_TURBO3_0,true);f+=test_bits(GGML_TYPE_TURBO4_0,true);printf("==============================\n%d/7 tests passed\n",7-f);return f?1:0;}

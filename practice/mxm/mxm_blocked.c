#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#define N 2000
#define MAX( a, b ) ( ( a > b) ? a : b )
#define MIN( a, b ) ( ( a < b) ? a : b )

double x[N][N];
double y[N][N];
double z[N][N];

double timer(void){
  struct timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec + time.tv_usec/1000000.0;
}

int main(int argc, char **argv){
  int i,j,k;
  int ii,jj,kk;
  int B;
  double r;
  double ts,tf;

  if (argc < 2)
    B = 16;
  else
    B = atoi(argv[1]);

  
  /* initialize y, z */
  for (i=0;i<N;++i){
    for (j=0;j<N;++j){
      y[i][j] = 1;
      z[i][j] = 1;
    }
  }

  ts = timer();
  for (jj=0;jj<N;jj+=B){
    for (kk=0;kk<N;kk+=B){
      for (i=0;i<N;++i){
	    for (j=jj;j<MIN(jj+B,N);++j){
	        r = 0;
	        for (k=kk;k<MIN(kk+B,N);++k){
                r += y[i][k]*z[k][j];
            }
	        x[i][j] += r;
	    }
      }
    }
  }
  tf = timer();
  /*
  for (int jj = 0;j<N;j+=B){
    for (int kk=0;k<N;kk+=B){
        for(int i=0;i<N;i++){
            for(int j=jj;j<MIN(N,jj+B);j++){
                int r=0;
                for(int k=kk;k<MIN(N,kk+B);k++){
                    r+=y[i][k]+z[k][j]
                }
                x[i][j] += r;
            }
        }
    }
  }
  */
  printf("%f %f(s)\n", x[0][0],tf-ts);
}

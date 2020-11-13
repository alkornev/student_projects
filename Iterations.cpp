#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <omp.h>


double** fillMatrix(int dim_A1, int dim_A2){
  double** res;
  res = (double**)malloc(dim_A1*sizeof(double*));

  for(int i = 0; i < dim_A1; i++){
    res[i] = (double*)malloc(dim_A2*sizeof(double));
  }

  for(int i = 0; i < dim_A1; i++){
    for(int j = 0; j < dim_A2; j++){
      if (i == j) res[i][j] = -2;
      else res[i][j] = 1;
    }
  }
  return res;
}

double** fillMatrixRandom(int dim_A1, int dim_A2){
  double** res;
  res = (double**)malloc(dim_A1*sizeof(double*));

  for(int i = 0; i < dim_A1; i++)
    res[i] = (double*)malloc(dim_A2*sizeof(double));


  for(int i = 0; i < dim_A1; i++){
    for(int j = i; j < dim_A2; j++){
      res[i][j] = 2*((double)rand() / (double)RAND_MAX) - 1;
      res[j][i] = res[i][j];
    }
  }
  return res;
}

double* fillArray(int dim){
  double* res;
  res = (double*)malloc(dim*sizeof(double));
  for(int i = 0; i < dim; i++){
    res[i] = 1;
  }
  return res;
}

double* fillArrayRandom(int dim){
  double* res;
  res = (double*)malloc(dim*sizeof(double));
  for(int i = 0; i < dim; i++){
    res[i] = 2*((double)rand() / (double)RAND_MAX) - 1;
  }
  return res;
}

double dot(double* a, int dim_a, double* b, int dim_b){
  if (dim_a != dim_b){
    std::cout << "Dimensions don't match" << std::endl;
    exit(-1);
  }
  double res = 0;
  int i;

#pragma omp parallel for shared(a, b, dim_a) reduction(+:res)
  for (i = 0; i < dim_a; i++){
      res += a[i]*b[i];
  }
  return res;
}


double normvec(double* x, int dim){
  return sqrt(dot(x, dim, x, dim));
}


void normalize(double* x, int dim){
  double norm = normvec(x, dim);
  int i;
#pragma omp parallel for
    for(i = 0; i < dim; i++){
      x[i] = x[i]/norm;
  }
}

void matvec(double** A, int dim_A1, int dim_A2, double* x, int dim_x, double* res, int dim_res){
  if (dim_A2 != dim_x || dim_x != dim_res) {
    std::cout << "Dimensions don't match" << std::endl;
    exit(-1);
  }

  double buffer[dim_res];
  int i,j;

#pragma omp parallel for private(j) shared(A, x)
  for (i = 0; i < dim_A1; i++){
    buffer[i] = 0;
    for (j = 0; j < dim_A2; j++){
      buffer[i] += A[i][j] * x[j];
    }
  }

#pragma omp parallel for
  for(i = 0; i < dim_x; i++){
    res[i] = buffer[i];
  }
}




double residue(double** A, int dim_A1, int dim_A2, double* x, int dim_x, double lambda){
  double row = 0;
  double res = 0;
  int i,j;
  #pragma omp parallel for private (i,j) shared(dim_A1, dim_A2, A, x) reduction(+:row,res)
  for(i = 0; i < dim_A1; i++){
    for(j = 0; j < dim_A2; j++){
      row += A[i][j]*x[j];
    }
    row -= lambda*x[i];
    res += row*row;
  }
  return sqrt(res);
}

void printArray(double* x, int dim_x){
  std::cout << "vector = ";
  for(int i = 0; i < dim_x;i++){
    std::cout << std::setw(10) << x[i];
  }
  std::cout <<  std::endl;
}

void printMatrix(double** A, int dim_A1, int dim_A2){
  std::cout << "matrix = "<< std::endl;
  for(int i = 0; i < dim_A1; i++){
    for(int j = 0; j < dim_A2; j++){
      std::cout << std::setw(10) << A[i][j];
    }
    std::cout << std::endl;
  }
}


int main(){
  //initialize matrix and starting vector
  int dim = 1000;
  int dim_x = dim, dim_A1 = dim, dim_A2 = dim;

  double mintol = 1e-10;
  double lambda;
  double norm;
  double tol;
  double time;

  double** A = fillMatrixRandom(dim_A1, dim_A2);
  double* x = fillArrayRandom(dim_x);
  double* Ax = (double *)malloc(dim_x*sizeof(double));

  //printArray(x, dim_x);
  //printMatrix(A, dim_A1, dim_A2);
  std::ofstream f;
  f.open("example.txt", std::ofstream::out);


  int threads = 9;
  for(int i = 1; i < threads; i++){
    omp_set_num_threads(i);
    f << "num of threads: " << i << std::endl;
    for(int j = 0; j < 5; j++){
      f << j << ": ";
      x = fillArrayRandom(dim_x);
      normalize(x, dim_x);
      matvec(A, dim_A1, dim_A2, x, dim_x, Ax, dim_x);
      time = omp_get_wtime();
      do {
        lambda = dot(x, dim_x, Ax, dim_x);
        tol = residue(A, dim_A1, dim_A2, x, dim_x, lambda);
        matvec(A, dim_A1, dim_A2, x, dim_x, x, dim_x);
        normalize(x, dim_x);
        matvec(A, dim_A1, dim_A2, x, dim_x, Ax, dim_x);
        //std::cout << "tol: " << tol << " lambda: " << lambda << std::endl;
      } while(tol > mintol);
      time = omp_get_wtime() - time;
      f << "time: " << time << std::endl;
      //std::cout <<"elapsed time: " << time << " with: " << i << " threads"<< std::endl;
      free(x);
    }
  }
  f.close();
  for(int i = 0; i < dim_A1; i++){
    free(A[i]);
  }
  free(A);
  free(Ax);
  return 0;
}

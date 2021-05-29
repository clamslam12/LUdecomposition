//-----------------------------------------------------------------------
// LU Decompsition with forward and back substitution to solve for vector x in Ax = b : C++ OpenMP 
//-----------------------------------------------------------------------
//  Programming by: Minh Durbin
//-----------------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>  
#include <vector>
using namespace std;
//-----------------------------------------------------------------------
//   Get user input of matrix dimension and printing option
//-----------------------------------------------------------------------
bool GetUserInput(int argc, char *argv[],int& n,int& numThreads)
{
	bool isOK = true;

	if(argc < 2) 
	{
		cout << "Arguments:<X> <Y> [<Z>]" << endl;
		cout << "X : Matrix size [X x X]" << endl;
		cout << "Y : Number of threads" << endl;
		isOK = false;
	}
	else 
	{
		//get matrix size
		n = atoi(argv[1]);
		if (n <=0) 
		{
			cout << "Matrix size must be larger than 0" <<endl;
			isOK = false;
		}

		//get number of threads
		numThreads = atoi(argv[2]);
		if (numThreads <= 0)
		{	cout << "Number of threads must be larger than 0" <<endl;
			isOK = false;
		}
	
	}
	return isOK;
}

//change m's 2nd dimension size for custom size; for array param testing
void printMatrix(float** m, int n){
	for(int i = 0; i < n; i++){

		cout<< "Row " << (i+1) << ":\t" ;
		for(int j =0; j< n; j++){
			printf("%.2f\t", m[i][j]);
		}
		cout << endl;
	}
}
//change a and lwr's 2nd dimension size for custom size; for array param testing
void ComputeGaussianElimination2(float** a,int n, float** lwr)
{
	float pivot,temp;
	int i,j,k;
	//k<n-1 works ; k < n+1 dont
	for(int k = 0; k < n-1; k++){
	//Compute the pivot
		pivot = -1.0/a[k][k];
        //#pragma omp parallel shared(a,gmax,gindmax) firstprivate(n,k) private(pivot,i,j,temp,pmax,pindmax)
		//Perform row reductions
		//#pragma omp parallel for shared(a) firstprivate(pivot,n,k) private(i,j,temp) schedule(dynamic)
        #pragma omp parallel for shared(k, pivot, a, n, lwr) private(i, j, temp) schedule(static)
		    for (i = k+1 ; i < n; i++)
		    {
			    temp = pivot*a[i][k];

			    for (j = k ; j < n ; j++)
			    {
				    a[i][j] = a[i][j] + temp*a[k][j];
					//get the opposite sign mutliplier and initialize in the same index for the lower matrix
					lwr[i][j] = (-1.0)*temp;	
			    }
		    }
	}
}
//change lwr's 2nd dimension size for custom size; for array param testing
void initDiagonalAndUpper(float** lwr, int n){
#pragma omp parallel for schedule(static)
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(i==j){
				lwr[i][j] = 1.0;
			}else if( j > i){//if column > row
				lwr[i][j] = 0.0;
			}
		}
	}
}
//change lwr's 2nd dimension size for custom size; for array param testing
void forwardSub(float* x, float* b, float** lwr, int n){
	int i, j;
	for(j = 0; j < n; j++){
		x[j] = b[j] / lwr[j][j];
		#pragma omp parallel for shared(j, b, lwr, n, x) private(i) schedule(static)
		for(i=j+1; i< n; i++){
			b[i] -= lwr[i][j]*x[j];
		}

	}

}
//change a's 2nd dimension size for custom size; for array param testing
void backSub(float* x2, float* x, float** a, int n){
	int i, j;
	for(j = n-1; j >=0; j--){
		x2[j] = x[j] / a[j][j];
		#pragma omp parallel for shared(j, x2, x, a, n) private(i) schedule(static)
		for(i=0; i<j+1;i++){
			x[i] -= a[i][j]*x2[j];
		}
	}

}

void InitializeMatrixA(float** a, int n)
{
	#pragma omp parallel for schedule(static) 
	for (int i = 0 ; i < n ; i++)
	{
		for (int j = 0 ; j < n ; j++)
		{	
            if (i == j) 
              a[i][j] = (((float)i+1)*((float)i+1))/(float)2;	
            else
              a[i][j] = (((float)i+1)+((float)j+1))/(float)2;
		}
	}
}

void InitializeMatrixLwr(float** lwr, int n)
{
	#pragma omp parallel for schedule(static) 
	for (int i = 0 ; i < n ; i++)
	{
		for (int j = 0 ; j < n ; j++)
		{	
            lwr[i][j] = 0.0;
		}
	}
}

void InitializeBVector(float* b, int n)
{
	
	#pragma omp parallel for schedule(static) 
	for(int i = 0 ; i<n; i++){
		b[i] = (((float)i+1)*((float)i+1))/(float)2;
	}
}
void printVector(float* v, int n){

	for(int i = 0 ; i<n; i++){
		cout << v[i] << endl;
	}
}

//------------------------------------------------------------------
// Main Program
//------------------------------------------------------------------
int main(int argc, char *argv[])
{
	//n = matrix size(1st argument), numThreads(2nd argument)
	int n,numThreads;
	double runtime;
	
	if (GetUserInput(argc,argv,n,numThreads)==false) return 1;

	//specify number of threads created in parallel region
	omp_set_num_threads(numThreads);

	//dynamically allocate memory
	
	float** a = new float*[n];
	float** lwr = new float*[n];
	float* b = new float[n];
	float* x2 = new float[n];
	float* x = new float[n];

	for(int i=0;i<n;i++){
		lwr[i] = new float[n];
	}
	
	for(int i=0;i<n;i++){
		a[i] = new float[n];
	}

	//initialize matrices and vectors
	InitializeMatrixA(a, n);
	InitializeBVector(b, n);
	InitializeMatrixLwr(lwr, n);

	cout << "Matrix size " << n << "; # of threads " << numThreads << endl;
	//starting LU Decomposition 
	runtime = omp_get_wtime();
	ComputeGaussianElimination2(a,n, lwr);
	//initialize lower matrix's diagonal to 1s and upper matrix to 0s
	initDiagonalAndUpper(lwr, n);
	//forward substitution
	forwardSub(x2, b, lwr, n);

	/*cout << "x' vector" << endl;
	printVector(x2, n);
	cout << endl;*/

	//back substitution
	backSub(x, x2, a, n);

	/*cout << "x vector" << endl;
	printVector(x, n);
	cout << endl;*/

	runtime = omp_get_wtime() - runtime;
	//print computing time
	cout<< "LU Decomposition w/ forward/back substitution to solve for vector x runs in "	<< setiosflags(ios::fixed) 
												<< setprecision(2)  
												<< runtime << " seconds\n";

	//deallocate memory
	delete[] b;
	delete[] x2;
	delete[] x;

	for(int i=0;i<n;i++){
		delete[] a[i];
	}
	delete[] a;

	for(int i=0;i<n;i++){
		delete[] lwr[i];
	}
	delete[] lwr;
 	/*
	//for testing correctness
	float a[3][3] = {{1.0, 1.0, -1.0}, {1.0, -2.0, 3.0}, {2.0, 3.0, 1.0}};
	float b[3] = {4.0, -6.0, 7.0};
	cout << "Matrix A" << endl;
	printMatrix(a, n);
	cout << endl;
	
	//change 1st/2nd dimension size for custom size
    float lwr[N][N] = {0.0};
	
	//Compute the Gaussian Elimination for matrix a[n x n]
	runtime = omp_get_wtime();
	ComputeGaussianElimination2(a,n, lwr);
	
	cout << "Matrix U " << endl;
	printMatrix(a, n);
	cout << endl << "Matrix L " << endl;
	
	//initialize lower matrix's diagonal to 1s and upper matrix to 0s
	initDiagonalAndUpper(lwr, n);
	
	printMatrix(lwr, n);
	cout << endl;
	cout << "x' vector" << endl;
	//forward sub
	float x2[3] = {};// x' vector
	forwardSub(x2, b, lwr, n);
	for(int i =0; i< n; i++){
		cout << x2[i] << endl;
	}
	cout << endl << "x (solution vector)";
	//back sub
	float x[3] = {};// x, solution vector
	backSub(x, x2, a, n);
	cout << endl;
	for(int i =0; i< n; i++){
		cout << x[i] << endl;
	}
	
	runtime = omp_get_wtime() - runtime;
	//print computing time
		cout<< "LU Decomposition runs in "	<< setiosflags(ios::fixed) 
												<< setprecision(2)  
												<< runtime << " seconds\n";
	//for testing correctness
	*/
	
	return 0;
}
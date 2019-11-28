#ifndef UTILITY_H
#define UTILITY_H

typedef struct knnresult{
  int* nidx;
  double* ndist;
  int m;
  int k;
}knnresult;

// Partition using Lomuto partition scheme and parallel update of the initial index table
//validated
double* partition(int size,int* a,double* left, double* right, double* pivotIndex);

// Returns the k-th smallest element of list within left..right
// (i.e. left <= k <= right) while updating initial index table using lomuto partition
//validated
double* quickselect(int size,int* a,double* left, double* right, int k);

//quicksort used to reorder k smallest distances in ascending order while updating the index table
void quickSort(int* a, double* low, double* high);


//function to calculate matrix of distances. Size of result is mxn
void calculateDistances(double* X,double* Y,int n,int m,int d,double** pointer);

//function to calculate new knnresult struct initially
knnresult kNN(double* X,double* Y,int n,int m,int d,int k);

//functions for merging new with old index and distance tables 
void merge(double* arr1, double* arr2,int* ind1,int* ind2, int k,int pid);

//function to calculate new knnresult struct after given received index and points tables
void kNNMpi(double* X,double* Y,double* kNeighborsDist,int* indx ,int n,int m,int d,int k,knnresult* holder,int pid);

//function to calculate knnresult an pass points around in ring fashion using mpi protocol
//for communication in synchronous way
knnresult distrAllkNN(double* X,int n,int d,int k);

#endif // UTILITY_H_INCLUDED

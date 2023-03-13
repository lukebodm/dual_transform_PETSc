#include <petsc.h>
#include <petscvec.h>

int return_dual_basis();

int main(int argc, char **argv)
{
  Vec         v; 
  Vec         *a;           // array of input vectors
  Vec         *q;           // array of output (orthonormal vectors)
  Vec         *p;           // array of projected components
  PetscRandom rand;         // for setting random vectors
  PetscReal   dot, nrm, n;  // dot product, normalization, vector size
  PetscInt    i, j, k = 7;  // number of vectors to orthogonalize
  PetscInt    r = 32;       // seed for random generator
  MPI_Comm    comm;
  n = k;                    // vector lenghth = number of vectors (square matrix)

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  comm = PETSC_COMM_WORLD;

  // create random parameters
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetSeed(rand, r));
  PetscCall(PetscRandomSeed(rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  // create vector v 
  PetscCall(VecCreate(comm, &v));
  PetscCall(VecSetSizes(v, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(v));

  // create input and output array by duplicating vector
  PetscCall(VecDuplicateVecs(v, k, &a));
  PetscCall(VecDuplicateVecs(v, k, &q));
  PetscCall(VecDuplicateVecs(v, k, &p));

  // randomize values in input array
  for (i=0; i<k; ++i) {
    PetscCall(VecSetRandom(a[i], rand));
  }

  // iteratively run psuedocode outer loop
  // that orthogonalizes the vectors
  // algortihm from page 1050-1051
  PetscCall(VecNorm(a[0], NORM_2, &nrm));
  PetscCall(VecCopy(a[0],q[0]));
  PetscCall(VecScale(q[0], 1./(nrm*nrm)));
  PetscCall(VecCopy(a[0],p[0]));
    for (k=1; k<n; ++k) {
      // compute p_k
      PetscCall(VecCopy(a[k], v));
      for(j=0; j<=k-1; ++j) {
        PetscCall(VecDot(q[j], a[k], &dot));
        PetscCall(VecAXPY(v, -dot, p[j]));
      }
      PetscCall(VecCopy(v, p[k]));
      PetscCall(VecNorm(v, NORM_2, &nrm));
      PetscCall(VecScale(v, 1./(nrm*nrm)));
      PetscCall(VecCopy(v, q[k]));
    }

  // create view of output vectors
  for (i=0; i<k; ++i) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)q[i], "q_"));
    PetscCall(VecViewFromOptions(q[i], NULL, "-vec_view"));
  }

 // print the dot product of all combinations of vectors of q
  for (i=0; i<k; ++i){
    for (j=0; j<k; ++j) {
      PetscCall(VecDot(q[i], a[j], &dot));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "q%d . a%d: %g\n",i, j, (double)dot));
    }
  }

  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroyVecs(k, &a));
  PetscCall(VecDestroyVecs(k, &q));
  PetscCall(VecDestroyVecs(k, &p));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

int return_dual_basis() {
  return 0;
}

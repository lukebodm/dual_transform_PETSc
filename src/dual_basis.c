#include <petsc.h>
#include <petscvec.h>

static PetscErrorCode return_dual_basis(Vec *a, Vec *q, Vec *p, Vec v, PetscInt k, PetscInt n); 
static PetscErrorCode test_biorthogonality(Vec *q, Vec *a, PetscInt k);
static PetscErrorCode test_orthogonality(Vec *q, PetscInt k);

int main(int argc, char **argv)
{
  Vec         v; 
  Vec         *a;           // array of input vectors
  Vec         *q;           // array of output (orthonormal vectors)
  Vec         *p;           // array of projected components
  PetscRandom rand;         // for setting random vectors
  PetscReal   n;            // vector size
  PetscInt    i, k = 7;     // iterator and number of vectors to orthogonalize
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

  // create input, output, and temp array by duplicating vector
  PetscCall(VecDuplicateVecs(v, k, &a));
  PetscCall(VecDuplicateVecs(v, k, &q));
  PetscCall(VecDuplicateVecs(v, k, &p));

  // randomize values in input array
  for (i=0; i<k; ++i) {
    PetscCall(VecSetRandom(a[i], rand));
  }

  return_dual_basis(a, p, q, v, k, n);  

  test_biorthogonality(q,a,k);

  test_orthogonality(q,k);

  // create view of output vectors
  for (i=0; i<k; ++i) {
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)q[i], "q_"));
    PetscCall(VecViewFromOptions(q[i], NULL, "-vec_view"));
  }

  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroyVecs(k, &a));
  PetscCall(VecDestroyVecs(k, &q));
  PetscCall(VecDestroyVecs(k, &p));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode return_dual_basis(Vec *a, Vec *p, Vec *q, Vec v, PetscInt k, PetscInt n) {
 
  PetscReal   dot, nrm;  // dot product, normalization, vector size
  PetscInt    j;  // number of vectors to orthogonalize

  // algortihm from page 1037
  PetscCall(VecNorm(a[0], NORM_2, &nrm));
  PetscCall(VecCopy(a[0], q[0]));
  PetscCall(VecScale(q[0], 1./(nrm*nrm)));
  PetscCall(VecCopy(a[0],p[0]));
    for (k=1; k<n; ++k) {
      // compute p_k
      PetscCall(VecCopy(a[k], v));
      for(j=0; j<n; ++j) {
        PetscCall(VecDot(q[j], a[k], &dot));
        PetscCall(VecAXPY(v, -dot, p[j]));
      }
      PetscCall(VecCopy(v, p[k]));
      PetscCall(VecNorm(v, NORM_2, &nrm));
      PetscCall(VecScale(v, 1./(nrm*nrm)));
      PetscCall(VecCopy(v, q[k]));
    }
    for(k=n-2; k>=0; --k) {
      for(j=k+1; j<=n-1;++j ){
        PetscCall(VecDot(q[k],a[j],&dot));
        PetscCall(VecAXPY(q[k], -dot, q[j]));
      }
    }
    return 0;
}

static PetscErrorCode test_biorthogonality(Vec *q, Vec *a, PetscInt k) {
  PetscInt i, j;
  PetscReal dot;
  // print the dot product of all combinations of vectors of q
  printf("\n=== dual list orthogonality ===\n");
  for (i=0; i<k; ++i){
    for (j=0; j<k; ++j) {
      PetscCall(VecDot(q[i], a[j], &dot));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "q%d . a%d: %g\n",i, j, (double)dot));
    }
  }
  return 0;
}

static PetscErrorCode test_orthogonality(Vec *q, PetscInt k) {
  PetscInt i, j;
  PetscReal dot;
  // print the dot product of all combinations of vectors of q
  printf("\n==== dual vectors orthogonality ===\n");
  for (i=0; i<k; ++i){
    for (j=i; j<k; ++j) {
      PetscCall(VecDot(q[i], q[j], &dot));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "q%d . q%d: %g\n",i, j, (double)dot));
    }
  }
  return 0;
}

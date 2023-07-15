import random

import numpy as np

# Initialize the random seeds
import torch
# print("okk")
# np.random.seed(0)
# random.seed(0)
def generate_projection_matrix(num_client, feature_dim=256, share_dims=0, qr=True):
    """
    Project features (v) to subspace parametrized by A:

    Returns: A.(A^T.A)^-1.A^T
    """
    rank = (feature_dim - share_dims) / num_client
    rank = int(rank)
    assert num_client * rank <= (feature_dim - share_dims), "Feature dimension should be less than num_tasks * rank"

    # Generate ONBs
    if qr:
        print('Generating ONBs from QR decomposition')
        rand_matrix = np.random.uniform(size=(feature_dim, feature_dim))
        q, r = np.linalg.qr(rand_matrix, mode='complete')
    else:
        print('Generating ONBs from Identity matrix')
        q = np.identity(feature_dim)
    projections = []
    # print(q)
    for tt in range(num_client):
        offset = tt * rank
        A = np.concatenate((q[:, offset:offset + rank], q[:, feature_dim - share_dims:]), axis=1)
        proj = np.matmul(A, np.transpose(A))
        projections.append(proj)

    return projections

def unit_test_projection_matrices(projection_matrices):
    """
    Unit test for projection matrices
    """
    num_matrices = len(projection_matrices)
    feature_dim = projection_matrices[0].shape[0]
    rand_vetcor = np.random.rand(1, feature_dim)
    projections = []
    for tt in range(num_matrices):
        print('Task:{}, Projection Dims: {}, Projection Rank: {}'.format(tt, projection_matrices[tt].shape, np.linalg.matrix_rank(projection_matrices[tt])))
        projections.append((np.squeeze(np.matmul(rand_vetcor, projection_matrices[tt]))))
    print('\n\n ******\n Sanity testing projections \n********')
    for i in range(num_matrices):
        for j in range(num_matrices):
            print('P{}.P{}={}'.format(i, j, np.dot(projections[i], projections[j])))

if __name__ == '__main__':
    projections = generate_projection_matrix(num_client=3, feature_dim=512, qr=True)

    unit_test_projection_matrices(projections)

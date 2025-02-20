import numpy as np

def mis_decode_degree(predictions, adj_matrix):
  predictions = predictions.flatten()
  predictions -= 1/2000*np.array(adj_matrix.sum(axis=0)).flatten()
  solution = np.zeros_like(predictions.astype(int))

  # 원본 배열의 인덱스를 랜덤하게 섞습니다.
  random_indices = np.random.permutation(len(predictions))

  # 랜덤하게 섞인 인덱스로 원본 배열을 재정렬합니다.
  arr_random = predictions[random_indices]

  # 재정렬된 배열을 argsort()로 정렬합니다.
  indices = np.argsort(-arr_random)

  # 랜덤하게 섞인 인덱스를 다시 적용하여 원래의 순서로 되돌립니다.
  sorted_predict_labels = random_indices[indices]

  csr_adj_matrix = adj_matrix.tocsr()

  # time1 = time.perf_counter()
  for i in sorted_predict_labels:
    next_node = i

    if solution[next_node] == -1:
      continue

    solution[csr_adj_matrix[next_node].nonzero()[1]] = -1
    solution[next_node] = 1
  # time2 = time.perf_counter()
  # print('time for ', time2-time1)
  return (solution == 1).astype(int)


def mis_decode_np(predictions, adj_matrix):
  """Decode the labels to the MIS."""
  # import time
  predictions = predictions.flatten()
  solution = np.zeros_like(predictions.astype(int))

  # 원본 배열의 인덱스를 랜덤하게 섞습니다.
  random_indices = np.random.permutation(len(predictions))

  # 랜덤하게 섞인 인덱스로 원본 배열을 재정렬합니다.
  arr_random = predictions[random_indices]

  # 재정렬된 배열을 argsort()로 정렬합니다.
  indices = np.argsort(-arr_random)

  # 랜덤하게 섞인 인덱스를 다시 적용하여 원래의 순서로 되돌립니다.
  sorted_predict_labels = random_indices[indices]

  csr_adj_matrix = adj_matrix.tocsr()

  # time1 = time.perf_counter()
  for i in sorted_predict_labels:
    next_node = i

    if solution[next_node] == -1:
      continue

    solution[csr_adj_matrix[next_node].nonzero()[1]] = -1
    solution[next_node] = 1
  # time2 = time.perf_counter()
  # print('time for ', time2-time1)
  return (solution == 1).astype(int)

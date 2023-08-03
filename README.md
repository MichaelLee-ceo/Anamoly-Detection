## Anomaly Detection on Windmill Dataset
### Evaluation using reconstruction error of training data

### Threshold
| Metric | Reconstruction Error |
| :-: | :-: |
| Mean Error + 0.5 std| 0.3362 |
| Mean Error + 1 std | 0.3510 |

### Result
| Metric | High_out | Normal_out | Normal_in | Low_in |
| :-: | :-: | :-: | :-: | :-: |
| Mean Error + 0.5 std| 92.72% | 81.62% | 78.84% | 53.66% |
| Mean Error + 1 std | 88.56% | 78.18% | 72.59% | 49.85% |
| Upper Bound | 100% | 87.62% | 85.18% | 59.29% |


### Reconstruction on Normal Data
![image](./figures/windmill/0.png)

### Reconstruction on Abnormal Data
![image](./figures/windmill/3.png)
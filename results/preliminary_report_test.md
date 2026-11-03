# Preliminary results (test split, seed 42)

MovieLens-1M, 6040 users, 3416 movies, full-catalog ranking. One held-out movie per user, so Recall@K equals HitRate@K.

## Main results

| model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | best_epoch |
|---|---|---|---|---|---|
| popularity | 0.0353 | 0.0173 | 0.0637 | 0.0244 | - |
| deepfm | 0.0702 | 0.0352 | 0.1233 | 0.0485 | 22 |
| sasrec | 0.3098 | 0.1838 | 0.4224 | 0.2123 | 190 |
| ngcf | 0.0768 | 0.0387 | 0.1212 | 0.0498 | 45 |

## Headroom for per-user fusion

| what | Recall@10 | NDCG@10 |
|---|---|---|
| best single model (sasrec) | 0.3098 | 0.1838 |
| oracle: best of deepfm/sasrec/ngcf per user | 0.3667 | 0.2169 |

## Which models get each user right (top-10)

| models that hit (top-10) | users |
|---|---|
| deepfm | 124 |
| sasrec | 1531 |
| deepfm + sasrec | 96 |
| ngcf | 120 |
| deepfm + ngcf | 100 |
| sasrec + ngcf | 140 |
| deepfm + sasrec + ngcf | 104 |
| none | 3825 |

## Recall@10 by user history length (quartiles of training interactions)

| history length | users | popularity R@10 | deepfm R@10 | sasrec R@10 | ngcf R@10 | oracle R@10 |
|---|---|---|---|---|---|---|
| 16-42 | 1524 | 0.0427 | 0.1063 | 0.3911 | 0.1286 | 0.4554 |
| 43-94 | 1514 | 0.0310 | 0.0760 | 0.3554 | 0.0786 | 0.4102 |
| 95-205 | 1492 | 0.0328 | 0.0523 | 0.2714 | 0.0597 | 0.3345 |
| 206-2275 | 1510 | 0.0344 | 0.0457 | 0.2199 | 0.0397 | 0.2656 |

## Recall@10 by popularity of the held-out movie

| held-out movie | users | popularity R@10 | deepfm R@10 | sasrec R@10 | ngcf R@10 |
|---|---|---|---|---|---|
| head (top 20% popular) | 3411 | 0.0624 | 0.1105 | 0.3688 | 0.1114 |
| tail (other 80%) | 2629 | 0.0000 | 0.0179 | 0.2332 | 0.0320 |

## Popularity bias and coverage

For reference, 56.5% of held-out movies are in the head.

| model | head share of top-10 | catalog coverage@10 |
|---|---|---|
| popularity | 1.000 | 0.058 |
| deepfm | 0.893 | 0.548 |
| sasrec | 0.648 | 0.748 |
| ngcf | 0.808 | 0.683 |

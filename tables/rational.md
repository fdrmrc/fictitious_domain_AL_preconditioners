## Rational block preconditioner


- - - - - - - - - - - - 
$-\Delta u + u = f \text{ in } \Omega,$
$u= g  \text{ on } \Gamma,$
$u= 0  \text{ on } \partial \Omega$

$P\coloneqq\texttt{diag}(A+M,-\Delta^{s})$, with $s=\frac{1}{2}$.



| DoF number (background + immersed)        | #iter   |
| :---------------------------------------- | :------ |
| 1089+33                                  | 43 |
| 16641+129                                 | 39 |
| 66049+257                                 | 37 |
| 263169+513                                | 37 |
| 1050625+1025                              | 35 |
| 4223931+4097                              | 35 |
| 16836469+8193                             | 35 |


- - - - - - - - 

$-\Delta u = f \text{ in } \Omega,$
$u= g  \text{ on } \Gamma,$
$u= 0  \text{ on } \partial \Omega$

$P\coloneqq\texttt{diag}(A,-\Delta^{s})$, with $s=\frac{1}{2}$.




| DoF number (background + immersed)        | #iter   |
| :---------------------------------------- | :------ |
| 1089+33                                  | 30 |
| 4225+65                                   | 30 |
| 16641+129                                 | 28 |
| 66049+257                                 | 28 |
| 263169+513                                | 26 |
| 1050625+1025                              | 28 |
| 1063467+2049                              | 28 |
| 4223931+4097                              | 28 |
| 16836469+8193                             | 28 |
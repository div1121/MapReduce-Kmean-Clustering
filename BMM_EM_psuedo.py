"""
Algorithm:

Definition: sum(x,i=1,i=k) -> summation of x from i=1 to i=k
Definition: product(x,i=1,i=k) -> product of x from i=1 to i=k
D: dimension of feature
K: number of cluster

1. Initialize the parameters: π(k) of K Bermoulis component, q(k) parameter of each Bernoullis distribution
variable Bnn: parameters of Bermoulis component -> shape = (K, D)
variable weigthing: K Bermoulis component -> shape = (K, 1)

For each element in Bnn: random value between 0 to 1
For each elmenent in weighting: all value betwwen 0 to 1 and sum of values = 1

2. E Step: assign each point x(n) an assignment score γ(z(n,k)) for each cluster k (Mapper)
γ(z(n,k)) = π(k)p(x(n)|q(k)) / sum(π(j)p(x(n)|q(j)),j=1,j=K) -> scalar
x(n): n-th data point -> shape: (1, D),  π(k): weighting[k], q(k): bnn[k]
where p(x|q) = product(q(i)^x(i) * (1-q(i))^(1-x(i)),i=1,i=D), x(i) in {0,1} -> scalar

For each i-th data point in the dataset: compute γ(z(n,k)) for each k Bernoullis distribution with weighting

since γ(z(n,k)) can help to update parameters π(k) and q(k), also we need to keep track of maximum likelihood: sum(log(sum(γ(z(n,k),k=1,k=K)),n=1,n=N)
MapReduce approach:
New_BNN = shape(K,D), New_weighting = shape(K,1), Counting = shape(K,1), Likelihood_sum = 0
For each i-th data point in the dataset:
    sum = 0
    For each k cluster:
        compute γ(z(n,k))
        New_BNN[k] += γ(z(n,k)) * x(i) -> x(i): i-th data point
        New_weighting[k] += γ(z(n,k))
        Counting[k] += 1
        sum += γ(z(n,k))
    Likelihood_sum += log(sum)
return New_BNN, New_weighting, Counting, Likelihood_sum 

3. M Step: For each cluster k, update new parameter π(k) and q(k) based on γ(z(n,k)) (reducer)
For each cluster k:
π(k) = sum(γ(z(n,k)), n=1, n=N) / N
q(k) = sum(γ(z(n,k)) * x(n), n=1, n=N) / sum(γ(z(n,k)), n=1, n=N)
x(n): n-th data point -> shape: (1, D)

Based on above New_BNN, New_weighting, Counting:
    For each k cluster:
        BNN[k] = New_BNN[k] / New_weighting[k]
        Weighting[k] = New_weighting[k] / Counting[k]

4. Evaluate Likelhood value 
If likelihood value or parameter converge (maximization), stop; Otherwise Goto Step 2
likelihood value: sum(log(sum(γ(z(n,k),k=1,k=K)),n=1,n=N)

From above parameter
Likelihood_sum save the likelihood value of dataset
while the increase of Likelihood_sum >= 0.1:
    Goto Step 2

"""
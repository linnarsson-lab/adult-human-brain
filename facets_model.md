## Facet learning
### Quantitative modelling of the transcriptome

In previous work, we and others have focused on partitioning cells in non-overlapping subsets representing distinct 'cell types'. Implicitly, we have worked with a mental model where each cell has a single identity, and there are a finite number of distinct identities. We have used clustering algorithms to discover cell types, again with the implicit assumption that types would show unambiguous differences in gene expression.

However, there are several situations where this framework has failed. One clear example involves the sex-specific genes (e.g. *Xist* and *Tsix*), which are expressed exclusively in female cells undergoing X chromosome inactivation. These genes are co-expressed in a distinct subset of cells (those from females), but appear completely independent of any other cell type-specific genes. Thus if we apply clustering algorithms to datasets containing both male and female cells, there is a tension between partitioning cells by sex or by cell type. What typically happens is that every cell type gets split in male and female version at the end of clustering (because there are very few sex-specific genes and they tend not to dominate clustering).

Another example are the cell cycle genes, which are expressed in any cell currently undergoing cell division. But since cells of disparate types are capable of cell divison, again the cell cycle genes are expressed independently across many types of cells. Attempting to cluster cells, there is a tension between grouping cells by type, or by cell cycle state. Here, what typically happens is that all cycling cells get lumped into a single cluster, which is then subdivided by cell type. This is because there are hundreds of highly expressed cell cycle-specific genes and they tend to dominate clustering.

A subtler example involves activity-dependent gene expression in neurons. Some genes (e.g. *Fos*) tend to be transiently induced in response to neuronal activity. In clustering-based analyses, in principle every type of neuron could be split into active and inactive subsets. But since there are an intermediate number of activity-dependent genes, and they tend to vary in a graded rather than all-or-nothing manner, the outcome is variable and results can be highly confusing.

What we would like to see instead is a partitioning of cells along multiple independent dimensions, or *facets*. Each facet would capture one important axis of variation, and each cell would be described by its classification in each of the different facets. For example, a (1) recently activated (2) inhibitory neuron, (3) not cycling, from a (4) female animal. Thus facets are a multidimensional classification that captures more than a single aspect of cellular identity.

### A facets model of gene expression

In order to account for the multiple facets of single cells, we introduce a model that simultaneously (1) discovers sets of genes salient to each of several facets, (2) clusters cells along the dimension of each facet. This probabilistic model automatically and simultaneously partitions cells along multiple facets. For example, it can simultaneously infer the sex, cell cycle state and type of each cell, as well as the most salient genes indicative of each. The model is strongly inspired by the clustering algortihm ProMMT (REF, Kenneth Harris) and ProMMT can be seen as a special case of facet learning where there is only one facet. 

We model gene expression as a mixture of negative binomial distributions. The probability of observing an expression profile $x_c$ when a cell c is assigned to cluster $k_{c,i}$ of facet $i$ is:

$$ P(x_c|k_{c,1}, k_{c,2}, ..., k_{c,i}) = \prod_g{
    \begin{cases}
    P(x_{c,g}|\mu_{g,0}) & g \notin \bigcup S_{i} \\
    P(x_{c,g}|\mu_{g,k_{c,1}}) & g \in S_1 \\
    P(x_{c,g}|\mu_{g,k_{c,2}}) & g \in S_2 \\
    ... \\
    P(x_{c,g}|\mu_{g,_{c,i}}) & g \in S_i
    \end{cases}
} $$

Here, each $S_i$ is the set of genes assigned to each facet, $k_{c,i}$ is the cluster assignment of cell $c$ for facet $i$, $\mu_{g,k_{c,i}}$ is the mean expression of gene $g$ in the set of cells assigned to the same cluster as $c$, and $\mu_{g,0}$ is the mean for gene $g$ across all cells.

Genes that are assigned to a facet are modelled with one mean per cluster of that facet. Thus a gene will always fit the data better if it is assigned to more clusters, and it will fit better the more those clusters correspond to actual differences in expression. Genes that are not assigned to any facet are modelled with the single overall mean $\mu_{g,0}$. 

Note that each gene is assigned to at most one facet, and that facets are thus mutually exclusive. For example, in this model a gene cannot be simultaneously assigned to *cell cycle* and *neuronal activity*. 

In order to force the model to learn salient genes and to partition the cells according to real differences in expression of those salient genes, the model must be regularized. We do this by allowing only a limited number of genes to participate in facets, i.e. by limiting the total size (cardinality) of the sets $S_i$. The effect is to create a competition where only the genes that gain the most by being modelled by cluster-specific expression patterns are allowed to contribute. This biases the clustering towards binary-like differences, rather than graded differences.


### Learning facets from data

**E step**

In the E step, we reassign each cell to a cluster, independently for each facet, given the current sets of genes assigned to each facet, and the current assignments of cells to clusters.

For cell and each facet, we can calculate the posterior log-probability of the observations $x_c$ given that the cell $c$ is assigned to cluster $k_{c,i}$ of that facet:

$$ log P(k_{i}|x_c) = const. + log(\pi_{k_i}) + \sum_{g \in S_i} x_{c,g} log(p_{g,k}) + r log(1 - p_{g,k})$$

The constant term includes the contribution from genes that do not belong to any of the facets, as well as the binomial term from the negative binomial distribution, none of which depend on $k$. The $p_{g,k}$ are the estimates of the $p$ parameter of the negative binomial distribution, given by 

$$ \mu_{g,k_i}/(r + \mu_{g,k_i}) $$

where $\mu_{g,k_i}$ is the mean of gene $g$ taken across the cells currently assigned to cluster $k_i$ of facet $i$.

We implement 'hard EM' by calculating the probabilities above, then assigning each cell to the single cluster (for each facet) with maximal probability.

**M step**

In the M step, we reselect the salient genes for each facet, given the current assignments of cells to clusters in each facet.

First, for each facet $i$, we calculate the log-likelihood gain obtained when a gene is allowed to vary between clusters:

$$

Y_{g,i} = \sum_c{x_{c,g}(log(p_{g,k})-log(p_{g,0})) + r (log(1 - p_{g,k}) - log(1 - p_{g,0}))}

$$

This expression simply adds the log-likelihood estimates obtained when the gene is allowed to vary by clusters and subtracts the estimate obtained without using clusters, for each cell $c$. Note that this expression can be calcuated independently for each gene. 

*Constraints on gene assignment*

The maximum likelihood fit would be obtained by selecting the set of genes for which the likelihood gain is greatest. However, recall that facets are mutually exclusive, so that each gene can only be assigned to a single facet. To account for these constraints, genes must be allocated to facets in mutually exclusive combinations such that the overall likelihood gain is maximized. This can be achieved by solving the *generalized assignment problem*, which is unfortunately known to be NP-hard. However there are greedy approximate algorithms that work well in practice. Here, we use the fully polynomial-time approximation scheme (FPTAS) to solve a Knapsack problem for each facet individually, then a greedy algorithm to reassign genes to facets. See https://github.com/madcat1991/knapsack and https://en.wikipedia.org/wiki/Knapsack_problem

As a result, each M step results in the reassignment of genes to facets such that the likelihood is maximally improved, given the current assignment of cells to clusters.

### Learning the number of clusters

For some facets, the number of clusters is fixed. For example, a facet for *sex* would be set to have two clusters, and *cell cycle* might have five clusters (for G0, G1, S, G2 and M). In other cases, the number of clusters is unknown and must be learned from the data. In those cases, we use the same splitting heuristics as used in ProMMT, and select the best-fitting model using the BIC score.

### From facets to grammar

We have observed that genes are often expressed in ways that suggest a grammatical structure to the transcriptome. For example, we often observe nested expression, where a large set of genes define a broad category, and more specific classes are defined by the additive expression of genes on this common background. There are also cases of mutual exclusion, where genes are never expressed in the same set of cells, and of independence, where sets of genes are expressed in partially overlapping sets of cells regardless of cell type. 

To learn these kinds of structures, we introduce the *nested* facet. A nested facet applies only to a single cluster defined by another facet. For example, this might be used to model neurotransmitter genes, which are active only in neurons. That is, the facet for neurotransmitters would apply only to cells clustered as 'neurons' by another facet.

Thus at the top level we have a set of independent facets. Each independent facet is associated with a zero or more nested facets. 

The complete model for the adolescent mouse brain could be defined as follows in JSON syntax:

```
model {
    name: 'adolescent mouse brain',
    facets: [
        { 
            name: 'cell cycle',
            k: 5,
            clamped: ['Cdk1', 'Top2a']
        },
        {
            name: 'sex',
            k: 2,
            clamped: ['Xist', 'Tsix']
        },
        {
            name: 'cell type',
            k: [5, 25],
            nested: [
                {
                    where: ['Stmn3 > 0.1'],
                    name: 'neurotransmitter',
                    k: 8,
                    clamped: ['Gad1', 'Gad2', 'Tph2', ...]
                }
            ]
        }
    ]
}
```
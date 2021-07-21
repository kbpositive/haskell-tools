# Iris-classifier
A two layer classifier made to solve the iris dataset.

Feed Forward
$$\mathbf{o} = \theta(\mathbf{i}\mathbf{W})$$
$$\mathbf{o}_2 = \theta(\mathbf{o} \mathbf{W}_2)$$
$$\mathbf{o}_n = \theta(\mathbf{o}_{n-1} \mathbf{W}_n)$$

SGDC
$$\mathbf{u} = (\mathbf{o}_n-\mathbf{t}) \circ \theta'(\mathbf{o}_n)$$
$$\mathbf{u} = (\mathbf{u}\mathbf{W}^T_n) \circ \theta'(\mathbf{o}_{n-1})$$
$$\mathbf{u} = (\mathbf{u}\mathbf{W}^T_3) \circ \theta'(\mathbf{o}_2)$$
$$\mathbf{u} = (\mathbf{u}\mathbf{W}^T_2) \circ \theta'(\mathbf{o})$$
$$\mathbf{W} := \mathbf{W} - (\mathbf{i}^T \otimes \mathbf{u})$$


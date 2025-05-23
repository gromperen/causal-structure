import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""A Linear Non-Gaussian Acyclic Model for Causal Discovery: https://www.cs.helsinki.fi/group/neuroinf/lingam/JMLR06.pdf""")
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import FastICA
    from scipy.optimize import linear_sum_assignment
    import itertools
    return FastICA, itertools, linear_sum_assignment, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Identifiability of Non-Gaussianity

    $$
    \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} 
    = \begin{bmatrix} 0 & 0 \\ 0.8 & 0 \end{bmatrix}
    \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} 
    + \begin{bmatrix} e_1 \\ e_2 \end{bmatrix}
    $$

    $$ \mathbf{x} =  B\mathbf{x}+\mathbf{e} \implies (I - B)\mathbf{x} = \mathbf{e} \implies \mathbf{x} = (I - B)^{-1}\mathbf{e}$$
    """
    )
    return


@app.cell
def _(mo, np):
    def sample_dist(n_samples, dist="uniform"):
        if dist == "uniform":
            return np.random.uniform(low=-1, high=1, size=n_samples)
        elif dist == "exponential":
            return np.random.exponential(scale=1.0, size=n_samples) - 1.0
        elif dist == "laplace":
            return np.random.laplace(loc=0.0, scale=1.0, size=n_samples)
        elif dist == "normal":
            return np.random.normal(loc=0, scale=1, size=n_samples)

        raise ValueError(f"Dont know about distribution {dist}")

    @mo.cache
    def gen_x(B, e):
        n = B.shape[0]
        I = np.eye(n)
        x = np.linalg.inv(I - B) @ e
        return x    
    return gen_x, sample_dist


@app.cell(hide_code=True)
def _(gen_x, mo, np, sample_dist):
    @mo.cache
    def plot_xs(ax, B, n_samples=1000, dist1="unif", dist2="unif"):
        e1 = sample_dist(n_samples, dist1)
        e2 = sample_dist(n_samples, dist2)

        e = np.vstack([e1, e2])
        x = gen_x(B, e)
        x1 = x[0]
        x2 = x[1]

        ax.scatter(x1, x2)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")    

    return (plot_xs,)


@app.cell(hide_code=True)
def _(mo):
    n_samples_slider = mo.ui.slider(5, 5000, value=1000, label="n")
    e1_dist_choice = mo.ui.dropdown(["uniform", "exponential", "laplace", "normal"], value="uniform", label="e1 distribution")
    e2_dist_choice = mo.ui.dropdown(["uniform", "exponential", "laplace", "normal"], value="uniform", label="e2 distribution")

    b1_matrix_input = mo.ui.text_area(
        value="[[0, 0], [0.8, 0]]",
        label="B_1 matrix (Python list syntax)"
    )


    b2_matrix_input = mo.ui.text_area(
        value="[[0, 0.8], [0, 0]]",
        label="B_2 matrix (Python list syntax)"
    )

    mo.md(
        f"""
        **Number of Samples.**
        {n_samples_slider}

        {e1_dist_choice}

        {e2_dist_choice}

        {b1_matrix_input} {b2_matrix_input}
        """
    )
    return (
        b1_matrix_input,
        b2_matrix_input,
        e1_dist_choice,
        e2_dist_choice,
        n_samples_slider,
    )


@app.cell(hide_code=True)
def _(
    b1_matrix_input,
    b2_matrix_input,
    e1_dist_choice,
    e2_dist_choice,
    n_samples_slider,
    np,
    plot_xs,
    plt,
):
    n_samples = n_samples_slider.value
    B1 = np.array(eval(b1_matrix_input.value))
    B2 = np.array(eval(b2_matrix_input.value))
    e1_dist = e1_dist_choice.value
    e2_dist = e2_dist_choice.value

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    plot_xs(ax1, B1, n_samples=n_samples, dist1=e1_dist, dist2=e2_dist)
    plot_xs(ax2, B2, n_samples=n_samples, dist1=e1_dist, dist2=e2_dist)

    plt.tight_layout()
    plt.show()
    return (n_samples,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Lingam Discovery Algorithm

    Section 4, Algorithm A: https://www.cs.helsinki.fi/group/neuroinf/lingam/JMLR06.pdf

    Adapted from https://github.com/cdt15/lingam
    """
    )
    return


@app.cell
def _(
    FastICA,
    est_causal_order,
    gen_x,
    linear_sum_assignment,
    n_samples,
    np,
    sample_dist,
):
    # Setup the data. Want to learn B

    B = np.array([
        [0, 0.8], 
        [0, 0]
    ])

    e1 = sample_dist(n_samples=n_samples, dist="uniform")
    e2 = sample_dist(n_samples=n_samples, dist="uniform")

    e = np.vstack([e1, e2])

    X = gen_x(B, e).T

    # Normalise x
    X = X - X.mean(axis=0, keepdims=True)


    ica = FastICA(whiten="unit-variance", random_state=42)
    ica.fit(X)
    W_ica = ica.components_


    # Step 2. Find only permutation of cols which yields matrix without 0 on diagonal.
    # In practice, all elements will be nearly non-zero. Hence minimise sum of |1/ W_{ii}|

    _, col_ind = linear_sum_assignment(1 / np.abs(W_ica))

    W_ica[col_ind] = W_ica

    diag_elems = np.diag(W_ica)

    W_est = W_ica / diag_elems[:, np.newaxis]

    B_est =  np.eye(W_est.shape[0]) - W_est

    # Step 5. Causal Order estimation
    # Want P s.t. B = P B_est P^T  is approx. lower triangular
    # minimise sum of B_ij^2 for i<=j (sum of upper triangle including diag)

    P = est_causal_order(B_est)
    print(B_est)
    print(P)

    B_est = P @ B_est @ P.T
    return


@app.cell
def _(itertools, np):
    def est_causal_order(B):
        # Exhaustive permutation search. Is O(d!)
        d = B.shape[0]
        perms = list(itertools.permutations(range(d)))
        best_score = float("inf")
        best_perm = None

        for perm in perms:
            P = np.eye(d)[list(perm)]
            B_perm = P @ B @ P.T
            score = np.sum(np.triu(B_perm)**2)
            if score < best_score:
                best_perm = perm
                best_score = score

        P_best = np.eye(d)[list(best_perm)]
        return P_best
    return (est_causal_order,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

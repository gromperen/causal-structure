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
    return mo, np, plt


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


@app.cell(hide_code=True)
def _(np):
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
    return (sample_dist,)


@app.cell
def _(mo, np, plt, sample_dist):
    @mo.cache
    def plot_xs(B, n_samples=1000, dist1="unif", dist2="unif"):
        e1 = sample_dist(n_samples, dist1)
        e2 = sample_dist(n_samples, dist2)

        I = np.eye(2)
        e = np.vstack([e1, e2])
        x = np.linalg.inv(I - B) @ e
        x1 = x[0]
        x2 = x[1]

        fig, ax = plt.subplots()
        ax.scatter(x1, x2)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")    
        return fig
    return (plot_xs,)


@app.cell(hide_code=True)
def _(mo):
    n_samples_slider = mo.ui.slider(100, 5000, value=1000, label="n")
    e1_dist_choice = mo.ui.dropdown(["uniform", "exponential", "laplace", "normal"], value="uniform", label="e1 distribution")
    e2_dist_choice = mo.ui.dropdown(["uniform", "exponential", "laplace", "normal"], value="uniform", label="e2 distribution")

    b_matrix_input = mo.ui.text_area(
        value="[[0, 0], [0.8, 0]]",
        label="B matrix (Python list syntax)"
    )

    mo.md(
        f"""
        **Number of Samples.**
        {n_samples_slider}

        {e1_dist_choice}

        {e2_dist_choice}

        {b_matrix_input}
        """
    )
    return b_matrix_input, e1_dist_choice, e2_dist_choice, n_samples_slider


@app.cell
def _(
    b_matrix_input,
    e1_dist_choice,
    e2_dist_choice,
    n_samples_slider,
    np,
    plot_xs,
):
    n_samples = n_samples_slider.value
    B = np.array(eval(b_matrix_input.value))
    e1_dist = e1_dist_choice.value
    e2_dist = e2_dist_choice.value

    plot_xs(B, n_samples=n_samples, dist1=e1_dist, dist2=e2_dist)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

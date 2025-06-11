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

    import lingam 
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
    def sample_dist(n_samples, dist="uniform", var=1):
        if dist == "uniform":
            return np.random.uniform(low=-1, high=1, size=n_samples)
        elif dist == "exponential":
            return np.random.exponential(scale=1.0, size=n_samples) - 1.0
        elif dist == "laplace":
            return np.random.laplace(loc=0.0, scale=1.0, size=n_samples)
        elif dist == "normal":
            return np.random.normal(loc=0, scale=var, size=n_samples)

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
    def plot_xs(ax, B, n_samples=1000, dist1="unif", dist2="unif", vars=[1,1]):
        e1 = sample_dist(n_samples, dist1, var=vars[0])
        e2 = sample_dist(n_samples, dist2, var=vars[1])

        e = np.vstack([e1, e2])
        x = gen_x(B, e)
        x1 = x[0]
        x2 = x[1]

        ax.scatter(x1, x2)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")    

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

    plot_xs(ax1, B1, n_samples=n_samples, dist1=e1_dist, dist2=e2_dist, vars=[1,0.36])
    plot_xs(ax2, B2, n_samples=n_samples, dist1=e1_dist, dist2=e2_dist, vars=[0.36, 1])

    ax1.set_ylim([-4, 4])
    ax1.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])
    ax2.set_xlim([-4, 4])

    plt.tight_layout()
    #plt.show()

    plt.savefig("identifiability_gaussian.png", dpi=400)
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
        [0, 0.8, 0.1], 
        [0, 0, 0.2],
        [0, 0, 0]
    ])

    ''' B = np.array([
        [0, 0.8, 0.4, 0.5], 
        [0, 0, 0.7, 0],
        [0, 0, 0, 0.9],
        [0, 0, 0, 0]
    ]).T

    B = np.array([
        [0, 0.8, 0], 
        [0, 0, 0],
        [0.7, 0.6, 0]
    ])
    '''
    e1 = sample_dist(n_samples=n_samples, dist="uniform")
    e2 = sample_dist(n_samples=n_samples, dist="uniform")
    e3 = sample_dist(n_samples=n_samples, dist="uniform")
    e4 = sample_dist(n_samples=n_samples, dist="uniform")



    e = np.vstack([e1, e2, e3])

    X1 = gen_x(B, e).T

    # Normalise x
    X1 = X1 - X1.mean(axis=0, keepdims=True)


    ica = FastICA(whiten="unit-variance", random_state=425)
    ica.fit(X1)
    W_ica = ica.components_

    print(np.round(W_ica, 2))


    # Step 2. Find only permutation of cols which yields matrix without 0 on diagonal.
    # In practice, all elements will be nearly non-zero. Hence minimise sum of |1/ W_{ii}|

    _, col_ind = linear_sum_assignment(1 / np.abs(W_ica))

    W_ica[col_ind] = W_ica

    print(np.round(W_ica, 2))

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
    # Fetch the Parkinson's telemonitoring dataset
    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    # Fetch dataset
    parkinsons_telemonitoring = fetch_ucirepo(id=189)

    # Extract data
    X = parkinsons_telemonitoring.data.features
    y = parkinsons_telemonitoring.data.targets

    print("Dataset Information:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of targets: {y.shape[1]}")

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget columns:")
    print(y.columns.tolist())

    print("\nFirst few rows of features:")
    print(X.head())

    print("\nFirst few rows of targets:")
    print(y.head())

    print("\nDataset statistics:")
    print(X.describe())

    print("\nTarget statistics:")
    print(y.describe())
    return X, pd, warnings, y


@app.cell
def _(X, pd, y):
    # Check for missing values and create complete dataset
    print("Missing values in features:")
    print(X.isnull().sum())

    print("\nMissing values in targets:")
    print(y.isnull().sum())

    # Combine features and targets for causal analysis
    # We'll include both UPDRS scores as part of the causal graph
    data_combined = pd.concat([X, y], axis=1)

    print(f"\nCombined dataset shape: {data_combined.shape}")
    print(f"Total variables for causal analysis: {data_combined.shape[1]}")

    # Check unique subjects to understand longitudinal structure
    print(f"\nDataset includes longitudinal data from patients over time")
    print(f"Age range: {X['age'].min()} - {X['age'].max()} years")
    print(f"Time range: {X['test_time'].min():.1f} - {X['test_time'].max():.1f} days from baseline")
    print(f"Gender distribution: {X['sex'].value_counts().to_dict()} (0=male, 1=female)")

    # Look at correlations between voice features and UPDRS scores
    voice_features = [col for col in X.columns if col not in ['age', 'test_time', 'sex']]
    print(f"\nVoice features for analysis: {len(voice_features)}")
    print(voice_features)

    # Check correlations
    correlations_motor = data_combined[voice_features + ['motor_UPDRS']].corr()['motor_UPDRS'].drop('motor_UPDRS')
    correlations_total = data_combined[voice_features + ['total_UPDRS']].corr()['total_UPDRS'].drop('total_UPDRS')

    print("\nTop voice features correlated with motor_UPDRS:")
    print(correlations_motor.abs().sort_values(ascending=False).head(10))

    print("\nTop voice features correlated with total_UPDRS:")
    print(correlations_total.abs().sort_values(ascending=False).head(10))
    return (data_combined,)


@app.cell
def _(data_combined, pd, warnings):
    # Prepare data for causal discovery
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    import networkx as nx
    from lingam.utils import make_dot
    warnings.filterwarnings('ignore')

    # For causal analysis, we'll focus on the core variables:
    # 1. Key voice features (based on literature and correlations)
    # 2. UPDRS scores  
    # 3. Some demographic variables

    # Select key variables for causal analysis
    key_voice_features = [
        'PPE',           # Fundamental frequency variation
        'HNR',           # Harmonics-to-noise ratio
        'RPDE',          # Nonlinear dynamical complexity
        'DFA',           # Signal fractal scaling
        'NHR',           # Noise-to-harmonics ratio
        'Jitter(%)',     # Frequency variation
        'Shimmer',       # Amplitude variation
        'Shimmer(dB)',   # Amplitude variation in dB
        'RPDE'           # Recurrence period density entropy
    ]

    # Remove duplicate RPDE
    key_voice_features = list(set(key_voice_features))

    # Add demographic and clinical variables
    analysis_variables = key_voice_features + ['age', 'sex', 'motor_UPDRS', 'total_UPDRS']

    # Create analysis dataset
    causal_data = data_combined[analysis_variables].copy()

    print(f"Selected variables for causal analysis: {len(analysis_variables)}")
    print(analysis_variables)

    # Check data distribution (non-Gaussianity is important for LiNGAM)
    print("\nSkewness of variables (non-Gaussian is good for LiNGAM):")
    for var in analysis_variables:
        skewness = causal_data[var].skew()
        print(f"{var}: {skewness:.3f}")

    # Standardize the data
    scaler = StandardScaler()
    causal_data_scaled = pd.DataFrame(
        scaler.fit_transform(causal_data),
        columns=causal_data.columns
    )

    print(f"\nScaled data shape: {causal_data_scaled.shape}")
    print("Data ready for causal discovery algorithms")

    # Save the prepared data
    causal_data_scaled.to_csv('parkinsons_causal_data.csv', index=False)
    print("Data saved to 'parkinsons_causal_data.csv'")
    return


@app.cell
def _(X, pd, y):
    def _():
        # Apply ICA-LiNGAM and DirectLiNGAM algorithms
        from lingam import ICALiNGAM, DirectLiNGAM
        import time

        print("Applying causal discovery algorithms to Parkinson's dataset")
        print("=" * 60)

        # Load the prepared data
        data = pd.read_csv('parkinsons_causal_data.csv')
        variables = list(data.columns)

        print(f"Dataset shape: {data.shape}")
        print(f"Variables: {variables}")

        # 1. ICA-LiNGAM Algorithm
        print("\n1. Running ICA-LiNGAM...")
        start_time = time.time()

        # Initialize and fit ICA-LiNGAM
        ica_model = ICALiNGAM(random_state=42, max_iter=1000)
        ica_model.fit(data.values)

        ica_time = time.time() - start_time

        print(f"ICA-LiNGAM completed in {ica_time:.2f} seconds")
        print(f"Causal order found: {[variables[i] for i in ica_model.causal_order_]}")

        # Extract results
        ica_adjacency = ica_model.adjacency_matrix_
        ica_causal_order = ica_model.causal_order_

        # 2. DirectLiNGAM Algorithm  
        print("\n2. Running DirectLiNGAM...")
        start_time = time.time()

        # Initialize and fit DirectLiNGAM
        direct_model = DirectLiNGAM(random_state=42)
        direct_model.fit(data.values)

        direct_time = time.time() - start_time

        print(f"DirectLiNGAM completed in {direct_time:.2f} seconds")
        print(f"Causal order found: {[variables[i] for i in direct_model.causal_order_]}")

        # Extract results
        direct_adjacency = direct_model.adjacency_matrix_
        direct_causal_order = direct_model.causal_order_

        # 3. Compare Results
        print("\n3. Comparing Results")
        print("=" * 30)

        # Function to get significant causal relationships
        def get_causal_relationships(adjacency_matrix, variables, threshold=0.1):
            relationships = []
            n_vars = len(variables)
            for i in range(n_vars):
                for j in range(n_vars):
                    if abs(adjacency_matrix[i, j]) > threshold:
                        relationships.append({
                            'from': variables[j],  # Column index -> row variable
                            'to': variables[i],    # Row index
                            'strength': adjacency_matrix[i, j]
                        })
            return relationships

        # Get relationships from both methods
        ica_relationships = get_causal_relationships(ica_adjacency, variables, threshold=0.1)
        direct_relationships = get_causal_relationships(direct_adjacency, variables, threshold=0.1)

        print(f"ICA-LiNGAM found {len(ica_relationships)} significant causal relationships")
        print(f"DirectLiNGAM found {len(direct_relationships)} significant causal relationships")

        # Print strongest relationships for each method
        print("\nTop 10 strongest causal relationships - ICA-LiNGAM:")
        ica_relationships_sorted = sorted(ica_relationships, key=lambda x: abs(x['strength']), reverse=True)
        for i, rel in enumerate(ica_relationships_sorted[:10]):
            print(f"{i+1:2d}. {rel['from']} → {rel['to']} (strength: {rel['strength']:.3f})")
        print("Missing values in features:")
        print(X.isnull().sum())

        print("\nMissing values in targets:")
        print(y.isnull().sum())

        # Combine features and targets for causal analysis
        # We'll include both UPDRS scores as part of the causal graph
        data_combined = pd.concat([X, y], axis=1)

        print(f"\nCombined dataset shape: {data_combined.shape}")
        print(f"Total variables for causal analysis: {data_combined.shape[1]}")

        # Check unique subjects to understand longitudinal structure
        print(f"\nDataset includes longitudinal data from patients over time")
        print(f"Age range: {X['age'].min()} - {X['age'].max()} years")
        print(f"Time range: {X['test_time'].min():.1f} - {X['test_time'].max():.1f} days from baseline")
        print(f"Gender distribution: {X['sex'].value_counts().to_dict()} (0=male, 1=female)")

        # Look at correlations between voice features and UPDRS scores
        print("\nTop 10 strongest causal relationships - DirectLiNGAM:")
        direct_relationships_sorted = sorted(direct_relationships, key=lambda x: abs(x['strength']), reverse=True)
        for i, rel in enumerate(direct_relationships_sorted[:10]):
            print(f"{i+1:2d}. {rel['from']} → {rel['to']} (strength: {rel['strength']:.3f})")
        return None


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

# High Probability Guarantees for Random Reshuffling

This repository contains the code used to produce all the plots in the paper [High Probability Guarantees for Random Reshuffling](https://arxiv.org/pdf/2311.11841).

## Run the Experiments

1. Clone the repository and navigate to the `rr_vs_sgd` directory:
    ```sh
    cd rr_vs_sgd
    ```

2. Set the desired hyperparameters in the [config.yaml](rr_vs_sgd/config.yaml) file.

3. Run the experiments comparing RR against SGD using the provided shell script:
    ```sh
    ./run.sh
    ```

4. Run the experiments verifying Lipschitz conditions for RR using the following command:
    ```sh
    python verify_rr_conditions.py
    ```

## Plotting Results

1. Generate histograms of the full gradient norms after the same number of iterations for RR and SGD, and compare the number of iterations needed by RR and SGD to reach the same accuracy:
    ```sh
    python plot_rr_vs_sgd.py rr_vs_sgd
    ```

2. Generate the estimated Lipschitz constants along the trajectory of RR:
    ```sh
    python plot_lipschitz_estimate.py estimate_lipschitz_parameter
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
